"""
tracker_diagnostics.py  --  characterize an event stream the way the spectral
holonomic tracker "sees" it, so real data can be diffed against the unit-test
synthetic data to find what actually differs.

USAGE
-----
    from subhkl.streaming.loader import EventStreamLoader
    from tracker_diagnostics import analyze_event_stream, compare

    # real data
    loader = EventStreamLoader(nexus, instrument, ki_vec, sample_offset,
                               gonio_axes=axes, gonio_names=names, gonio_offsets=offs)
    real = analyze_event_stream(loader.get_batches(10000),
                                gonio_axes=axes, gonio_offsets=offs, label="REAL")

    # synthetic (same harness you feed the tests)
    synth = analyze_event_stream(get_fake_batches(sim_data, batch_size=10000),
                                 label="SYNTH")

    compare(synth, real)

WHAT IT REPORTS (all are things the tracker's behaviour depends on)
-------------------------------------------------------------------
  * stream basics: N, rate (Hz), batch span, timestamp units sanity
  * vector sanity: |q| distribution, ki normalization & directional spread
  * goniometer motion: per-axis range, and PER-BATCH sweep (validity of the
    mean-R_batch / ki_batch[0] approximations)
  * coverage: lab-frame solid-angle fraction actually populated (this is what
    the coverage window must learn), plus the sample-frame footprint
  * SH moments per shell l=1..L_max: the deviatoric 2nd-moment energy
    ||A_dev_l||_F (the orientation signal the filter matches on) and the 1st
    moment ||z_1st_l|| (dipole -> lobes/background/coverage asymmetry), each as
    a RATIO over an isotropic Monte-Carlo baseline at the same N.
  * background/coherence proxies (model-free, no UB needed): angular clustering
    and |q|-shell discreteness.

The moments use the exact e3x call the tracker uses, so the numbers are directly
comparable to what the Kalman update consumes.
"""

import numpy as np
import jax.numpy as jnp
import e3x
from scipy.spatial import cKDTree

try:
    from subhkl.instrument.goniometer import sample_to_lab, lab_to_sample
except Exception:
    sample_to_lab = None
    lab_to_sample = None

try:
    from spectral_tracker import spectrum_learning as sl          # Route B, Step 1 estimator (pure numpy)
except Exception:
    sl = None


# ---------------------------------------------------------------------------
def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack([np.sin(phi) * np.cos(theta),
                     np.sin(phi) * np.sin(theta),
                     np.cos(phi)], axis=1)


def _unit(v):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.where(n == 0, 1.0, n)


def _sh_moments(dirs_unit, L_max):
    """Per-shell ||z_1st|| and ||A_dev||_F from unit directions, exactly as the
    tracker forms them (orthonormal SH, deviatoric 2nd moment)."""
    Y = np.asarray(e3x.so3.irreps.spherical_harmonics(
        jnp.asarray(dirs_unit, dtype=jnp.float32),
        max_degree=L_max, cartesian_order=False, normalization="orthonormal"))
    N = Y.shape[0]
    z_norm, adev_norm = {}, {}
    for l in range(1, L_max + 1):
        s, e, dim = l * l, (l + 1) * (l + 1), 2 * l + 1
        Yl = Y[:, s:e]
        z = Yl.mean(axis=0)                                  # 1st moment
        A = (Yl.T @ Yl) / N * (4.0 * np.pi / dim)            # 2nd moment (tracker norm)
        A_dev = A - (np.trace(A) / dim) * np.eye(dim)
        z_norm[l] = float(np.linalg.norm(z))
        adev_norm[l] = float(np.linalg.norm(A_dev))
    return z_norm, adev_norm


def _adev_tensors(dirs_unit, L_max, weights=None):
    """Per-shell deviatoric 2nd-moment TENSORS (not just norms), optionally
    population-weighted. Same SH convention as _sh_moments, so the OBSERVED
    (uniform over events) and PREDICTED (weighted over reflections) tensors
    live in the same basis and compare by Frobenius overlap."""
    Y = np.asarray(e3x.so3.irreps.spherical_harmonics(
        jnp.asarray(dirs_unit, dtype=jnp.float32),
        max_degree=L_max, cartesian_order=False, normalization="orthonormal"))
    w = np.ones(Y.shape[0]) if weights is None else np.asarray(weights, float)
    wsum = float(w.sum()) if w.sum() > 0 else 1.0
    out = {}
    for l in range(1, L_max + 1):
        s, e, dim = l * l, (l + 1) * (l + 1), 2 * l + 1
        Yl = Y[:, s:e]
        A = (Yl.T * w) @ Yl / wsum * (4.0 * np.pi / dim)
        out[l] = A - (np.trace(A) / dim) * np.eye(dim)
    return out

def _coverage(dirs_unit, n_grid=768):
    """Fraction of a Fibonacci tessellation that holds >=1 event, plus a
    clustering measure (share of events in the densest 10% of filled cells)."""
    grid = _fibonacci_sphere(n_grid)
    _, idx = cKDTree(grid).query(dirs_unit, k=1)
    counts = np.bincount(idx, minlength=n_grid)
    filled = counts > 0
    frac = float(filled.mean())
    fc = np.sort(counts[filled])[::-1]
    k = max(1, int(0.10 * fc.size))
    top10_share = float(fc[:k].sum() / counts.sum()) if counts.sum() else 0.0
    return frac, top10_share


def _angular_spread_deg(dirs_unit):
    """Spread of a set of directions about their mean (0 = all parallel)."""
    m = dirs_unit.mean(axis=0)
    mn = np.linalg.norm(m)
    if mn == 0:
        return 180.0
    cos = np.clip(dirs_unit @ (m / mn), -1, 1)
    return float(np.degrees(np.arccos(cos)).std())


_SPARK = "▁▂▃▄▅▆▇█"


def _sparkline(vals):
    v = np.asarray(vals, float)
    vmax = v.max() if v.size and v.max() > 0 else 1.0
    idx = np.clip((v / vmax * (len(_SPARK) - 1)).round().astype(int), 0, len(_SPARK) - 1)
    return "".join(_SPARK[i] for i in idx)


def _intensity_lookup(structure_factors):
    """Build an (h,k,l)->I dict from a gemmi MTZ, a dict, or None."""
    if structure_factors is None:
        return None
    if isinstance(structure_factors, dict):
        return structure_factors
    try:  # gemmi.Mtz duck-typing
        h = structure_factors.column_with_label('H') or structure_factors.columns[0]
        k = structure_factors.column_with_label('K') or structure_factors.columns[1]
        l = structure_factors.column_with_label('L') or structure_factors.columns[2]
        ic = next((c for c in structure_factors.columns if c.type == 'J' or c.label == 'I'), None)
        if ic is None:
            ic = next((c for c in structure_factors.columns if c.type == 'F' or c.label == 'F'), None)
        if ic is None:
            return None
        ha, ka, la, ia = (np.array(h.array), np.array(k.array),
                          np.array(l.array), np.array(ic.array))
        if ic.type == 'F':
            ia = ia ** 2
        return {(int(a), int(b), int(c)): float(v) for a, b, c, v in zip(ha, ka, la, ia)}
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Gate-acceptance orientation test (pure numpy). Replaces the A_dev-overlap
# verdict, which is cap-dominated and misleading under heavy anisotropic
# background -- it rewards any orientation that fills the detector footprint,
# not the one whose reflections sit on real peaks. Here the discriminator is
# the gate ACCEPTANCE COUNT normalized by a uniform-fill-of-footprint null,
# which cancels the cap. See also _pool_spacing_deg: if the pool tiles the
# sphere finer than the gate cone, NO metric can discriminate at that cos_gate.
# ---------------------------------------------------------------------------
def _pool_dirs(cell, U, h_max, d_min, d_max, structure_factors=None, drop_absences=True):
    B = sl.reciprocal_B(*cell)
    hv = np.arange(-h_max, h_max + 1)
    H, K, L = np.meshgrid(hv, hv, hv, indexing="ij")
    hkl = np.stack([H.ravel(), K.ravel(), L.ravel()]).astype(float)
    hkl = hkl[:, ~((hkl[0] == 0) & (hkl[1] == 0) & (hkl[2] == 0))]
    q = B @ hkl; qn = np.linalg.norm(q, axis=0)
    res = (qn > 1.0 / d_max) & (qn < 1.0 / d_min)
    hkl, q, qn = hkl[:, res], q[:, res], qn[res]
    imap = _intensity_lookup(structure_factors)
    if imap is not None and drop_absences:        # drop systematic absences: their
        I = np.array([imap.get((int(h), int(k), int(l)),   # "peaks" catch only bg
                               imap.get((-int(h), -int(k), -int(l)), 0.0))
                      for h, k, l in hkl.T], float)
        keep = I > 0
        hkl, q, qn = hkl[:, keep], q[:, keep], qn[keep]
    pred = (np.asarray(U, float) @ (q / qn)).T
    return pred / np.linalg.norm(pred, axis=1, keepdims=True)

def _pool_spacing_deg(pred_unit):
    """Mean nearest-neighbour angular spacing (deg) of the predicted directions."""
    if len(pred_unit) < 2:
        return 180.0
    d, _ = cKDTree(pred_unit).query(pred_unit, k=2)
    return float(np.degrees(2 * np.arcsin(np.clip(d[:, 1] / 2, 0, 1))).mean())

def _gate_acceptance(q_unit, pred_unit, cos_gate, chunk=20000):
    nn = np.empty(len(q_unit))
    for s in range(0, len(q_unit), chunk):
        nn[s:s+chunk] = (q_unit[s:s+chunk] @ pred_unit.T).max(axis=1)
    return float((nn > cos_gate).mean())

def _uniform_in_footprint(observed_dirs, n, rng, n_grid=768):
    grid = _fibonacci_sphere(n_grid); tree = cKDTree(grid)
    occ = np.zeros(n_grid, bool); occ[np.unique(tree.query(observed_dirs, k=1)[1])] = True
    out, got = [], 0
    while got < n:
        v = rng.normal(size=(n, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
        k = v[occ[tree.query(v, k=1)[1]]]; out.append(k); got += len(k)
    return np.vstack(out)[:n]

def angular_power_spectrum(rep, U_seed, structure_factors=None, lorentz=True, phi="auto",
                           L_max=8, h_max=8, d_min=2.0, d_max=10.0, wl_band="auto",
                           coverage_mask=True, n_grid=768, overlap_thr=0.45,
                           L_cov=3, sigmoid_k=10, cov_threshold_frac=0.3, verbose=True):
    """Per-shell A_dev overlap of OBSERVED events vs the U_seed Bragg model under
    THREE predicted-pool weightings: no mask, the hard detector footprint, and the
    SH-vis reconstruction the tracker actually uses. The hard-vs-SH gap at the seed
    is the bias the Kalman descends. (coverage_mask kept for call compatibility;
    all three are always reported.)"""
    ctx = rep["_scan"]
    q_unit, ki_mean, cell = ctx["q_unit"], ctx["ki_mean"], ctx["cell"]
    if phi == "auto":
        sp = rep.get("spectrum") or {}; phi = sp.get("phi") if sp.get("available") else None
    if wl_band == "auto":
        sp = rep.get("spectrum") or {}; wl_band = sp.get("wl_band") if sp.get("available") else None

    B = sl.reciprocal_B(*cell)
    hv = np.arange(-h_max, h_max + 1)
    H, K, L = np.meshgrid(hv, hv, hv, indexing="ij")
    hkl = np.stack([H.ravel(), K.ravel(), L.ravel()])
    hkl = hkl[:, ~((hkl[0] == 0) & (hkl[1] == 0) & (hkl[2] == 0))]
    q_theo = B @ hkl; qn = np.linalg.norm(q_theo, axis=0)
    res = (qn > 1.0 / d_max) & (qn < 1.0 / d_min)
    hkl, q_theo, qn = hkl[:, res], q_theo[:, res], qn[res]
    qhat = q_theo / np.where(qn == 0, 1.0, qn)
    pred = (np.asarray(U_seed, float) @ qhat).T

    lam, _ = sl.bragg_wavelengths(q_theo, U_seed, ki_mean)
    finite = np.isfinite(lam) & (lam > 0)
    if wl_band is None:
        fl = lam[finite]; wl_band = (max(0.1, float(np.percentile(fl, 1))), float(np.percentile(fl, 99)))
    in_band = finite & (lam > wl_band[0]) & (lam < wl_band[1])
    imap = _intensity_lookup(structure_factors)
    if imap is not None:
        I = np.array([imap.get((int(h), int(k), int(l)),
                               imap.get((-int(h), -int(k), -int(l)), 0.01)) for h, k, l in hkl.T], float)
        I = np.where(I > 0, I, 0.01)
    else:
        I = np.ones(hkl.shape[1])
    Lf = (np.where((qn > 0) & (lam > 0), 4.0 * (lam ** 2) / (qn ** 2), 1.0) if lorentz else np.ones(hkl.shape[1]))
    spec = np.ones(hkl.shape[1])
    if phi is not None:
        s = np.asarray(phi(lam), float); spec = np.where(np.isfinite(s) & (s > 0), s, 0.0)
    p = np.clip(np.where(in_band, 1.0, 0.0) * I * Lf * spec, 0.0, None)
    in_pool = in_band & (p > 0)
    if int(in_pool.sum()) < 8:
        return {"available": False, "reason": f"only {int(in_pool.sum())} in-band refl"}

    # --- the three masks ---
    foot = _footprint_mask(pred, q_unit, n_grid).astype(float)          # hard detector footprint
    Yc_obs = np.asarray(e3x.so3.irreps.spherical_harmonics(
        jnp.asarray(q_unit, jnp.float32), max_degree=L_cov, cartesian_order=False,
        normalization="orthonormal"))
    Yc_pred = np.asarray(e3x.so3.irreps.spherical_harmonics(
        jnp.asarray(pred, jnp.float32), max_degree=L_cov, cartesian_order=False,
        normalization="orthonormal"))
    cov_coeffs = Yc_obs.mean(axis=0)
    C_obs, C_pred = Yc_obs @ cov_coeffs, Yc_pred @ cov_coeffs
    med = float(np.median(C_obs[C_obs > 1e-9])) if np.any(C_obs > 1e-9) else 1.0
    cov_scale = max(cov_threshold_frac * med, 1e-6)
    vis = 1.0 / (1.0 + np.exp(-sigmoid_k * (C_pred / cov_scale - 1.0)))   # SH-vis, tracker formula

    w0 = np.where(in_pool, p, 0.0)
    weights = {"none": w0, "hard": w0 * foot, "SH": w0 * vis}

    obs_t = _adev_tensors(q_unit, L_max)
    pt = {m: _adev_tensors(pred, L_max, w) for m, w in weights.items()}
    rng = np.random.default_rng(0)
    iso = _adev_tensors(_unit(rng.normal(size=(min(len(q_unit), 100000), 3))), L_max)

    per_shell, ov = {}, {m: [] for m in weights}
    for l in range(1, L_max + 1):
        Ao, Ai = obs_t[l], iso[l]; no = float(np.linalg.norm(Ao)); ni = float(np.linalg.norm(Ai)) + 1e-12
        row = {"obs_over_iso": no / ni}
        for m in weights:
            Ap = pt[m][l]; npd = float(np.linalg.norm(Ap))
            o = float(np.sum(Ao * Ap) / (no * npd)) if (no > 0 and npd > 0) else float("nan")
            row[f"overlap_{m}"] = o
            if l >= 2 and np.isfinite(o): ov[m].append(o)
        per_shell[l] = row
    mean_ov = {m: (float(np.mean(v)) if v else float("nan")) for m, v in ov.items()}

    gap = mean_ov["hard"] - mean_ov["SH"]
    if mean_ov["hard"] > overlap_thr and gap > 0.15:
        verdict = (f"HARD footprint recovers the seed (mean l>=2 overlap {mean_ov['hard']:+.2f}) but "
                   f"SH-vis loses {gap:+.2f} of it (SH={mean_ov['SH']:+.2f}) -> the SH coverage "
                   f"reconstruction is the bias source; replace vis with the hard footprint.")
    elif mean_ov["hard"] > overlap_thr:
        verdict = (f"hard and SH agree (hard={mean_ov['hard']:+.2f}, SH={mean_ov['SH']:+.2f}); the "
                   f"mask is not the bottleneck -- look elsewhere (gradient-through-coverage, spectrum).")
    else:
        verdict = (f"even the hard footprint is low ({mean_ov['hard']:+.2f}) -> not a mask problem; "
                   f"broadband non-Bragg structure or wrong band/cell.")
    if verbose:
        cov_sh = float((vis[in_pool] > 0.5).mean()); cov_hd = float(foot[in_pool].mean())
        print(f"\n angular power spectrum & per-shell A_dev overlap (at U_seed)")
        print(f"   {'l':>2} | {'obs/iso':>8} | {'ov(none)':>8} | {'ov(hard)':>8} | {'ov(SH)':>8}")
        for l in range(1, L_max + 1):
            d = per_shell[l]
            print(f"   {l:>2} | {d['obs_over_iso']:>8.1f} | {d['overlap_none']:>+8.2f} | "
                  f"{d['overlap_hard']:>+8.2f} | {d['overlap_SH']:>+8.2f}")
        print(f"   mean l>=2:  none {mean_ov['none']:+.2f} | hard {mean_ov['hard']:+.2f} | SH {mean_ov['SH']:+.2f}")
        print(f"   pool kept:  hard {100*cov_hd:.0f}% | SH(>0.5) {100*cov_sh:.0f}%  (of in-band refl)")
        print(f"   => {verdict}")
    return {"available": True, "per_shell": per_shell, "mean_overlap": mean_ov,
            "hard_minus_sh": gap, "verdict": verdict}

def _spectrum_block(q_unit, ki_mean, cell, U_est, h_max, d_min, d_max,
                    wl_band, structure_factors, family, cos_min, geom_fn, lorentz=True):
    """
    Learn the APPARENT incident spectrum from reflection populations, using the
    Step 1 estimator. Active only when cell + U_est are supplied. Reports the
    fitted shape, the data's wavelength span, and a sparkline of binned phi_hat
    vs the fit -- a read-only sanity check before wiring phi into the tracker.
    """
    if sl is None:
        return {"available": False, "reason": "spectrum_learning module not importable"}
    if U_est is None or cell is None:
        return {"available": False,
                "reason": "needs cell + U_est (orientation) to tag wavelengths"}

    B = sl.reciprocal_B(*cell)
    hv = np.arange(-h_max, h_max + 1)
    H, K, L = np.meshgrid(hv, hv, hv, indexing="ij")
    hkl = np.stack([H.ravel(), K.ravel(), L.ravel()])
    hkl = hkl[:, ~((hkl[0] == 0) & (hkl[1] == 0) & (hkl[2] == 0))]
    q_theo = B @ hkl
    qn = np.linalg.norm(q_theo, axis=0)
    res = (qn > 1.0 / d_max) & (qn < 1.0 / d_min)
    hkl, q_theo, qn = hkl[:, res], q_theo[:, res], qn[res]
    if hkl.shape[1] < 8:
        return {"available": False, "reason": f"only {hkl.shape[1]} reflections in d-range"}

    lam, _ = sl.bragg_wavelengths(q_theo, U_est, ki_mean)
    finite = np.isfinite(lam) & (lam > 0)
    if wl_band is None:                       # auto-detect from the data's own lambda span
        fl = lam[finite]
        wl_band = (max(0.1, float(np.percentile(fl, 1))), float(np.percentile(fl, 99)))
    in_band = finite & (lam > wl_band[0]) & (lam < wl_band[1])

    qhat = q_theo / np.where(qn == 0, 1.0, qn)
    pred = (np.asarray(U_est, float) @ qhat).T          # (M,3) unit, sample frame
    band_idx = np.where(in_band)[0]
    if band_idx.size < 8:
        return {"available": False, "reason": f"only {band_idx.size} in-band reflections"}

    # Assign each event to its nearest predicted Bragg direction (in OR out of
    # band); drop events beyond cos_min (background far from every reflection).
    tree = cKDTree(pred)
    d_thresh = np.sqrt(max(2.0 * (1.0 - cos_min), 0.0))
    dist, nn = tree.query(q_unit, k=1)
    keep = dist <= d_thresh
    counts = np.bincount(nn[keep], minlength=hkl.shape[1]).astype(float)

    # Model-free background floor: out-of-band reflections whose ray has no
    # in-band member satisfy no Bragg condition for this band, so their captured
    # counts are pure background. Their median is the per-cap background, the
    # same constant for every (equal-radius) cap; subtract it. This rescues the
    # fit under background (which otherwise inflates the low-phi band edges via
    # the 1/lambda^2 geometry and flips the curvature). Recovers the peak well;
    # residual Poisson scatter still inflates the fitted WIDTH, so treat s as an
    # upper bound under heavy background.
    reduced = sl._reduce_hkl(hkl)
    from collections import defaultdict
    occ = defaultdict(int)
    for j in band_idx:
        occ[tuple(reduced[:, j])] += 1
    monitor = np.array([(not in_band[j]) and occ[tuple(reduced[:, j])] == 0
                        for j in range(hkl.shape[1])])
    c_bg = float(np.median(counts[monitor])) if monitor.any() else 0.0
    counts = np.clip(counts - c_bg, 0.0, None)

    imap = _intensity_lookup(structure_factors)
    if imap is not None:
        I = np.array([imap.get((int(h), int(k), int(l)),
                               imap.get((-int(h), -int(k), -int(l)), 0.01))
                      for h, k, l in hkl.T], float)
        I = np.where(I > 0, I, 0.01)
    else:
        I = np.ones(hkl.shape[1])

    if lorentz:
        # Kinematic Lorentz factor for stationary sample (Laue)
        # L = lambda^4 / sin^2(theta)
        # Substitute sin(theta) = lambda / 2d = lambda * qn / 2  ->  L = 4 * lambda^2 / qn^2
        # Fitter expects this folded into the predictor I.
        L_factor = np.where(qn > 0, 4.0 * (lam ** 2) / (qn ** 2), 1.0)
        I = I * L_factor

    geom = None
    if geom_fn is not None:
        geom = np.where(in_band, np.asarray(geom_fn(lam), float), 1.0)

    singles = sl.singles_mask(hkl, in_band)
    try:
        params, phi, info = sl.learn_spectrum(
            lam, counts, I, geom=geom, singles=singles,
            family=family, lam_band=wl_band, min_count=5)
    except Exception as ex:
        return {"available": False, "reason": f"fit failed: {ex}",
                "n_singles_inband": int(singles.sum()),
                "assigned_frac": float(keep.mean())}

    if family == "lognormal":
        mode = params["lam0"] * np.exp(-params["s"] ** 2)
    elif family == "maxwellian":
        mode = params["lam_T"] * np.sqrt(2.0 / 5.0)
    else:
        mode = float("nan")

    lam_used = info["lam_used"]
    nb = 24
    edges = np.linspace(wl_band[0], wl_band[1], nb + 1)
    cen = 0.5 * (edges[:-1] + edges[1:])
    binid = np.clip(np.digitize(lam_used, edges) - 1, 0, nb - 1)
    obs = np.zeros(nb); wsum = np.zeros(nb)
    for bi, val, ww in zip(binid, info["phi_hat_used"], info["weights"]):
        obs[bi] += val * ww; wsum[bi] += ww
    obs = np.where(wsum > 0, obs / np.where(wsum == 0, 1.0, wsum), 0.0)

    return {"available": True, "family": family, "params": params, "mode": float(mode),
            "phi": phi,                 # callable lam -> unnormalized weight (for tracker)
            "wl_band": (float(wl_band[0]), float(wl_band[1])),
            "n_singles_inband": int(singles.sum()), "n_singles_used": info["n_singles_used"],
            "assigned_frac": float(keep.mean()), "cos_min": float(cos_min),
            "c_bg": c_bg, "n_monitors": int(monitor.sum()),
            "lam_used_pct": tuple(float(x) for x in np.percentile(lam_used, [5, 50, 95])),
            "divided_I": imap is not None, "divided_geom": geom is not None, "lorentz": lorentz,
            "spark_obs": _sparkline(obs), "spark_fit": _sparkline(phi(cen))}

def _footprint_mask(test_dirs, observed_dirs, n_grid=768):
    """True for test directions that land in a Fibonacci cell the observed events
    actually populate -- i.e. inside the detector footprint. Lets the predicted
    set be compared on the SAME ~solid angle the data covers."""
    grid = _fibonacci_sphere(n_grid)
    tree = cKDTree(grid)
    occ = np.zeros(n_grid, bool)
    occ[np.unique(tree.query(np.asarray(observed_dirs, float), k=1)[1])] = True
    return occ[tree.query(np.asarray(test_dirs, float), k=1)[1]]

def _rand_rotations(n, rng):
    q = rng.normal(size=(n, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    R = np.empty((n, 3, 3))
    R[:, 0, 0] = 1-2*(y*y+z*z); R[:, 0, 1] = 2*(x*y-z*w); R[:, 0, 2] = 2*(x*z+y*w)
    R[:, 1, 0] = 2*(x*y+z*w); R[:, 1, 1] = 1-2*(x*x+z*z); R[:, 1, 2] = 2*(y*z-x*w)
    R[:, 2, 0] = 2*(x*z-y*w); R[:, 2, 1] = 2*(y*z+x*w); R[:, 2, 2] = 1-2*(x*x+y*y)
    return R

def _angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))

def orientation_scan(rep, U_seed, structure_factors=None, lorentz=True, phi="auto",
                     L_max=6, h_max=8, d_min=1.0, d_max=10.0, wl_band="auto",
                     coverage_mask=True, n_grid=768, n_coarse=3000,
                     refine_scales=(0.3, 0.1, 0.03, 0.01), n_refine=1500,
                     seed=0, verbose=True):
    """Search SO(3) for R maximizing the mean l>=2 overlap of predicted(R @ U_seed)
    vs observed A_dev. Resolves the seed fork:
      best~1, angle~0    -> seed is correct; lock failure is optimizer/annealing.
      best~1, angle large-> seed is in the WRONG FRAME; use R @ U_seed (this R is
                            your indexer<->loader goniometer/ki offset).
      best stays low     -> no orientation matches: cell/d-range/band/data problem
                            (wrong or stale index), not initialization.
    Needs analyze_event_stream(...) to have stashed rep['_scan']."""
    ctx = rep["_scan"]
    q_unit, ki_mean, cell = ctx["q_unit"], ctx["ki_mean"], ctx["cell"]
    if phi == "auto":
        sp = rep.get("spectrum") or {}
        phi = sp.get("phi") if sp.get("available") else None
    if wl_band == "auto":
        sp = rep.get("spectrum") or {}
        wl_band = sp.get("wl_band") if sp.get("available") else None
    rng = np.random.default_rng(seed)

    # R-independent pool
    B = sl.reciprocal_B(*cell)
    hv = np.arange(-h_max, h_max + 1)
    H, K, L = np.meshgrid(hv, hv, hv, indexing="ij")
    hkl = np.stack([H.ravel(), K.ravel(), L.ravel()])
    hkl = hkl[:, ~((hkl[0] == 0) & (hkl[1] == 0) & (hkl[2] == 0))]
    q_theo = B @ hkl
    qn = np.linalg.norm(q_theo, axis=0)
    res = (qn > 1.0 / d_max) & (qn < 1.0 / d_min)
    hkl, q_theo, qn = hkl[:, res], q_theo[:, res], qn[res]
    qhat = q_theo / np.where(qn == 0, 1.0, qn)
    imap = _intensity_lookup(structure_factors)
    if imap is not None:
        I = np.array([imap.get((int(h), int(k), int(l)),
                               imap.get((-int(h), -int(k), -int(l)), 0.01))
                      for h, k, l in hkl.T], float)
        I = np.where(I > 0, I, 0.01)
    else:
        I = np.ones(hkl.shape[1])

    obs_t = _adev_tensors(q_unit, L_max)               # observed, once
    grid = _fibonacci_sphere(n_grid); gtree = cKDTree(grid)
    occ = np.zeros(n_grid, bool)
    occ[np.unique(gtree.query(np.asarray(q_unit, float), k=1)[1])] = True
    U_seed = np.asarray(U_seed, float)

    def objective(Ucand):
        proj = np.asarray(ki_mean, float) @ (Ucand @ qhat)
        with np.errstate(divide="ignore", invalid="ignore"):
            lam = -2.0 * proj / np.where(qn == 0, np.nan, qn)
        finite = np.isfinite(lam) & (lam > 0)
        band = wl_band if wl_band is not None else (
            max(0.1, float(np.percentile(lam[finite], 1))), float(np.percentile(lam[finite], 99)))
        in_band = finite & (lam > band[0]) & (lam < band[1])
        if int(in_band.sum()) < 8:
            return -1.0, None
        pred = (Ucand @ qhat).T
        Lf = (np.where((qn > 0) & (lam > 0), 4.0 * (lam ** 2) / (qn ** 2), 1.0)
              if lorentz else np.ones(hkl.shape[1]))
        spec = np.ones(hkl.shape[1])
        if phi is not None:
            s = np.asarray(phi(lam), float); spec = np.where(np.isfinite(s) & (s > 0), s, 0.0)
        p = np.where(in_band, 1.0, 0.0) * I * Lf * spec
        p = np.clip(p, 0.0, None)
        m = in_band & (p > 0)
        if coverage_mask:
            m = m & occ[gtree.query(pred, k=1)[1]]
        if int(m.sum()) < 8:
            return -1.0, None
        w = np.where(m, p, 0.0)              # full-length weights -> pred stays (M,3)
        pt = _adev_tensors(pred, L_max, w)   # SH shape is constant -> compiles ONCE
        ovs = []
        for l in range(2, L_max + 1):
            Ao, Ap = obs_t[l], pt[l]
            no, npd = np.linalg.norm(Ao), np.linalg.norm(Ap)
            if no > 0 and npd > 0:
                ovs.append(float(np.sum(Ao * Ap) / (no * npd)))
        return (float(np.mean(ovs)) if ovs else -1.0), pt

    seed_ov, _ = objective(U_seed)
    best_R = np.eye(3); best, _ = objective(U_seed)
    for R in _rand_rotations(n_coarse, rng):
        v, _ = objective(R @ U_seed)
        if v > best:
            best, best_R = v, R
    for sc in refine_scales:
        ax = rng.normal(size=(n_refine, 3)); ax /= np.linalg.norm(ax, axis=1, keepdims=True)
        ths = sc * rng.normal(size=n_refine)
        for i in range(n_refine):
            a = ax[i]; th = ths[i]
            w_ = np.cos(th/2); x_, y_, z_ = np.sin(th/2) * a
            Rd = np.array([[1-2*(y_*y_+z_*z_), 2*(x_*y_-z_*w_), 2*(x_*z_+y_*w_)],
                           [2*(x_*y_+z_*w_), 1-2*(x_*x_+z_*z_), 2*(y_*z_-x_*w_)],
                           [2*(x_*z_-y_*w_), 2*(y_*z_+x_*w_), 1-2*(x_*x_+y_*y_)]])
            v, _ = objective((Rd @ best_R) @ U_seed)
            if v > best:
                best, best_R = v, Rd @ best_R

    _, pt = objective(best_R @ U_seed)
    per_shell = {}
    for l in range(1, L_max + 1):
        Ao = obs_t[l]; Ap = pt[l] if pt is not None else np.zeros_like(Ao)
        no, npd = float(np.linalg.norm(Ao)), float(np.linalg.norm(Ap))
        per_shell[l] = (float(np.sum(Ao*Ap)/(no*npd)) if no > 0 and npd > 0 else float("nan"))
    ang = _angle_deg(best_R)
    if best > 0.8 and ang < 5:
        verdict = "seed is correct -> lock failure is optimizer/annealing, not the seed"
    elif best > 0.8:
        verdict = f"seed is in the WRONG FRAME by {ang:.1f} deg -> use (best_R @ U_seed) as the seed"
    elif best > 0.5:
        verdict = f"partial match at {ang:.1f} deg -> tighten d-range/band; cell may be slightly off"
    else:
        verdict = "NO orientation matches -> cell / d-range / band / wrong-or-stale index, not init"
    if verbose:
        print(f"\n SO(3) orientation scan (best R @ U_seed vs observed A_dev)")
        print(f"   seed overlap (l>=2)  = {seed_ov:.3f}")
        print(f"   best overlap (l>=2)  = {best:.3f}   at angle(R) = {ang:.1f} deg from identity")
        print(f"   per-shell @ best: " + " ".join(f"l{l}:{per_shell[l]:+.2f}" for l in sorted(per_shell)))
        print(f"   => {verdict}")
    return {"best_R": best_R, "best_overlap": best, "seed_overlap": seed_ov,
            "angle_deg": ang, "per_shell": per_shell, "verdict": verdict}

# ---------------------------------------------------------------------------
def analyze_event_stream(event_batches, gonio_axes=None, gonio_offsets=None,
                         L_max=8, n_grid=768, mc_baseline=True, label="",
                         cell=None, U_est=None, ki_vec=None,
                         h_max=6, d_min=1.0, d_max=10.0, wl_band=None,
                         accept_cos_gate=0.9999, accept_d_min=None, accept_h_max=None,
                         structure_factors=None, spectrum_family="lognormal",
                         cos_min=0.999, geom_fn=None, lorentz=False):
    qs, ts, kis, angs = [], [], [], []
    batch_spans = []          # per-batch goniometer sweep (deg)
    n_batches = 0

    for b in event_batches:
        q, t = np.asarray(b[0]), np.asarray(b[1]).ravel()
        ang, ki = np.asarray(b[5]), np.asarray(b[7])
        if t.size == 0:
            continue
        n_batches += 1
        qs.append(q); ts.append(t); kis.append(ki); angs.append(np.atleast_2d(ang))
        a2 = np.atleast_2d(ang)
        if a2.shape[0] == 1 and a2.shape[1] != q.shape[0]:
            a2 = a2.T
        batch_spans.append(np.ptp(a2, axis=0) if a2.size else np.zeros(1))

    q = np.concatenate(qs); t = np.concatenate(ts)
    ki = np.concatenate(kis); ang = np.concatenate(angs, axis=0)
    N = len(t)
    q_unit = _unit(q.astype(np.float64))

    # --- timing / rate ---
    dur = float(t.max() - t.min()) if N else 0.0
    rate = N / dur if dur > 0 else float("nan")

    # --- vector sanity ---
    qmag = np.linalg.norm(q.astype(np.float64), axis=1)
    ki_unit = _unit(ki.astype(np.float64))

    # The loader de-rotates every event into the SAMPLE frame, so the streamed ki
    # (b[7]) is already the sample-frame beam -- exactly what bragg_wavelengths
    # needs, since U @ q_hat also lives in the sample frame. This is the source of
    # truth for tagging. A user-supplied ki_vec follows the subhkl convention and
    # is in the LAB frame; it must be de-rotated by the goniometer (R^T) before it
    # can be used for tagging, or it tilts every tagged wavelength by the setting.
    ki_mean = _unit(np.median(ki_unit, axis=0)[None])[0]
    if ki_vec is not None:
        ki_lab = _unit(np.asarray(ki_vec, float)[None])[0]
        ki_lab_in_sample = ki_lab
        if (lab_to_sample is not None and gonio_axes is not None
                and ang.size and ang.shape[-1] == len(np.asarray(gonio_axes))):
            try:
                a0 = ang[0] if ang.shape[0] == N else ang[:, 0]   # (num_axes,) representative setting
                ki_lab_in_sample = _unit(np.atleast_2d(np.asarray(
                    lab_to_sample(ki_lab, np.asarray(gonio_axes), a0, None,
                                  zero_offsets=gonio_offsets, is_vector=True))))[0]
            except Exception as ex:
                print(f"  [spectrum] lab->sample ki de-rotation skipped: {ex}")
        # Prefer the stream; the converted lab vector is a cross-check / fallback.
        disagree = float(np.degrees(np.arccos(np.clip(ki_mean @ ki_lab_in_sample, -1, 1))))
        if disagree > 1.0:
            print(f"  [spectrum] note: supplied (lab) ki de-rotates to a sample-frame beam "
                  f"{disagree:.1f} deg from the streamed beam; using the streamed beam for "
                  f"tagging. Pass ki_vec=None to silence.")
    nbins = 50
    hq, _ = np.histogram(qmag, bins=nbins)
    densest_qbin = float(hq.max() / hq.sum()) if hq.sum() else 0.0   # uniform ~ 1/nbins

    # --- goniometer ---
    ang_range = (np.ptp(ang, axis=0) if ang.size else np.zeros(1)).astype(float)
    max_batch_sweep = float(np.max([np.max(s) for s in batch_spans])) if batch_spans else 0.0
    ki_spread = _angular_spread_deg(ki_unit)

    # --- coverage: sample frame, and lab frame (de-rotated) ---
    cov_s, clus_s = _coverage(q_unit, n_grid)
    cov_l, clus_l = cov_s, clus_s
    lab_done = False
    if gonio_axes is not None and sample_to_lab is not None and ang.size and np.ptp(ang) != 0:
        try:
            a_na_N = ang.T if ang.shape[0] == N else ang     # -> (num_axes, N)
            q_lab = np.asarray(sample_to_lab(q_unit, np.asarray(gonio_axes), a_na_N,
                                             None, zero_offsets=gonio_offsets, is_vector=True))
            cov_l, clus_l = _coverage(_unit(q_lab), n_grid)
            lab_done = True
        except Exception as ex:
            print(f"  [coverage] lab de-rotation skipped: {ex}")

    # --- SH moments (the tracker observable) + isotropic baseline ---
    z_norm, adev_norm = _sh_moments(q_unit, L_max)
    z_ratio, adev_ratio = {}, {}
    if mc_baseline:
        rng = np.random.default_rng(0)
        Nmc = min(N, 200_000)
        mc = _unit(rng.normal(size=(Nmc, 3)))
        z_b, a_b = _sh_moments(mc, L_max)
        for l in range(1, L_max + 1):
            z_ratio[l] = z_norm[l] / (z_b[l] + 1e-12)
            adev_ratio[l] = adev_norm[l] / (a_b[l] + 1e-12)

    # --- learned (apparent) incident spectrum, if cell + orientation supplied ---
    spectrum = _spectrum_block(
        q_unit, ki_mean, cell, U_est, h_max, d_min, d_max, wl_band,
        structure_factors, spectrum_family, cos_min, geom_fn, lorentz)

    _ss = np.random.default_rng(0).choice(N, size=min(N, 100_000), replace=False)
    rep_scan = {"q_unit": q_unit[_ss], "ki_mean": ki_mean, "cell": cell}

    rep = dict(
        label=label, n_events=N, n_batches=n_batches,
        events_per_batch=N / max(n_batches, 1), duration=dur, rate_hz=rate,
        t_dtype=str(t.dtype),
        qmag_pct=tuple(float(x) for x in np.percentile(qmag, [5, 50, 95])),
        qmag_unit_frac=float(np.mean(np.abs(qmag - 1.0) < 1e-3)),
        densest_qbin_frac=densest_qbin, qbin_uniform=1.0 / nbins,
        ki_norm_med=float(np.median(np.linalg.norm(ki, axis=1))),
        ki_spread_deg=ki_spread,
        gonio_range_deg=tuple(float(x) for x in np.atleast_1d(ang_range)),
        max_batch_sweep_deg=max_batch_sweep,
        coverage_frac_sample=cov_s, coverage_frac_lab=cov_l, lab_frame=lab_done,
        eff_solid_angle_pct=100.0 * cov_l, clustering_top10_lab=clus_l,
        z_ratio=z_ratio, adev_ratio=adev_ratio,
        z_norm=z_norm, adev_norm=adev_norm, L_max=L_max,
        spectrum=spectrum,
        _scan=rep_scan,
    )
    _print_report(rep)
    return rep

def _moment_vec(dirs, L_max, weights=None):
    """Stacked [z_1st, A_dev.flatten()] per shell l=1..L_max, TRUE weighted
    average (matches the patched forward model)."""
    Y = np.asarray(e3x.so3.irreps.spherical_harmonics(
        jnp.asarray(dirs, jnp.float32), max_degree=L_max,
        cartesian_order=False, normalization="orthonormal"))
    w = np.ones(Y.shape[0]) if weights is None else np.asarray(weights, float)
    wsum = float(w.sum()) + 1e-12
    parts = []
    for l in range(1, L_max + 1):
        s, e, dim = l * l, (l + 1) * (l + 1), 2 * l + 1
        Yl = Y[:, s:e]
        z1 = (Yl * w[:, None]).sum(0) / wsum
        A = (Yl.T * w) @ Yl / wsum * (4.0 * np.pi / dim)
        Adev = A - np.trace(A) / dim * np.eye(dim)
        parts.append(z1); parts.append(Adev.ravel())
    return np.concatenate(parts)


def seed_vs_random_innovation(rep, U_seed, structure_factors=None, phi="auto",
                              L_max=8, h_max=8, d_min=2.0, d_max=10.0, wl_band="auto",
                              coverage_mask=True, n_grid=768, n_rand=32, seed=0, verbose=True):
    """Static landscape test: is U_seed a true minimum of ||z_data - z_pred||,
    and under which population weighting? For each weighting, reports the seed's
    innovation vs the random-orientation distribution. The seed is a genuine
    minimum iff its innovation is below even the BEST random orientation
    (margin = rand_min / seed > 1). Pick the weighting with the largest margin;
    that is the objective whose minimum is at the truth."""
    ctx = rep["_scan"]; q_unit, ki_mean, cell = ctx["q_unit"], ctx["ki_mean"], ctx["cell"]
    if phi == "auto":
        sp = rep.get("spectrum") or {}; phi = sp.get("phi") if sp.get("available") else None
    if wl_band == "auto":
        sp = rep.get("spectrum") or {}; wl_band = sp.get("wl_band") if sp.get("available") else None

    B = sl.reciprocal_B(*cell)
    hv = np.arange(-h_max, h_max + 1); H, K, L = np.meshgrid(hv, hv, hv, indexing="ij")
    hkl = np.stack([H.ravel(), K.ravel(), L.ravel()])
    hkl = hkl[:, ~((hkl[0] == 0) & (hkl[1] == 0) & (hkl[2] == 0))]
    q_theo = B @ hkl; qn = np.linalg.norm(q_theo, axis=0)
    res = (qn > 1.0 / d_max) & (qn < 1.0 / d_min)
    hkl, q_theo, qn = hkl[:, res], q_theo[:, res], qn[res]
    qhat = q_theo / np.where(qn == 0, 1.0, qn)
    imap = _intensity_lookup(structure_factors)
    if imap is not None:
        Fsq = np.array([imap.get((int(h), int(k), int(l)),
                                 imap.get((-int(h), -int(k), -int(l)), 0.01)) for h, k, l in hkl.T], float)
        Fsq = np.where(Fsq > 0, Fsq, 0.01)
    else:
        Fsq = np.ones(hkl.shape[1])

    obs_vec = _moment_vec(q_unit, L_max)                       # fixed, independent of U
    # boolean over the stacked vector marking l>=2 entries
    l2 = np.concatenate([np.full(2 * l + 1 + (2 * l + 1) ** 2, l >= 2) for l in range(1, L_max + 1)])

    grid = _fibonacci_sphere(n_grid); gtree = cKDTree(grid)
    occ = np.zeros(n_grid, bool); occ[np.unique(gtree.query(np.asarray(q_unit, float), k=1)[1])] = True

    kinds = ["flat", "F2", "F2_lorentz"] + (["F2_lorentz_phi"] if phi is not None else [])

    def innov(U, kind):
        proj = np.asarray(ki_mean, float) @ (U @ qhat)
        with np.errstate(divide="ignore", invalid="ignore"):
            lam = -2.0 * proj / np.where(qn == 0, np.nan, qn)
        finite = np.isfinite(lam) & (lam > 0)
        band = wl_band if wl_band is not None else (
            max(0.1, float(np.percentile(lam[finite], 1))), float(np.percentile(lam[finite], 99)))
        w = (finite & (lam > band[0]) & (lam < band[1])).astype(float)
        if kind != "flat":
            w = w * Fsq
        if kind in ("F2_lorentz", "F2_lorentz_phi"):
            w = w * np.where((qn > 0) & (lam > 0), 4.0 * (lam ** 2) / (qn ** 2), 0.0)
        if kind == "F2_lorentz_phi" and phi is not None:
            s = np.asarray(phi(lam), float); w = w * np.where(np.isfinite(s) & (s > 0), s, 0.0)
        pred = (U @ qhat).T
        if coverage_mask:
            w = w * occ[gtree.query(pred, k=1)[1]]
        w = np.clip(w, 0.0, None)
        if (w > 0).sum() < 8:
            return np.nan
        d = obs_vec - _moment_vec(pred, L_max, w)
        return float(np.linalg.norm(d[l2]))                    # l>=2 innovation

    rng = np.random.default_rng(seed)
    Rs = _rand_rotations(n_rand, rng)
    U_seed = np.asarray(U_seed, float)
    rows = {}
    for kind in kinds:
        s = innov(U_seed, kind)
        r = np.array([innov(R @ U_seed, kind) for R in Rs]); r = r[np.isfinite(r)]
        rows[kind] = dict(seed=s, rand_med=float(np.median(r)) if r.size else np.nan,
                          rand_min=float(np.min(r)) if r.size else np.nan,
                          margin=(float(np.min(r)) / s) if (r.size and s and s > 0) else np.nan)
    if verbose:
        print(f"\n seed-vs-random innovation  ||z_data - z_pred||  (l>=2, coverage_mask={coverage_mask})")
        print(f"   {'weighting':>16} | {'seed':>9} | {'rand med':>9} | {'rand min':>9} | {'margin':>7}")
        for k, v in rows.items():
            m = v["margin"]
            flag = ("  TRUE MIN" if (np.isfinite(m) and m > 1.5) else
                    "  weak"     if (np.isfinite(m) and m > 1.05) else "  NOT a min")
            print(f"   {k:>16} | {v['seed']:>9.4f} | {v['rand_med']:>9.4f} | {v['rand_min']:>9.4f} | {m:>7.2f}{flag}")
        print("   margin = rand_min / seed ; >1.5 => even the BEST random orientation is worse than truth")
        best = max(rows, key=lambda k: (rows[k]["margin"] if np.isfinite(rows[k]["margin"]) else -1))
        print(f"   => use weighting '{best}' (largest margin); if all margins ~1, the forward model")
        print(f"      is degenerate at this normalization (flat landscape -> dead/indifferent filter).")
    return rows

def _print_report(r):
    print(f"\n{'='*64}\n EVENT-STREAM DIAGNOSTIC: {r['label']}\n{'='*64}")
    print(f" events={r['n_events']:,}  batches={r['n_batches']}  "
          f"~{r['events_per_batch']:.0f}/batch")
    print(f" duration={r['duration']:.3g} (t dtype {r['t_dtype']})  "
          f"rate={r['rate_hz']:.1f} Hz   <- is duration ~seconds & rate sane?")
    print(f"\n geometry sanity")
    print(f"   |q|  p5/50/95 = {r['qmag_pct'][0]:.3f}/{r['qmag_pct'][1]:.3f}/{r['qmag_pct'][2]:.3f}"
          f"   (unit-norm frac {r['qmag_unit_frac']:.2f})")
    print(f"   |q| densest-bin share = {r['densest_qbin_frac']:.3f} "
          f"(uniform {r['qbin_uniform']:.3f}) <- high => discrete Bragg shells")
    print(f"   ki |.|med={r['ki_norm_med']:.3f}  directional spread={r['ki_spread_deg']:.2f} deg "
          f"<- large => beam sweeps (ki_batch[0] approx breaks)")
    print(f"\n goniometer")
    print(f"   per-axis range (deg) = {tuple(round(x,2) for x in r['gonio_range_deg'])}")
    print(f"   MAX per-batch sweep  = {r['max_batch_sweep_deg']:.3f} deg "
          f"<- large => mean-R_batch smears the mask")
    print(f"\n coverage")
    print(f"   lab-frame footprint  = {r['eff_solid_angle_pct']:.1f}% of 4pi "
          f"(frame={'LAB' if r['lab_frame'] else 'SAMPLE(no gonio)'})")
    print(f"   sample-frame filled  = {100*r['coverage_frac_sample']:.1f}%   "
          f"clustering(top10%)={r['clustering_top10_lab']:.2f} "
          f"<- ~0.1 diffuse, ->1 Bragg-peaky")
    print(f"\n SH moments per shell (ratio over isotropic baseline; >>1 = real structure)")
    print(f"   {'l':>2} | {'||z_1st|| (dipole)':>20} | {'||A_dev||_F (orient.)':>22}")
    for l in range(1, r['L_max'] + 1):
        zr = r['z_ratio'].get(l, float('nan'))
        ar = r['adev_ratio'].get(l, float('nan'))
        flag = "  <- ORIENT SIGNAL" if (l >= 2 and ar > 3) else ("  <- lobe/bg" if (l == 1 and zr > 3) else "")
        print(f"   {l:>2} | {zr:>20.1f} | {ar:>22.1f}{flag}")

    sp = r.get('spectrum')
    if sp is not None:
        print(f"\n learned incident spectrum (apparent; from reflection populations)")
        if not sp.get("available"):
            print(f"   [skipped] {sp.get('reason')}")
        else:
            par = ", ".join(f"{k}={v:.3f}" for k, v in sp["params"].items())
            print(f"   family={sp['family']}  {par}   mode~{sp['mode']:.2f} A "
                  f"(peak reliable; width = upper bound under background)")
            lo, hi = sp["wl_band"]
            p = sp["lam_used_pct"]
            print(f"   fit window [{lo:.2f}, {hi:.2f}] A | tagged-lambda p5/50/95 = "
                  f"{p[0]:.2f}/{p[1]:.2f}/{p[2]:.2f} A")
            print(f"   singles used {sp['n_singles_used']}/{sp['n_singles_inband']} | "
                  f"events assigned {100*sp['assigned_frac']:.0f}% (cos>= {sp['cos_min']:.3f}) | "
                  f"bg/cap={sp['c_bg']:.0f} ({sp['n_monitors']} monitors) | "
                  f"divided: I={'Y' if sp['divided_I'] else 'N'} geom={'Y' if sp['divided_geom'] else 'N'} lorentz={'Y' if sp.get('lorentz') else 'N'}")

            print(f"   phi_hat  {sp['spark_obs']}")
            print(f"   fit      {sp['spark_fit']}   ({lo:.1f} -> {hi:.1f} A)")
            if not sp['divided_I']:
                print("   note: no |F|^2 supplied -> curve is spectrum x structure (pass "
                      "structure_factors to deconvolve)")

def compare(a, b):
    """Side-by-side of the discriminating scalars + per-shell A_dev ratios."""
    print(f"\n{'='*64}\n COMPARE: {a['label']}  vs  {b['label']}\n{'='*64}")
    rows = [
        ("rate (Hz)", "rate_hz", "{:.1f}"),
        ("duration", "duration", "{:.3g}"),
        ("max batch sweep (deg)", "max_batch_sweep_deg", "{:.3f}"),
        ("ki spread (deg)", "ki_spread_deg", "{:.2f}"),
        ("|q| densest-bin share", "densest_qbin_frac", "{:.3f}"),
        ("coverage (% 4pi, lab)", "eff_solid_angle_pct", "{:.1f}"),
        ("clustering top10%", "clustering_top10_lab", "{:.2f}"),
    ]
    print(f"   {'metric':<26} {a['label']:>14} {b['label']:>14}")
    for name, key, fmt in rows:
        print(f"   {name:<26} {fmt.format(a[key]):>14} {fmt.format(b[key]):>14}")
    print(f"\n   {'A_dev ratio by shell':<26} {a['label']:>14} {b['label']:>14}")
    for l in range(1, min(a['L_max'], b['L_max']) + 1):
        print(f"   l={l:<24} {a['adev_ratio'].get(l, float('nan')):>14.1f} "
              f"{b['adev_ratio'].get(l, float('nan')):>14.1f}")

    sa, sb = a.get('spectrum'), b.get('spectrum')
    if sa and sb and sa.get("available") and sb.get("available"):
        def _peak(s): return s["mode"]
        print(f"\n   {'spectrum mode (A)':<26} {_peak(sa):>14.2f} {_peak(sb):>14.2f}")
        print(f"   {'spectrum family':<26} {sa['family']:>14} {sb['family']:>14}")
        print(f"   {'singles used':<26} {sa['n_singles_used']:>14} {sb['n_singles_used']:>14}")
    print("\n Read it as: if SYNTH has large l>=2 A_dev ratios (strong orientation")
    print(" signal) but REAL is flat at l>=2 with a big l=1 z-dipole, real data is")
    print(" coverage/background dominated, not orientation-bearing -> the filter")
    print(" has little to lock onto. Mismatched rate/sweep/coverage point at the")
    print(" loader geometry or scan regime instead.")
