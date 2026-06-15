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

def _orientation_match_block(q_unit, ki_mean, cell, U_est, L_max, h_max,
                             d_min, d_max, wl_band, structure_factors,
                             lorentz, phi=None):
    """At the SEED, build predicted shell moments the way the tracker weights them
    and compare to the OBSERVED moments from event directions. Headline is the
    per-shell Frobenius overlap of A_dev:
      ~+1  -> predicted & observed align; seed is a true minimum, so failure to
              lock lives in the optimizer/annealing, not the model.
      ~0/neg -> prediction doesn't reproduce the observation at the seed; the model
              (d-range, band, spectrum, Lorentz) is wrong -- e.g. a d_min that drops
              a detector sector collapses the overlap on every shell.
    Uses the SAME d_min/d_max/wl_band passed in: set those to your TRACKER's values
    to test the tracker configuration."""
    if sl is None or U_est is None or cell is None:
        return {"available": False, "reason": "needs cell + U_est"}

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
    if wl_band is None:
        fl = lam[finite]
        wl_band = (max(0.1, float(np.percentile(fl, 1))), float(np.percentile(fl, 99)))
    in_band = finite & (lam > wl_band[0]) & (lam < wl_band[1])
    if int(in_band.sum()) < 8:
        return {"available": False, "reason": f"only {int(in_band.sum())} in-band reflections"}

    qhat = q_theo / np.where(qn == 0, 1.0, qn)
    pred = (np.asarray(U_est, float) @ qhat).T          # (M,3) unit, sample frame

    imap = _intensity_lookup(structure_factors)
    if imap is not None:
        I = np.array([imap.get((int(h), int(k), int(l)),
                               imap.get((-int(h), -int(k), -int(l)), 0.01))
                      for h, k, l in hkl.T], float)
        I = np.where(I > 0, I, 0.01)
    else:
        I = np.ones(hkl.shape[1])
    Lf = (np.where((qn > 0) & (lam > 0), 4.0 * (lam ** 2) / (qn ** 2), 1.0)
          if lorentz else np.ones(hkl.shape[1]))
    if phi is not None:
        spec = np.asarray(phi(lam), float)
        spec = np.where(np.isfinite(spec) & (spec > 0), spec, 0.0)
    else:
        spec = np.ones(hkl.shape[1])

    p = np.where(in_band, 1.0, 0.0) * I * Lf * spec
    p = np.clip(p, 0.0, None)
    mask = in_band & (p > 0)
    if int(mask.sum()) < 8:
        return {"available": False, "reason": f"only {int(mask.sum())} weighted in-band reflections"}

    obs_t = _adev_tensors(q_unit, L_max)                  # uniform over events
    pred_t = _adev_tensors(pred[mask], L_max, p[mask])    # population-weighted

    per_shell = {}
    for l in range(1, L_max + 1):
        Ao, Ap = obs_t[l], pred_t[l]
        no = float(np.linalg.norm(Ao)); npd = float(np.linalg.norm(Ap))
        ov = float(np.sum(Ao * Ap) / (no * npd)) if (no > 0 and npd > 0) else float("nan")
        per_shell[l] = {"obs": no, "pred": npd, "overlap": ov}

    ov2 = [per_shell[l]["overlap"] for l in range(2, L_max + 1)
           if np.isfinite(per_shell[l]["overlap"])]
    return {"available": True, "per_shell": per_shell,
            "mean_overlap_l2": float(np.mean(ov2)) if ov2 else float("nan"),
            "n_pred": int(mask.sum()),
            "wl_band": (float(wl_band[0]), float(wl_band[1])),
            "d_range": (float(d_min), float(d_max)),
            "weighted_phi": phi is not None, "lorentz": bool(lorentz),
            "weighted_I": imap is not None}

# ---------------------------------------------------------------------------
def analyze_event_stream(event_batches, gonio_axes=None, gonio_offsets=None,
                         L_max=8, n_grid=768, mc_baseline=True, label="",
                         cell=None, U_est=None, ki_vec=None,
                         h_max=6, d_min=1.0, d_max=10.0, wl_band=None,
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

    orient_match = _orientation_match_block(
        q_unit, ki_mean, cell, U_est, L_max, h_max, d_min, d_max,
        (spectrum.get("wl_band") if spectrum.get("available") else wl_band),
        structure_factors, lorentz,
        phi=(spectrum.get("phi") if spectrum.get("available") else None))

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
        orientation_match=orient_match,
    )
    _print_report(rep)
    return rep


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

    om = r.get('orientation_match')
    if om is not None:
        print(f"\n seed orientation match (predicted vs observed A_dev at U_est)")
        if not om.get("available"):
            print(f"   [skipped] {om.get('reason')}")
        else:
            dlo, dhi = om["d_range"]; wlo, whi = om["wl_band"]
            print(f"   pool d=[{dlo:.2f},{dhi:.2f}] A  lambda=[{wlo:.2f},{whi:.2f}] A  "
                  f"({om['n_pred']} in-band refl) <- set to your TRACKER's d_min/d_max/wl")
            print(f"   weights: phi={'Y' if om['weighted_phi'] else 'N'} "
                  f"|F|2={'Y' if om['weighted_I'] else 'N'} lorentz={'Y' if om['lorentz'] else 'N'}")
            print(f"   {'l':>2} | {'||A_dev|| obs':>13} | {'||A_dev|| pred':>14} | {'overlap':>8}")
            for l in sorted(om["per_shell"]):
                d = om["per_shell"][l]
                flag = ("  <- aligned" if (l >= 2 and d["overlap"] > 0.8) else
                        "  <- MISMATCH" if (l >= 2 and d["overlap"] < 0.5) else "")
                print(f"   {l:>2} | {d['obs']:>13.3f} | {d['pred']:>14.3f} | {d['overlap']:>8.3f}{flag}")
            mo = om["mean_overlap_l2"]
            verdict = ("seed is a true minimum -> debug optimizer/annealing" if mo > 0.8 else
                       "prediction != observation at seed -> model (d-range/band/spectrum) mismatch"
                       if mo < 0.5 else "marginal -> tighten d-range/band to the data")
            print(f"   mean overlap (l>=2) = {mo:.3f}  <- {verdict}")


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
