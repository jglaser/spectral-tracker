import os
import h5py
import numpy as np
import jax
import jax.numpy as jnp
import scipy.special
import jax.scipy.linalg
import e3x
import gemmi
from functools import partial
from subhkl.instrument.goniometer import sample_to_lab

def skew_symmetric(v):
    """ Computes the 3x3 skew-symmetric matrix cross-product operator. """
    return jnp.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ])


def vector_to_rotation_matrix(omega):
    """ Maps a 3D Lie algebra tangent vector to a proper SO(3) matrix via Rodrigues' formula. """
    theta = jnp.linalg.norm(omega)
    def small_theta():
        return jnp.eye(3) + skew_symmetric(omega)
    def large_theta():
        k = omega / theta
        K = skew_symmetric(k)
        return jnp.eye(3) + jnp.sin(theta) * K + (1.0 - jnp.cos(theta)) * jnp.matmul(K, K)
    return jax.lax.cond(theta < 1e-5, small_theta, large_theta)

def _sample_to_lab_matrix(axes, rep_ang, offsets):
    """
    R (3,3) with v_lab = R @ v_sample at goniometer angles `rep_ang` (num_axes,).
    sample_to_lab applies R_cum to each input row, so feeding the identity basis
    returns R_cum^T (basis images as rows) -> transpose to get R_cum. `offsets`
    are the per-axis zero-point angle calibrations; is_vector=True drops the
    lever-arm translations, leaving a pure rotation.
    """
    axes = np.asarray(axes)
    rep_ang = np.asarray(rep_ang, dtype=float)
    cols = sample_to_lab(np.eye(3), axes, rep_ang, None, zero_offsets=offsets, is_vector=True)
    return np.asarray(cols, dtype=np.float32).T

# ----------------------------------------------------------------------------
# Forward model (renamed *_sample; adds lab-frame visibility mask).
# ----------------------------------------------------------------------------
def predict_all_shells_q_space(
    omega, U_base, q_theo_sample_jax, I_weights_jax, w_l_j, ki_batch, L_max,
    R_batch, cov_coeffs, cov_scale, L_cov, use_coverage, sigmoid_k=10,
):
    """
    ZERO-WIGNER FORWARD MODEL with a data-driven lab-frame detector mask.
    Moments are formed in the SAMPLE frame (== data frame). The visibility
    weight is the only quantity evaluated in the lab frame, via the constant
    per-batch rotation R_batch.
    """
    R_perturb = vector_to_rotation_matrix(omega)
    U_curr = jnp.matmul(U_base, R_perturb)

    # Predicted reflection directions in the SAMPLE frame.
    q_sample = jnp.matmul(U_curr, q_theo_sample_jax).T            # (M, 3)

    # SH features for the moments (sample frame == data frame).
    Y_sample = e3x.so3.irreps.spherical_harmonics(
        q_sample, max_degree=L_max, cartesian_order=False, normalization="orthonormal")

    # --- Ewald window (frame-invariant dot product), unchanged ---
    Y_beam = e3x.so3.irreps.spherical_harmonics(
        ki_batch[0], max_degree=L_max, cartesian_order=False, normalization="orthonormal")
    ewald_window = jnp.zeros(Y_sample.shape[0])
    for l in range(L_max + 1):
        dim_l = 2 * l + 1
        start, end = l ** 2, (l + 1) ** 2
        p_l = jnp.matmul(Y_sample[:, start:end], Y_beam[start:end])
        ewald_window += (w_l_j[l] / float(dim_l)) * p_l
    ewald_window = jnp.clip(ewald_window, 0.0, 1.0) * I_weights_jax

    # --- Data-driven detector visibility, evaluated in the LAB frame ---
    q_lab_pred = jnp.matmul(R_batch, q_sample.T).T               # (M, 3)
    Y_cov = e3x.so3.irreps.spherical_harmonics(
        q_lab_pred, max_degree=L_cov, cartesian_order=False, normalization="orthonormal")
    C = jnp.matmul(Y_cov, cov_coeffs)                            # band-limited occupancy

    vis = jax.nn.sigmoid(sigmoid_k*(C/cov_scale - 1.0))
    # Warm-up (no coverage estimate yet) -> vis == 1 (full-sphere behaviour).
    vis = jnp.where(use_coverage > 0.5, vis, jnp.ones_like(vis))

    weight = ewald_window * vis
    total_window_mass = jnp.maximum(jnp.sum(weight), 1.0)

    preds = []
    for l in range(1, L_max + 1):
        dim_l = 2 * l + 1
        start, end = l ** 2, (l + 1) ** 2
        Y_l = Y_sample[:, start:end]

        z_1st = jnp.sum(Y_l * weight[:, None], axis=0) / total_window_mass

        A_sig = jnp.matmul(Y_l.T, Y_l * weight[:, None]) / total_window_mass
        A_sig = A_sig * (4.0 * jnp.pi / float(dim_l))
        A_dev = A_sig - (jnp.trace(A_sig) / float(dim_l)) * jnp.eye(dim_l)

        preds.append(z_1st)
        preds.append(A_dev.flatten())

    return jnp.concatenate(preds)


# ----------------------------------------------------------------------------
# Kalman update (renamed *_sample; threads coverage args; L_cov is static).
# ----------------------------------------------------------------------------
@partial(jax.jit, static_argnames=["L_max", "L_cov"])
def kalman_subspace_update(
    P_prev, A_sample_all, Y_events_sample, Y_sample_all_sum, q_theo_sample_jax, I_weights_jax,
    w_l_j, ki_batch, U_base, process_q_scale, dt, ridge_inflation, meas_noise_1st,
    meas_weight_2nd, num_events, L_max, low_l_damp,
    R_batch, cov_coeffs, cov_scale, L_cov, use_coverage, sigmoid_k
):
    """Consumes precomputed WEIGHTED sufficient statistics (A_lab_all,
    Y_lab_all_sum) and an effective (gated) event count."""
    omega_state = jnp.zeros(3)
    P_state = P_prev + jnp.eye(3) * (process_q_scale * dt)

    z_data_list, R_diag_list = [], []
    for l in range(1, L_max + 1):
        dim_l = 2 * l + 1
        start, end = l ** 2, (l + 1) ** 2

        A_sample_norm = A_sample_all[start:end, start:end] * (4.0 * jnp.pi / float(dim_l))
        A_sample_dev = A_sample_norm - jnp.eye(dim_l) / float(dim_l)
        sigma_l = (meas_noise_1st * (l * (l + 1) + 1.0)) / (num_events * float(dim_l)) + ridge_inflation

        # Suppress the dipole shell (lobe lives here; centrosymmetric signal does not).
        if l == 1:
            sigma_l = sigma_l * low_l_damp

        z_1st_norm = Y_sample_all_sum[start:end] / num_events
        z_data_list.append(z_1st_norm)
        R_diag_list.append(jnp.full(dim_l, sigma_l))

        z_data_list.append(A_sample_dev.flatten())
        R_diag_list.append(jnp.full(dim_l * dim_l, sigma_l * meas_weight_2nd / float(dim_l)))

    z_data = jnp.concatenate(z_data_list)
    R_diag = jnp.concatenate(R_diag_list)

    # Closure so jacfwd differentiates ONLY omega; coverage args held constant.
    def fwd(om):
        return predict_all_shells_q_space(
            om, U_base, q_theo_sample_jax, I_weights_jax, w_l_j, ki_batch, L_max,
            R_batch, cov_coeffs, cov_scale, L_cov, use_coverage, sigmoid_k)

    z_pred = fwd(omega_state)
    H_global = jax.jacfwd(fwd)(omega_state)

    # --- Woodbury / information-form update ---
    R_eff_inv = 1.0 / (R_diag + ridge_inflation)
    Ht_Rinv = H_global.T * R_eff_inv[None, :]

    P_inv = jnp.linalg.pinv(P_state + jnp.eye(3) * 1e-9)
    information_matrix = P_inv + jnp.matmul(Ht_Rinv, H_global)
    K_gain = jnp.matmul(jnp.linalg.pinv(information_matrix), Ht_Rinv)

    omega_update = jnp.matmul(K_gain, (z_data - z_pred))

    P_new = jnp.linalg.pinv(information_matrix)
    P_new = 0.5 * (P_new + P_new.T)

    U_new = jnp.matmul(U_base, vector_to_rotation_matrix(omega_update))
    return U_new, P_new, z_pred, z_data


# ----------------------------------------------------------------------------
# Per-chunk processor (renamed *_sample; threads coverage args; bug fixed).
# ----------------------------------------------------------------------------
def process_chunk_field_kalman(
    P_prev, q_batch, ki_batch, t_batch,
    q_theo_sample_jax, w_l_j, I_weights_jax,
    meas_noise_1st, meas_weight_2nd, ridge_inflation, L_max, U_base,
    current_q_scale,
    R_batch, cov_coeffs, cov_scale, L_cov, use_coverage, sigmoid_k,
    cos_gate=0.99, gate_temp=0.003, low_l_damp=1e6,
):
    # NOTE: q_batch is in the SAMPLE frame (loader output).
    dt_chunk = jnp.maximum(1e-4, t_batch[-1] - t_batch[0])
    total_rate = q_batch.shape[0] / dt_chunk

    actual_events = q_batch.shape[0]
    num_events = jnp.maximum(float(actual_events), 1.0)

    Y_events_sample = e3x.so3.irreps.spherical_harmonics(
        q_batch, max_degree=L_max, cartesian_order=False, normalization="orthonormal")

    # --- event-level signal gate -------------------------------------------------
    # Soft proximity of each (unit) event direction to the nearest predicted
    # Bragg direction. Discrimination scales with how sparsely the active
    # reflections tile the sphere -- best with the correct narrow bandpass.
    q_pred = U_base @ q_theo_sample_jax            # (3, M); columns are unit vectors
    cos_sim = q_batch @ q_pred                     # (N, M)
    nn_cos = jnp.max(cos_sim, axis=1)              # (N,)
    w = jax.nn.sigmoid((nn_cos - cos_gate) / gate_temp)  # (N,) in [0, 1]
 
    # WEIGHTED sufficient statistics (background suppressed in the moment itself).
    eff_count = jnp.maximum(jnp.sum(w), 1.0)
    Y_w = Y_events_sample * w[:, None]
    Y_sample_all_sum = jnp.sum(Y_w, axis=0)
    A_sample_all = jnp.matmul(Y_w.T, Y_events_sample) / eff_count   # sum_i w_i Y_i Y_i^T / sum_i w_i

    U_new, P_new, z_pred, z_data = kalman_subspace_update(
        P_prev, A_sample_all, Y_events_sample, Y_sample_all_sum, q_theo_sample_jax, I_weights_jax,
        w_l_j, ki_batch, U_base, current_q_scale, dt_chunk, ridge_inflation,
        meas_noise_1st, meas_weight_2nd, num_events, L_max, low_l_damp,
        R_batch, cov_coeffs, cov_scale, L_cov, use_coverage,
        sigmoid_k
    )

    U_final = U_new if actual_events > 0 else U_base
    P_final = P_new if actual_events > 0 else (P_prev + jnp.eye(3) * (current_q_scale * dt_chunk))

    # A MEANINGFUL signal/background readout: the fraction the gate accepted.
    accepted = float(jnp.sum(w))
    sig_rate = (accepted / max(actual_events, 1)) * total_rate
    bg_rate = jnp.maximum(total_rate - sig_rate, 0.0)

    innovation = z_data - z_pred
    spectral_nll = 0.5 * jnp.sum(jnp.square(innovation))

    return P_final, U_final, spectral_nll, sig_rate, bg_rate


# ============================================================================
# TRACKER LOOP -- coverage state + per-batch R_batch / map update.
# Splice the new/changed pieces below into run_spectral_holonomic_tracker.
# Keep the existing [0/3]..[1/3] setup, the w_l_j / q_theo / I_weights
# construction, U_curr / P_spectral_full init, etc.
#
# New parameters to add to run_spectral_holonomic_tracker's signature:
#     gonio_axes=None, gonio_offsets=None,
#     L_cov: int = 3, cov_ema_weight: float = 0.1,   # keep LOW: see note below
#     cov_scale_percentile: float = 25.0, cov_warmup_events: int = 20000,
# ============================================================================
def _tracker_loop_reference(
    finder_file, event_batches, U_curr, P_spectral_full,
    q_theo_sample_jax, w_l_j, I_weights_jax,
    meas_noise_1st, meas_weight_2nd, ridge_inflation, L_max,
    process_q_scale_start, process_q_scale_end, annealing_rate,
    streaming_callback,
    gonio_axes=None, gonio_offsets=None,
    L_cov: int = 3, cov_ema_weight=0.1, cov_threshold_frac=0.3, cov_warmup_events=20000, sigmoid_k=12,
    q_scale_floor: float = 1e-5,
):
    import h5py
 
    # Fallback: read axes from the finder file if the caller did not pass them.
    if gonio_axes is None:
        with h5py.File(finder_file, "r") as f:
            if "goniometer/axes" in f:
                gonio_axes = f["goniometer/axes"][()]
            if gonio_offsets is None and "goniometer/offsets" in f:
                off = f["goniometer/offsets"]
                gonio_offsets = off[()] if isinstance(off, h5py.Dataset) else None
    has_gonio = gonio_axes is not None and np.asarray(gonio_axes).size > 0
 
    # --- Lab-frame coverage state (host-side, band-limited SH occupancy) ---
    n_cov = (L_cov + 1) * (L_cov + 1)
    cov_coeffs_np = np.zeros(n_cov, dtype=np.float32)
    cov_scale_val = 1.0
    coverage_ready = False
 
    tracking_history = [(0.0, np.array(U_curr))]
 
    for batch_data in event_batches:
        (q_batch_np, t_batch_np, banks_np, pr_np, pc_np,
         angles_np, slab_np, ki_sample_np, cumulative_count) = batch_data
        if len(t_batch_np) == 0:
            continue
 
        t_state = float(t_batch_np[-1])
 
        # ---- build the per-batch sample->lab rotation (host-side) ----
        if has_gonio and angles_np is not None and np.size(angles_np) > 0:
            ang2d = np.atleast_2d(angles_np)                 # (N, num_axes)
            rep_ang = ang2d.mean(axis=0)                     # (num_axes,)
            R_batch_np = _sample_to_lab_matrix(gonio_axes, rep_ang, gonio_offsets)
            q_lab_obs = (R_batch_np @ q_batch_np.T).T
        else:
            R_batch_np = np.eye(3, dtype=np.float32)         # static: sample == lab
            q_lab_obs = q_batch_np
 
        q_lab_obs = q_lab_obs / (np.linalg.norm(q_lab_obs, axis=1, keepdims=True) + 1e-9)
 
        # ---- accumulate the lab-frame occupancy map (EMA) ----
        Y_cov_obs = np.asarray(e3x.so3.irreps.spherical_harmonics(
            jnp.asarray(q_lab_obs, dtype=jnp.float32),
            max_degree=L_cov, cartesian_order=False, normalization="orthonormal"))
        batch_coeffs = Y_cov_obs.mean(axis=0)
        cov_coeffs_np = (1.0 - cov_ema_weight) * cov_coeffs_np + cov_ema_weight * batch_coeffs
 
        if cumulative_count >= cov_warmup_events:
            coverage_ready = True
 
        if coverage_ready:
            C_obs = Y_cov_obs @ cov_coeffs_np                # reconstruction on covered dirs
            med = float(np.median(C_obs[C_obs > 1e-9])) if np.any(C_obs > 1e-9) else 1.0
            # Threshold in the GAP below the covered density: covered dirs have
            # C ~ med (well above), uncovered dirs have C ~ 0 (well below). The
            # sigmoid in predict_all_shells_q_space then gates present-vs-absent
            # rather than carving the within-region density variation. This is
            # what makes the mask inert for full coverage (no acquisition
            # penalty) AND flat across a partial cap (no internal ell=1 ramp).
            cov_scale_val = max(cov_threshold_frac * med, 1e-6)
            use_cov_flag = 1.0
        else:
            use_cov_flag = 0.0 

        # ---- device transfers (all traced inputs, no recompile on value change) ----
        q_batch = jax.device_put(q_batch_np)
        q_batch = q_batch / (jnp.linalg.norm(q_batch, axis=1, keepdims=True) + 1e-9)
        t_batch = jax.device_put(t_batch_np)
        ki_batch = jax.device_put(ki_sample_np)
        ki_batch = ki_batch / (jnp.linalg.norm(ki_batch, axis=1, keepdims=True) + 1e-9)
 
        R_batch = jnp.asarray(R_batch_np)
        cov_coeffs = jnp.asarray(cov_coeffs_np)
        cov_scale = jnp.asarray(np.float32(cov_scale_val))
        use_coverage = jnp.asarray(np.float32(use_cov_flag))
 
        # ---- annealing schedule (unchanged) ----
        progress_fraction = min(t_state / (5.0 * annealing_rate), 1.0)
        current_q_scale = process_q_scale_start * (process_q_scale_end / process_q_scale_start) ** progress_fraction
        current_q_scale = max(current_q_scale, q_scale_floor)

        P_spectral_full, U_curr, spectral_nll, sig_rate, bg_rate = process_chunk_field_kalman(
            P_spectral_full, q_batch, ki_batch, t_batch,
            q_theo_sample_jax, w_l_j, I_weights_jax,
            meas_noise_1st, meas_weight_2nd, ridge_inflation, L_max, U_curr,
            current_q_scale,
            R_batch, cov_coeffs, cov_scale, L_cov, use_coverage,
            sigmoid_k
        )
 
        U_curr.block_until_ready()
        U_best = np.array(U_curr)
        tracking_history.append((t_state, U_best))
 
        norm_gap_metric = float(jnp.trace(jnp.linalg.pinv(P_spectral_full)))
 
        if cumulative_count % 50000 < len(t_batch_np):
            cov_state = "ON" if coverage_ready else "warmup"
            print(f"    Time {t_state:.2f}s | Sig/Bg: {float(sig_rate):.0f}/{float(bg_rate):.0f} Hz "
                  f"| Coherent-Mass: {norm_gap_metric:8.2f} | Coverage: {cov_state}")
 
        if streaming_callback is not None:
            new_events = {"banks": banks_np, "pixel_r": pr_np, "pixel_c": pc_np,
                          "angles": angles_np, "s_lab": slab_np}
            streaming_callback(
                time=t_state, U_preds=np.expand_dims(U_best, axis=0),
                losses=np.array([float(spectral_nll)]), best_idx=0,
                neutron_count=cumulative_count, new_events=new_events,
                metrics={"loss": float(spectral_nll), "eigengap": norm_gap_metric,
                         "sig_rate": float(sig_rate), "bg_rate": float(bg_rate),
                         "coverage_ready": coverage_ready})
 
    return tracking_history

def build_band_weights(q_mags, wl_min, wl_max, L_max, spectrum=None, n_quad=4096):
    """
    Ewald band weights w_l_j of shape (L_max+1, M).

    spectrum=None  -> EXACT original analytic flat (top-hat) weights (no change).
    spectrum=phi   -> weighted Legendre quadrature with assumed shape phi(lambda).
    """
    q = np.asarray(q_mags, dtype=float)
    # Per-reflection x-interval the band covers. x = -0.5*|q|*lambda, so the
    # short-wavelength edge (wl_min) is the LEAST negative x (x_max), and the
    # long-wavelength edge (wl_max) is the MOST negative (x_min). Clip to [-1, 1].
    x_max = np.clip(-0.5 * q * wl_min, -1.0, 1.0)
    x_min = np.clip(-0.5 * q * wl_max, -1.0, 1.0)

    if spectrum is None:
        # ---- exact analytic flat top-hat (preserves the tracker's convention) ----
        P_min = [np.ones_like(q), x_min]
        P_max = [np.ones_like(q), x_max]
        for l in range(1, L_max + 1):
            P_min.append(((2 * l + 1) * x_min * P_min[-1] - l * P_min[-2]) / (l + 1))
            P_max.append(((2 * l + 1) * x_max * P_max[-1] - l * P_max[-2]) / (l + 1))
        w = [0.5 * (x_max - x_min)]
        for l in range(1, L_max + 1):
            w.append(0.5 * (P_max[l + 1] - P_max[l - 1] - (P_min[l + 1] - P_min[l - 1])))
        return jnp.array(np.stack(w, axis=0))

    # ---- weighted quadrature for an arbitrary assumed spectrum ----
    u = (np.arange(n_quad) + 0.5) / n_quad                       # midpoints in (0,1)
    x = x_min[:, None] + (x_max - x_min)[:, None] * u[None, :]    # (M, n_quad)
    dx = ((x_max - x_min) / n_quad)[:, None]                      # (M, 1) >= 0

    with np.errstate(divide="ignore", invalid="ignore"):
        lam = np.where(q[:, None] > 0, -2.0 * x / q[:, None], 0.0)  # wavelength at each node
    phi = np.asarray(spectrum(lam), dtype=float)
    phi = np.where(np.isfinite(phi), phi, 0.0)

    P = [np.ones_like(x), x]
    for l in range(1, L_max + 1):
        P.append(((2 * l + 1) * x * P[-1] - l * P[-2]) / (l + 1))

    w = []
    for l in range(L_max + 1):
        w.append(0.5 * (2 * l + 1) * np.sum(phi * P[l] * dx, axis=1))
    return jnp.array(np.stack(w, axis=0))

def tracker(
    finder_file: str,
    event_batches,
    structure_factors: gemmi.Mtz = None,
    instrument_name: str | None = None,
    assumed_spectrum = None,
    streaming_callback=None,
    process_q_scale_start: float = 1e-3,
    process_q_scale_end: float = 1e-7,
    q_scale_floor: float = 1e-5,
    annealing_rate: float = 1.0,
    h_max: int = 6,
    d_min: float = 2.0,
    d_max: float = 8.0,
    wl_min_tracking: float = 0.5,
    wl_max_tracking: float = 12.0,
    L_max: int = 8,
    prior_ridge: float = 0.15,
    meas_noise_1st: float = 0.5,
    meas_weight_2nd: float = 1.0,
    ridge_inflation: float = 1e-4,
    gonio_axes=None,
    gonio_offsets=None,
    L_cov: int = 3,
    cov_ema_weight: float = 0.1,
    cov_threshold_frac: float = 0.3,
    cov_warmup_events: int = 20000,
    sigmoid_k = 10,
):
    from subhkl.optimization import FindUB

    print(f"[0/3] Preparing Monolithic Lie Algebra Tangent Workspace (SO(3) Dimension=3)...")

    print(f"\n[1/3] Initializing Reciprocal Space from: {finder_file}")
    ub_helper = FindUB()
    U_init = None
    with h5py.File(finder_file, "r") as f:
        ub_helper.a = f["sample/a"][()] if "sample/a" in f else 10.0
        ub_helper.b = f["sample/b"][()] if "sample/b" in f else 10.0
        ub_helper.c = f["sample/c"][()] if "sample/c" in f else 10.0
        ub_helper.alpha = f["sample/alpha"][()] if "sample/alpha" in f else 90.0
        ub_helper.beta = f["sample/beta"][()] if "sample/beta" in f else 90.0
        ub_helper.gamma = f["sample/gamma"][()] if "sample/gamma" in f else 90.0
        sg = f["sample/space_group"][()] if "sample/space_group" in f else b"P 1"
        ub_helper.space_group = sg.decode("utf-8") if isinstance(sg, bytes) else str(sg)

        for key in ["sample/U_init", "sample/initial_U", "sample/U_seed", "orientation/U", "sample/U"]:
            if key in f:
                U_init = f[key][()]
                break

    B_mat = ub_helper.reciprocal_lattice_B()
    h_vals = np.arange(-h_max, h_max + 1)
    hc, kc, lc = np.meshgrid(h_vals, h_vals, h_vals, indexing="ij")
    hkl_c = np.stack([hc.flatten(), kc.flatten(), lc.flatten()], axis=0)
    mask_hkl_c = ~((hkl_c[0] == 0) & (hkl_c[1] == 0) & (hkl_c[2] == 0))
    theo_hkl = hkl_c[:, mask_hkl_c].astype(np.float32)

    q_theo_cryst = np.array(B_mat @ theo_hkl)
    q_mags_np = np.linalg.norm(q_theo_cryst, axis=0)
    res_mask = (q_mags_np < (1.0 / d_min)) & (q_mags_np > (1.0 / d_max))

    q_theo_cryst = q_theo_cryst[:, res_mask]
    q_mags_jax = jnp.array(q_mags_np[res_mask])
    q_theo_sample_jax = jnp.array(q_theo_cryst / np.where(q_mags_np[res_mask] == 0, 1.0, q_mags_np[res_mask]))
    num_peaks = float(q_theo_sample_jax.shape[1])

    M_peaks = q_mags_np[res_mask].shape[0]
    I_weights = np.ones(M_peaks, dtype=np.float32)

    if structure_factors is not None:
        print("[*] Gemmi Structure Factors provided. Mapping physical intensities to Ewald model...")

        # Locate Intensity (I) or Amplitude (F) columns
        h_col = structure_factors.column_with_label('H') or structure_factors.columns[0]
        k_col = structure_factors.column_with_label('K') or structure_factors.columns[1]
        l_col = structure_factors.column_with_label('L') or structure_factors.columns[2]

        i_col = next((c for c in structure_factors.columns if c.type == 'J' or c.label == 'I'), None)

        # Fallback to Amplitude if Intensity isn't explicitly provided
        if i_col is None:
            i_col = next((c for c in structure_factors.columns if c.type == 'F' or c.label == 'F'), None)

        if i_col is not None:
            h_arr, k_arr, l_arr, i_arr = np.array(h_col.array), np.array(k_col.array), np.array(l_col.array), np.array(i_col.array)

            # Convert Amplitude to Intensity if necessary
            if i_col.type == 'F':
                i_arr = i_arr ** 2

            hkl_to_intensity = {(int(h), int(k), int(l)): val for h, k, l, val in zip(h_arr, k_arr, l_arr, i_arr)}

            valid_theo_hkl = theo_hkl[:, res_mask]
            for idx in range(M_peaks):
                h, k, l = int(valid_theo_hkl[0, idx]), int(valid_theo_hkl[1, idx]), int(valid_theo_hkl[2, idx])

                # Check for exact match or Friedel opposite
                if (h, k, l) in hkl_to_intensity:
                    I_weights[idx] = hkl_to_intensity[(h, k, l)]
                elif (-h, -k, -l) in hkl_to_intensity:
                    I_weights[idx] = hkl_to_intensity[(-h, -k, -l)]
                else:
                    I_weights[idx] = 0.01  # Small floor for unmeasured peaks

            # Normalize to preserve overall scaling
            if np.sum(I_weights) > 0:
                I_weights /= np.mean(I_weights)
        else:
            print("    Warning: Could not find Intensity (type J) or Amplitude (type F) column in MTZ. Defaulting to uniform.")

    I_weights_jax = jax.device_put(I_weights)

    w_l_j = build_band_weights(
       q_mags_np[res_mask], wl_min_tracking, wl_max_tracking, L_max,
       spectrum=assumed_spectrum,
    )

    if U_init is not None:
        U_curr = jnp.array(U_init)
    else:
        U_curr = jnp.eye(3)

    P_spectral_full = jnp.eye(3) * prior_ridge

    print(f"\n[2/3] Executing Field Tracker Pipeline (3 Local Subspace Rotational Degrees Active)...")

    tracking_history = _tracker_loop_reference(
        finder_file, event_batches, U_curr, P_spectral_full,
        q_theo_sample_jax, w_l_j, I_weights_jax,
        meas_noise_1st, meas_weight_2nd, ridge_inflation, L_max,
        process_q_scale_start, process_q_scale_end, annealing_rate,
        streaming_callback,
        gonio_axes, gonio_offsets,
        L_cov, cov_ema_weight, cov_threshold_frac, cov_warmup_events, sigmoid_k,
        q_scale_floor
    )

    print(f"\n[3/3] Global Tracking complete. Saving continuous SO(3) state dataset.")
    with h5py.File(finder_file, "a") as f:
        if "tracking" in f: del f["tracking"]
        group = f.create_group("tracking")
        group.create_dataset("final_u_matrix", data=tracking_history[-1][1])
        group.create_dataset("timestamps", data=np.array([h[0] for h in tracking_history]))

    return tracking_history[-1][1]
