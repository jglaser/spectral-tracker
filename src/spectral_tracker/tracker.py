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
    ], dtype=v.dtype)

def vector_to_rotation_matrix(omega):
    """ Maps a 3D Lie algebra tangent vector to a proper SO(3) matrix via Rodrigues' formula. """
    theta = jnp.linalg.norm(omega)
    def small_theta():
        return jnp.eye(3, dtype=omega.dtype) + skew_symmetric(omega)
    def large_theta():
        k = omega / theta
        K = skew_symmetric(k)
        return jnp.eye(3, dtype=omega.dtype) + jnp.sin(theta) * K + (1.0 - jnp.cos(theta)) * jnp.matmul(K, K)
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
    # Add dtype enforcement to arrays initialized within the forward model
    sh_dtype = omega.dtype

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
    ewald_window = jnp.zeros(Y_sample.shape[0], dtype=sh_dtype)
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
    total_window_mass = jnp.maximum(jnp.sum(weight), 1e-6)

    preds = []
    for l in range(1, L_max + 1):
        dim_l = 2 * l + 1
        start, end = l ** 2, (l + 1) ** 2
        Y_l = Y_sample[:, start:end]

        z_1st = jnp.sum(Y_l * weight[:, None], axis=0) / total_window_mass

        A_sig = jnp.matmul(Y_l.T, Y_l * weight[:, None]) / total_window_mass
        A_sig = A_sig * (4.0 * jnp.pi / float(dim_l))
        A_dev = A_sig - (jnp.trace(A_sig) / float(dim_l)) * jnp.eye(dim_l, dtype=sh_dtype)

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
    meas_weight_2nd, num_events, L_max, low_l_damp, damp_below_l,
    R_batch, cov_coeffs, cov_scale, L_cov, use_coverage, sigmoid_k
):
    """Consumes precomputed WEIGHTED sufficient statistics (A_lab_all,
    Y_lab_all_sum) and an effective (gated) event count."""
    sh_dtype = P_prev.dtype
    omega_state = jnp.zeros(3, dtype=sh_dtype)
    P_state = P_prev + jnp.eye(3, dtype=sh_dtype) * (process_q_scale * dt)

    z_data_list, R_diag_list = [], []
    for l in range(1, L_max + 1):
        dim_l = 2 * l + 1
        start, end = l ** 2, (l + 1) ** 2

        A_sample_norm = A_sample_all[start:end, start:end] * (4.0 * jnp.pi / float(dim_l))
        A_sample_dev = A_sample_norm - jnp.eye(dim_l, dtype=sh_dtype) / float(dim_l)
        sigma_l = (meas_noise_1st * (l * (l + 1) + 1.0)) / (num_events * float(dim_l)) + ridge_inflation

        # Suppress the low shells where the cap / smooth background lives.
        sigma_l = jnp.where(l < damp_below_l, sigma_l * low_l_damp, sigma_l)

        z_1st_norm = Y_sample_all_sum[start:end] / num_events
        z_data_list.append(z_1st_norm)
        R_diag_list.append(jnp.full(dim_l, sigma_l, dtype=sh_dtype))

        z_data_list.append(A_sample_dev.flatten())
        R_diag_list.append(jnp.full(dim_l * dim_l, sigma_l * meas_weight_2nd / float(dim_l), dtype=sh_dtype))

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

    P_inv = jnp.linalg.pinv(P_state + jnp.eye(3, dtype=sh_dtype) * 1e-9)
    information_matrix = P_inv + jnp.matmul(Ht_Rinv, H_global)
    K_gain = jnp.matmul(jnp.linalg.pinv(information_matrix), Ht_Rinv)

    omega_update = jnp.matmul(K_gain, (z_data - z_pred))

    P_new = jnp.linalg.pinv(information_matrix)
    P_new = 0.5 * (P_new + P_new.T)

    U_new = jnp.matmul(U_base, vector_to_rotation_matrix(omega_update))

    # 1. Condition of the Information Matrix (Hessian / Stiffness)
    info_eigvals = jnp.linalg.eigvalsh(information_matrix)

    # 2. The raw Gradient pulling the state (before stiffness restricts it)
    raw_gradient = jnp.matmul(Ht_Rinv, (z_data - z_pred))

    diagnostics = {
        "omega_step_deg": jnp.linalg.norm(omega_update) * (180.0 / jnp.pi),
        "grad_norm": jnp.linalg.norm(raw_gradient),
        "info_eigval_min": info_eigvals[0],
        "info_eigval_max": info_eigvals[-1],
        "z_data_norm": jnp.linalg.norm(z_data),
        "z_pred_norm": jnp.linalg.norm(z_pred)
    }

    return U_new, P_new, z_pred, z_data, diagnostics


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
    use_gate=True, damp_below_l=2,
):
    sh_dtype = q_batch.dtype

    # NOTE: q_batch is in the SAMPLE frame (loader output).
    dt_chunk = jnp.maximum(1e-4, t_batch[-1] - t_batch[0])
    total_rate = q_batch.shape[0] / dt_chunk

    actual_events = q_batch.shape[0]
    num_events = jnp.maximum(float(actual_events), 1.0)

    Y_events_sample = e3x.so3.irreps.spherical_harmonics(
        q_batch, max_degree=L_max, cartesian_order=False, normalization="orthonormal")

    # --- event-level signal gate (optional) ---
    if use_gate:
        q_pred = U_base @ q_theo_sample_jax
        nn_cos = jnp.max(q_batch @ q_pred, axis=1)
        w = jax.nn.sigmoid((nn_cos - cos_gate) / gate_temp)
    else:
        w = jnp.ones(q_batch.shape[0], dtype=sh_dtype)

    # WEIGHTED sufficient statistics (background suppressed in the moment itself).
    eff_count = jnp.maximum(jnp.sum(w), 1.0)
    Y_w = Y_events_sample * w[:, None]
    Y_sample_all_sum = jnp.sum(Y_w, axis=0)
    A_sample_all = jnp.matmul(Y_w.T, Y_events_sample) / eff_count   # sum_i w_i Y_i Y_i^T / sum_i w_i

    U_new, P_new, z_pred, z_data, step_diags = kalman_subspace_update(
        P_prev, A_sample_all, Y_events_sample, Y_sample_all_sum, q_theo_sample_jax, I_weights_jax,
        w_l_j, ki_batch, U_base, current_q_scale, dt_chunk, ridge_inflation,
        meas_noise_1st, meas_weight_2nd, num_events, L_max, low_l_damp, damp_below_l,
        R_batch, cov_coeffs, cov_scale, L_cov, use_coverage,
        sigmoid_k
    )

    U_final = U_new if actual_events > 0 else U_base
    P_final = P_new if actual_events > 0 else (P_prev + jnp.eye(3, dtype=sh_dtype) * (current_q_scale * dt_chunk))

    # Handle zero-event batches safely
    if actual_events == 0:
        step_diags = {k: 0.0 for k in step_diags.keys()}

    # A MEANINGFUL signal/background readout: the fraction the gate accepted.
    accepted = float(jnp.sum(w))
    sig_rate = (accepted / max(actual_events, 1)) * total_rate
    bg_rate = jnp.maximum(total_rate - sig_rate, 0.0)

    innovation = z_data - z_pred
    spectral_nll = 0.5 * jnp.sum(jnp.square(innovation))

    return P_final, U_final, spectral_nll, sig_rate, bg_rate, step_diags


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
    cos_gate=0.99, gate_temp=0.003, low_l_damp=1e6, use_gate=True, damp_below_l=2,
    use_coverage_mask=True,
    sh_dtype=None,
    measurement="spectral", q_mags_jax=None,
    wl_min_tracking=0.5, wl_max_tracking=12.0,
    tau_deg_start=8.0, tau_deg_end=0.15, tau_anneal_batches=40,
    sigma_deg=0.2, max_step_scale=0.5, p_floor_deg=0.05,
    d_min_start=None, d_min_end=2.0, refine_schedule=None,
):
    import h5py

    if sh_dtype is None:
        sh_dtype = jnp.result_type(float)
 
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
    events_seen = 0
    n_batches = 0
    stage, stage_batches = 0, 0

    tracking_history = [(0.0, np.array(U_curr))]
 
    for batch_data in event_batches:
        (q_batch_np, t_batch_np, banks_np, pr_np, pc_np,
         angles_np, slab_np, ki_sample_np, cumulative_count) = batch_data
        if len(t_batch_np) == 0:
            continue
 
        t_state = float(t_batch_np[-1])
        events_seen += len(t_batch_np)   # NEW

        if measurement == "correspondence":
            n_batches += 1
            q_batch = jax.device_put(q_batch_np).astype(sh_dtype)
            q_batch = q_batch / (jnp.linalg.norm(q_batch, axis=1, keepdims=True) + 1e-9)
            ki_batch = jax.device_put(ki_sample_np).astype(sh_dtype)
            ki_batch = ki_batch / (jnp.linalg.norm(ki_batch, axis=1, keepdims=True) + 1e-9)
            t_batch = jax.device_put(t_batch_np).astype(sh_dtype)

            if refine_schedule:
                # Hold each (resolution, window) pair for a fixed number of
                # batches, then step to the next and stay on the last one.
                # A continuous anneal fails here: at every intermediate setting
                # the correspondences have to actually settle before the shell
                # opens, and one Gauss-Newton step per batch is not enough.
                while (stage < len(refine_schedule) - 1
                       and stage_batches >= refine_schedule[stage][2]):
                    stage += 1
                    stage_batches = 0
                stage_batches += 1
                d_cur, tau_deg = refine_schedule[stage][0], refine_schedule[stage][1]
            else:
                # Continuous fallback. The window has to start above the seed
                # misorientation or nothing is inside capture range and the step
                # averages to zero, and end near the spot width or it keeps
                # admitting neighbouring reflections. Measured to be worse than
                # the staged schedule above on real data -- kept for tuning.
                frac = min(n_batches / max(tau_anneal_batches, 1), 1.0)
                tau_deg = tau_deg_start * (tau_deg_end / tau_deg_start) ** frac
                d_cur = d_min_start * (d_min_end / d_min_start) ** frac
            q_max = 1.0 / d_cur
            progress_fraction = min(t_state / (5.0 * annealing_rate), 1.0)
            current_q_scale = max(
                process_q_scale_start
                * (process_q_scale_end / process_q_scale_start) ** progress_fraction,
                q_scale_floor)

            P_spectral_full, U_curr, n_acc, step_diags = process_chunk_correspondence(
                P_spectral_full, q_batch, ki_batch, t_batch,
                q_theo_sample_jax, q_mags_jax, U_curr, current_q_scale,
                wl_min_tracking, wl_max_tracking,
                np.radians(tau_deg), np.radians(sigma_deg),
                np.radians(max_step_scale * tau_deg), np.radians(p_floor_deg),
                q_max)
            U_curr.block_until_ready()
            U_best = np.array(U_curr)
            tracking_history.append((t_state, U_best))

            if streaming_callback is not None:
                metrics = {
                    "loss": 0.0,
                    "eigengap": float(jnp.trace(jnp.linalg.pinv(P_spectral_full))),
                    "sig_rate": float(n_acc), "bg_rate": 0.0,
                    "coverage_ready": True, "tau_deg": tau_deg, "d_min_cur": d_cur,
                    "stage": stage,
                    **{k: float(v) for k, v in step_diags.items()},
                }
                streaming_callback(
                    time=t_state, U_preds=np.expand_dims(U_best, axis=0),
                    losses=np.array([0.0]), best_idx=0,
                    neutron_count=cumulative_count,
                    new_events={"banks": banks_np, "pixel_r": pr_np, "pixel_c": pc_np,
                                "angles": angles_np, "s_lab": slab_np},
                    metrics=metrics)
            continue


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
            jnp.asarray(q_lab_obs, dtype=sh_dtype),
            max_degree=L_cov, cartesian_order=False, normalization="orthonormal"))
        batch_coeffs = Y_cov_obs.mean(axis=0)
        cov_coeffs_np = (1.0 - cov_ema_weight) * cov_coeffs_np + cov_ema_weight * batch_coeffs
 
        if events_seen >= cov_warmup_events:
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
            use_cov_flag = 1.0 if use_coverage_mask else 0.0
        else:
            use_cov_flag = 0.0 

        # Do not move U until the coverage map is representative. During warmup
        # vis==1 (full-sphere) OR the map is unrepresentative (panel ramp / mid-
        # stream resume); against cap-confined data that is a coverage-shaped
        # innovation that rotates U off a correct seed.
        if not coverage_ready:
            tracking_history.append((t_state, np.array(U_curr)))
            if streaming_callback is not None:
                new_events = {"banks": banks_np, "pixel_r": pr_np, "pixel_c": pc_np,
                              "angles": angles_np, "s_lab": slab_np}
                streaming_callback(time=t_state, U_preds=np.expand_dims(np.array(U_curr), 0),
                                   losses=np.array([0.0]), best_idx=0,
                                   neutron_count=cumulative_count, new_events=new_events,
                                   metrics={"loss": 0.0, "eigengap": 0.0, "sig_rate": 0.0,
                                            "bg_rate": 0.0, "coverage_ready": False})
            continue

        # Enforce target dtype on device transfers
        q_batch = jax.device_put(q_batch_np).astype(sh_dtype)
        q_batch = q_batch / (jnp.linalg.norm(q_batch, axis=1, keepdims=True) + 1e-9)
        t_batch = jax.device_put(t_batch_np).astype(sh_dtype)
        ki_batch = jax.device_put(ki_sample_np).astype(sh_dtype)
        ki_batch = ki_batch / (jnp.linalg.norm(ki_batch, axis=1, keepdims=True) + 1e-9)
 
        R_batch = jnp.asarray(R_batch_np, dtype=sh_dtype)
        cov_coeffs = jnp.asarray(cov_coeffs_np, dtype=sh_dtype)
        cov_scale = jnp.asarray(np.float32(cov_scale_val), dtype=sh_dtype)
        use_coverage = jnp.asarray(np.float32(use_cov_flag), dtype=sh_dtype)

        # ---- annealing schedule (unchanged) ----
        progress_fraction = min(t_state / (5.0 * annealing_rate), 1.0)
        current_q_scale = process_q_scale_start * (process_q_scale_end / process_q_scale_start) ** progress_fraction
        current_q_scale = max(current_q_scale, q_scale_floor)

        P_spectral_full, U_curr, spectral_nll, sig_rate, bg_rate, step_diags = process_chunk_field_kalman(
            P_spectral_full, q_batch, ki_batch, t_batch,
            q_theo_sample_jax, w_l_j, I_weights_jax,
            meas_noise_1st, meas_weight_2nd, ridge_inflation, L_max, U_curr,
            current_q_scale,
            R_batch, cov_coeffs, cov_scale, L_cov, use_coverage,
            sigmoid_k,
            cos_gate, gate_temp, low_l_damp,
            use_gate, damp_below_l,
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

            metrics = {
                "loss": float(spectral_nll),
                "eigengap": norm_gap_metric,
                "sig_rate": float(sig_rate),
                "bg_rate": float(bg_rate),
                "coverage_ready": coverage_ready,
                "omega_step_deg": float(step_diags["omega_step_deg"]),
                "grad_norm": float(step_diags["grad_norm"]),
                "info_eigval_min": float(step_diags["info_eigval_min"]),
                "info_eigval_max": float(step_diags["info_eigval_max"]),
                "z_data_norm": float(step_diags["z_data_norm"]),
                "z_pred_norm": float(step_diags["z_pred_norm"]),
            }
            streaming_callback(
                time=t_state, U_preds=np.expand_dims(U_best, axis=0),
                losses=np.array([float(spectral_nll)]), best_idx=0,
                neutron_count=cumulative_count, new_events=new_events,
                metrics=metrics)

    return tracking_history

# ----------------------------------------------------------------------------
# Correspondence measurement.
#
# The spectral measurement above matches the degree <= L_max moments of the
# event-direction distribution. Its angular resolution is ~180/L_max degrees,
# and it only carries orientation information while the excited reflections are
# few enough to make those moments anisotropic. Both conditions hold for the
# 10 A cubic cell in tests/ (~150 reflections in band, 4.6 deg spot width) and
# neither holds for a macromolecular cell: T4 lysozyme (61.5, 61.5, 95.9,
# P3_221) excites ~10^4 reflections between 2.8 and 4.5 A, whose l <= 8 moments
# are isotropic to ~0.1%, while the spots themselves are ~0.1 deg wide. Measured
# on CG4D_1808: ||z_data - z_pred|| changes by +0.07% at 1 deg from the indexing
# solution and is LOWER at 32 deg -- no usable gradient and a deeper false
# minimum, which is what makes the tracker walk from 3 deg to 35 deg.
#
# The measurement below instead assigns each event to its nearest predicted
# reflection and solves the resulting linear least-squares problem for the
# sample-frame rotation vector. It has no intrinsic resolution limit and returns
# a proper information matrix, so it drops straight into the Kalman state.
# ----------------------------------------------------------------------------
@partial(jax.jit, static_argnames=[])
def correspondence_normal_equations(q_obs, p_pred, active, tau, sigma):
    """Gauss-Newton normal equations for the sample-frame rotation vector.

    For an event q matched to a prediction p, a small rotation exp([w]x) moves
    p by w x p, so the tangential residual gives (w x p) = q - (q.p) p. Stacking
    the per-event normal equations of that over all (event, candidate) pairs:

        A = sum_ij w_ij (I - p_j p_j^T)      b = sum_ij w_ij (p_j x q_i)

    and w_LS = A^-1 b. Both sums factor through per-candidate quantities, so the
    cost is O(N*M) for the correlation and O(M) after -- never O(N*M*3).

    Weights are a Gaussian in the match angle, normalised per event and then
    scaled by that event's best weight. Per-event normalisation stops one event
    with a very close candidate from dominating; the acceptance factor stops
    background events -- which have no close candidate and would otherwise still
    contribute unit weight spread over distant ones -- from contributing at all.

    NOTE ON SIGN: q = k_f - k_i is a signed vector and `active` already keeps
    only the scattering-side half of the candidates, so there is no antipodal
    ambiguity to fold. Folding it (as a projective nearest-neighbour test would)
    doubles the candidate density and halves the discriminant.
    """
    inv2t2 = 1.0 / (0.5 * tau ** 2)
    # Inactive candidates are pushed to cos = -1 rather than masked after the
    # exponential, so the row maximum below is always a real candidate.
    #
    # PRECISION IS NOT OPTIONAL HERE. jax's default matmul precision resolves to
    # bfloat16 on this GPU, giving max |d cos| = 6.7e-4 between unit vectors.
    # The quantity being measured is 1 - cos(theta): 1.4e-5 at 0.3 deg, 1.4e-3
    # at 3 deg. So the default error is 49x the signal at the accuracy this is
    # meant to reach, and comparable to it even at the widest window; the
    # weights are exp() of that difference times up to 2.6e4, so at tau = 0.5
    # deg the default randomises them by e^17. It survived at all only because
    # at d_min = 8 A the candidates are ~20 deg apart, so the assignment stays
    # right even when the weights are noise.
    c = jnp.clip(jnp.matmul(q_obs, p_pred.T, precision=jax.lax.Precision.HIGHEST),
                 -1.0, 1.0)                               # (N, M) cosines
    c = jnp.where(active[None, :] > 0, c, -1.0)
    cmax = jnp.max(c, axis=1)                             # (N,)
    # Shift before exponentiating: 1/(0.5 tau^2) reaches ~1e5 at tau = 0.15 deg,
    # so exp((c-1)*inv2t2) overflows to inf on the float32 round-off that puts c
    # a part in 1e6 above 1, and inf/inf then poisons A and b with NaN.
    w = jnp.exp((c - cmax[:, None]) * inv2t2) * active[None, :]
    acc = jnp.exp((cmax - 1.0) * inv2t2)                  # per-event acceptance
    w = w / (jnp.sum(w, axis=1, keepdims=True) + 1e-30) * acc[:, None]

    W = jnp.sum(w, axis=0)                                # (M,) mass per candidate
    v = jnp.matmul(w.T, q_obs)                            # (M, 3)

    A = (jnp.eye(3, dtype=q_obs.dtype) * jnp.sum(W)
         - jnp.einsum("m,mi,mj->ij", W, p_pred, p_pred))
    b = jnp.sum(jnp.cross(p_pred, v), axis=0)
    inv_var = 1.0 / (sigma ** 2)
    return A * inv_var, b * inv_var, jnp.sum(acc)


# Validated on CG4D_1808 (T4 lysozyme, P3_221, 61.5/61.5/95.9). Each entry is
# (resolution limit in A, match window in deg, batches to hold it for).
#
# Both ends are pinned by measurement. The window cannot start wider than ~3 deg:
# at 6 deg it drags an already-correct seed out to 5.3 deg, because once the
# window is comparable with the spacing between candidates the weighted
# assignment stops being dominated by the right one. The shell cannot start
# coarser than ~8 A either: at 12 A only ~18 reflections are in band, too few to
# constrain three angles against 1762 observed spots, and it biases a perfect
# seed to 1.2 deg before the finer stages pull it back. Starting at 6 A instead
# is worse still (13 deg from a perfect seed) -- ~315 candidates against the
# same spots is already enough decoys to capture the fit.
#
# Together those bound the capture range at 2-3 deg, which is a property of the
# measurement, not of the schedule: recovering from a worse seed needs a global
# search, i.e. re-indexing.
DEFAULT_REFINE_SCHEDULE = ((8.0, 3.0, 25), (8.0, 1.5, 25),
                           (8.0, 0.8, 25), (6.0, 0.5, 25))


def process_chunk_correspondence(
    P_prev, q_batch, ki_batch, t_batch, q_theo_sample_jax, q_mags_jax,
    U_base, current_q_scale, wl_min, wl_max, tau, sigma, max_step, p_floor,
    q_max, min_accepted=8.0,
):
    """One Kalman step from the correspondence measurement.

    The state is the sample-frame rotation vector, so the update is applied on
    the LEFT (U <- exp([w]x) U). That differs from the spectral path, which
    perturbs in the crystal frame; the two are exclusive.

    Two details do all the work in making this stable on real data:

    * The per-event measurement noise is max(sigma, tau), not sigma. While the
      window is wide most assignments are wrong, so the batch deserves little
      confidence; the resulting gain A/(A + P^-1) damps the step exactly the way
      a Levenberg parameter would. Using the final sigma throughout instead
      makes every batch look like an exact measurement, and the first coarse
      batch then locks the state onto whatever the wrong assignments implied.
    * P is floored at p_floor^2 each batch. Without it the information keeps
      accumulating, the gain decays like 1/n_batches, and the estimate freezes
      at the coarse-window answer instead of following tau down.
    """
    sh_dtype = q_batch.dtype
    dt_chunk = jnp.maximum(1e-4, t_batch[-1] - t_batch[0])
    eye = jnp.eye(3, dtype=sh_dtype)

    p_pred = jnp.matmul(U_base, q_theo_sample_jax).T                  # (M, 3)
    s0 = ki_batch[0] / (jnp.linalg.norm(ki_batch[0]) + 1e-30)

    # Elastic condition: lambda = -2 (p_hat . s0) / |q|, and only the half with
    # p_hat . s0 < 0 can scatter at all.
    dot = jnp.matmul(p_pred, s0)
    lam = -2.0 * dot / jnp.maximum(q_mags_jax, 1e-30)
    # q_max walks the resolution limit inwards->outwards over the run. The match
    # window only rejects background while it is well inside the mean spacing
    # between candidates: at d_min = 5 A this cell puts ~670 reflections in band,
    # i.e. 7.8 deg apart, so a 7.5 deg window accepts every background event and
    # the least-squares solution just follows the densest part of the detector.
    # Opening the shell gradually keeps tau/spacing small throughout.
    active = ((dot < 0) & (lam > wl_min) & (lam < wl_max)
              & (q_mags_jax <= q_max)).astype(sh_dtype)

    sigma_eff = jnp.maximum(sigma, tau)
    A, b, n_acc = correspondence_normal_equations(
        q_batch, p_pred, active, tau, sigma_eff)

    P_state = P_prev + eye * jnp.maximum(current_q_scale * dt_chunk, p_floor ** 2)
    information_matrix = jnp.linalg.inv(P_state) + A
    P_new = jnp.linalg.inv(information_matrix + eye * 1e-12)
    P_new = 0.5 * (P_new + P_new.T)
    omega = jnp.matmul(P_new, b)

    # Trust region: the linearisation only holds while |w| stays inside the
    # match window, and a batch that accepted almost nothing must not move U.
    step = jnp.linalg.norm(omega)
    omega = jnp.where(step > max_step, omega * (max_step / (step + 1e-30)), omega)
    omega = jnp.where(jnp.isfinite(omega).all() & (n_acc > min_accepted),
                      omega, jnp.zeros_like(omega))

    U_new = jnp.matmul(vector_to_rotation_matrix(omega), U_base)

    diagnostics = {
        "omega_step_deg": jnp.linalg.norm(omega) * (180.0 / jnp.pi),
        "grad_norm": jnp.linalg.norm(b),
        "info_eigval_min": jnp.linalg.eigvalsh(information_matrix)[0],
        "info_eigval_max": jnp.linalg.eigvalsh(information_matrix)[-1],
        "n_active": jnp.sum(active),
        "n_accepted": n_acc,
    }
    return P_new, U_new, n_acc, diagnostics


def build_band_weights(q_mags, wl_min, wl_max, L_max, spectrum=None, lorentz=True, n_quad=4096):
    """
    Ewald band weights w_l_j of shape (L_max+1, M).

    spectrum=None  -> EXACT original analytic flat (top-hat) weights.
    spectrum=phi   -> weighted Legendre quadrature with assumed shape phi(lambda)
                      and optional kinematic Lorentz correction.
    """
    q = np.asarray(q_mags, dtype=float)
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

    if lorentz:
        # Fold the kinematic Lorentz factor (4 * lambda^2 / q^2) into the spectrum weight
        L_factor = np.where(q[:, None] > 0, 4.0 * (lam ** 2) / (q[:, None] ** 2), 0.0)
        phi = phi * L_factor

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
    lorentz_correction: bool = False,
    streaming_callback=None,
    process_q_scale_start: float = 1e-3,
    process_q_scale_end: float = 1e-7,
    q_scale_floor: float = 1e-5,
    annealing_rate: float = 1.0,
    h_max: int | None = None,
    d_min: float = 2.0,
    d_max: float = 8.0,
    wl_min_tracking: float | None = None,
    wl_max_tracking: float | None = None,
    # "spectral" matches SH moments of the event-direction distribution; it is
    # what tests/ exercises and is only informative when the number of excited
    # reflections is small enough to keep those moments anisotropic (~10 A cell).
    # "correspondence" assigns events to predicted reflections and is what a
    # macromolecular cell needs -- see the block above
    # correspondence_normal_equations for the measurement that shows why.
    measurement: str = "spectral",
    tau_deg_start: float = 8.0,
    tau_deg_end: float = 0.15,
    tau_anneal_batches: int = 40,
    sigma_deg: float = 0.2,
    max_step_scale: float = 0.5,
    p_floor_deg: float = 0.05,
    d_min_start: float | None = None,
    refine_schedule=DEFAULT_REFINE_SCHEDULE,
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
    cos_gate=0.99, gate_temp=0.003, low_l_damp=1e6,
    sigmoid_k = 10,
    use_gate=False,
    damp_below_l=3,
    use_coverage_mask=True,
    sh_dtype=None,
):
    from subhkl.optimization import FindUB

    if sh_dtype is None:
        sh_dtype = jnp.result_type(float)

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

        # The band the instrument can actually excite. Tracking with a band far
        # wider than the real one admits reflections that were never in the data
        # -- for CG4D the file says [2.8, 4.5] A while the old defaults spanned
        # [0.5, 12.0], roughly five times too wide in lambda.
        if "instrument/wavelength" in f:
            wl_file = np.ravel(f["instrument/wavelength"][()])
            if wl_min_tracking is None:
                wl_min_tracking = float(np.min(wl_file))
            if wl_max_tracking is None:
                wl_max_tracking = float(np.max(wl_file))
    if wl_min_tracking is None:
        wl_min_tracking = 0.5
    if wl_max_tracking is None:
        wl_max_tracking = 12.0
    print(f"    tracking wavelength band: [{wl_min_tracking}, {wl_max_tracking}] A")

    B_mat = ub_helper.reciprocal_lattice_B()

    # A single cubic |h| <= h_max box is only complete when every cell edge is
    # shorter than h_max * d_min. |h_i| <= |a_i| / d_min is the exact bound, and
    # for (61.5, 61.5, 95.9) at d_min = 2 A it is (31, 31, 48) -- the old
    # default of 6 silently kept ~0.2% of the reflections, biased towards low l.
    if h_max is None:
        A_real = np.linalg.inv(B_mat).T
        h_bounds = np.ceil(np.linalg.norm(A_real, axis=0) / d_min).astype(int)
    else:
        h_bounds = np.full(3, int(h_max))
    print(f"    hkl enumeration bounds: {tuple(int(x) for x in h_bounds)}")
    hc, kc, lc = np.meshgrid(*[np.arange(-b, b + 1) for b in h_bounds], indexing="ij")
    hkl_c = np.stack([hc.flatten(), kc.flatten(), lc.flatten()], axis=0)
    mask_hkl_c = ~((hkl_c[0] == 0) & (hkl_c[1] == 0) & (hkl_c[2] == 0))
    theo_hkl = hkl_c[:, mask_hkl_c].astype(np.float32)

    q_theo_cryst = np.array(B_mat @ theo_hkl)
    q_mags_np = np.linalg.norm(q_theo_cryst, axis=0)
    res_mask = (q_mags_np < (1.0 / d_min)) & (q_mags_np > (1.0 / d_max))

    q_theo_cryst = q_theo_cryst[:, res_mask]
    q_mags_jax = jnp.array(q_mags_np[res_mask], dtype=sh_dtype)
    q_theo_sample_jax = jnp.array(q_theo_cryst / np.where(q_mags_np[res_mask] == 0, 1.0, q_mags_np[res_mask]), dtype=sh_dtype)
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

    I_weights_jax = jax.device_put(I_weights).astype(sh_dtype)

    w_l_j = jnp.asarray(build_band_weights(
       q_mags_np[res_mask], wl_min_tracking, wl_max_tracking, L_max,
       spectrum=assumed_spectrum,
       lorentz=lorentz_correction,
    ), dtype=sh_dtype)

    if U_init is not None:
        U_curr = jnp.array(U_init, dtype=sh_dtype)
    else:
        U_curr = jnp.eye(3, dtype=sh_dtype)

    P_spectral_full = jnp.eye(3, dtype=sh_dtype) * prior_ridge

    print(f"\n[2/3] Executing Field Tracker Pipeline (3 Local Subspace Rotational Degrees Active)...")

    tracking_history = _tracker_loop_reference(
        finder_file, event_batches, U_curr, P_spectral_full,
        q_theo_sample_jax, w_l_j, I_weights_jax,
        meas_noise_1st, meas_weight_2nd, ridge_inflation, L_max,
        process_q_scale_start, process_q_scale_end, annealing_rate,
        streaming_callback,
        gonio_axes, gonio_offsets,
        L_cov, cov_ema_weight, cov_threshold_frac, cov_warmup_events, sigmoid_k,
        q_scale_floor,
        cos_gate, gate_temp, low_l_damp,
        use_gate, damp_below_l,
        use_coverage_mask,
        sh_dtype=sh_dtype,
        measurement=measurement, q_mags_jax=q_mags_jax,
        wl_min_tracking=wl_min_tracking, wl_max_tracking=wl_max_tracking,
        tau_deg_start=tau_deg_start, tau_deg_end=tau_deg_end,
        tau_anneal_batches=tau_anneal_batches,
        sigma_deg=sigma_deg, max_step_scale=max_step_scale,
        p_floor_deg=p_floor_deg,
        d_min_start=(d_min_start if d_min_start is not None else d_min),
        d_min_end=d_min, refine_schedule=refine_schedule,
    )

    print(f"\n[3/3] Global Tracking complete. Saving continuous SO(3) state dataset.")
    with h5py.File(finder_file, "a") as f:
        if "tracking" in f: del f["tracking"]
        group = f.create_group("tracking")
        group.create_dataset("final_u_matrix", data=tracking_history[-1][1])
        group.create_dataset("timestamps", data=np.array([h[0] for h in tracking_history]))

    return tracking_history[-1][1]
