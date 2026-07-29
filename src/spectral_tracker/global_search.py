"""Global orientation search from a Laue peak list.

The correspondence tracker has a 2-3 deg capture range, so reaching it from an
unknown orientation needs a search that ranks the truth into the top handful of
a grid over SO(3), followed by refinement. This module is the search half; the
refinement reuses the tracker's own normal equations.

THE TRICK THAT MAKES SCORING CHEAP

Scoring a trial orientation appears to require predicting q = U B h for every
reflection and gating to the wavelength band. The gate looks orientation
dependent -- it is a condition on the predicted direction -- but it is not:

    lambda_ij = -2 (q_hat_i . s0_hat) / |q_j|

`q_hat_i . s0_hat` is a frame-invariant observable of peak i and `|q_j|` is a
property of reflection j, so which (peak, reflection) pairs are wavelength
allowed is fixed before the search starts. Every trial reuses the same mask,
and all that is left per orientation is rotating the observations into the
crystal frame and one matmul against a constant matrix of reflection
directions. Measured: ~13,500 orientations/s, a 200k grid in 14 s.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

# Alexa, "Super-Fibonacci Spirals", CVPR 2022.
_PHI = np.sqrt(2.0)
_PSI = 1.533751168755204288118041


def super_fibonacci(n: int) -> np.ndarray:
    """Near-uniform SO(3) sampling as unit quaternions, shape (n, 4).

    Low discrepancy with no lattice artefacts. An Euler-angle grid clumps at
    the poles, and clumping is what inflates the number of trials needed for a
    given capture radius.
    """
    s = np.arange(n) + 0.5
    t = s / n
    d = 2.0 * np.pi * s
    r, R = np.sqrt(t), np.sqrt(1.0 - t)
    a, b = d / _PHI, d / _PSI
    return np.stack([r * np.sin(a), r * np.cos(a),
                     R * np.sin(b), R * np.cos(b)], axis=1)


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], axis=-1).reshape(q.shape[:-1] + (3, 3))


def reflection_shell(B, d_min, d_max=1e4):
    """Unit reflection directions in the crystal frame, and their |q|.

    Enumeration bounds are per axis (|h_i| <= |a_i| / d_min); a cubic box is
    only complete when every cell edge is shorter than h_max * d_min.
    """
    B = np.asarray(B, float)
    A = np.linalg.inv(B).T
    bounds = np.ceil(np.linalg.norm(A, axis=0) / d_min).astype(int)
    H = np.stack(np.meshgrid(*[np.arange(-b, b + 1) for b in bounds],
                             indexing="ij"), -1).reshape(-1, 3).astype(float)
    H = H[np.any(H != 0, axis=1)]
    qn = np.linalg.norm(H @ B.T, axis=1)
    m = (qn >= 1.0 / d_max) & (qn <= 1.0 / d_min)
    H, qn = H[m], qn[m]
    return (B @ H.T / qn).T, qn, H


def band_mask(q_obs, s0, q_mags, wl_min, wl_max):
    """(N_peaks, M_refl) mask of wavelength-allowed pairs. Orientation free."""
    sin_t = -(np.asarray(q_obs) @ np.asarray(s0))     # = |q| lambda / 2
    lo = 2.0 * sin_t / wl_max
    hi = 2.0 * sin_t / wl_min
    ok = ((q_mags[None, :] >= lo[:, None]) & (q_mags[None, :] <= hi[:, None]))
    return ok & (sin_t > 0)[:, None]


@partial(jax.jit, static_argnames=[])
def _score_block(q_obs, U_block, p_cry, allowed, inv2t2):
    hp = jax.lax.Precision.HIGHEST
    # Rotating the OBSERVATIONS into the crystal frame keeps p_cry constant
    # across the whole grid, so the inner matmul reuses one operand.
    x = jnp.einsum("bij,ni->bnj", U_block, q_obs, precision=hp)
    c = jnp.einsum("bnj,mj->bnm", x, p_cry, precision=hp)
    # HIGHEST is required, not cosmetic: jax's default matmul precision is
    # bfloat16 on recent GPUs, giving max |d cos| = 6.7e-4 between unit
    # vectors, against 1 - cos(1 deg) = 1.5e-4. At the default the score
    # exceeds its own maximum of 1 and random orientations outscore the truth.
    c = jnp.clip(c, -1.0, 1.0)
    c = jnp.where(allowed[None, :, :], c, -1.0)
    cmax = jnp.max(c, axis=2)
    return jnp.mean(jnp.exp((cmax - 1.0) * inv2t2), axis=1)


def score_orientations(q_obs, U_all, p_cry, allowed, tau_deg, block=64):
    """Mean per-peak best match, one value per trial orientation."""
    inv2t2 = np.float32(1.0 / (0.5 * np.radians(tau_deg) ** 2))
    q_j = jnp.asarray(np.asarray(q_obs, np.float32))
    p_j = jnp.asarray(np.asarray(p_cry, np.float32))
    a_j = jnp.asarray(allowed)
    out = np.empty(len(U_all), np.float32)
    for i in range(0, len(U_all), block):
        Ub = jnp.asarray(np.asarray(U_all[i:i + block], np.float32))
        out[i:i + block] = np.asarray(_score_block(q_j, Ub, p_j, a_j, inv2t2))
    return out


@partial(jax.jit, static_argnames=[])
def _refine_step(q_obs, U_block, p_cry, allowed, res_mask, inv2t2, max_step):
    hp = jax.lax.Precision.HIGHEST
    p = jnp.einsum("bij,mj->bmi", U_block, p_cry, precision=hp)
    c = jnp.clip(jnp.einsum("ni,bmi->bnm", q_obs, p, precision=hp), -1.0, 1.0)
    ok = allowed[None, :, :] & res_mask[None, None, :]
    c = jnp.where(ok, c, -1.0)
    cmax = jnp.max(c, axis=2)
    w = jnp.exp((c - cmax[:, :, None]) * inv2t2) * ok
    acc = jnp.exp((cmax - 1.0) * inv2t2)
    w = w / (jnp.sum(w, axis=2, keepdims=True) + 1e-30) * acc[:, :, None]

    W = jnp.sum(w, axis=1)
    v = jnp.einsum("bnm,ni->bmi", w, q_obs, precision=hp)
    A = (jnp.eye(3)[None] * jnp.sum(W, axis=1)[:, None, None]
         - jnp.einsum("bm,bmi,bmj->bij", W, p, p, precision=hp))
    b = jnp.sum(jnp.cross(p, v), axis=1)
    ridge = (jnp.eye(3)[None] * 1e-9
             * jnp.maximum(jnp.trace(A, axis1=1, axis2=2), 1.0)[:, None, None])
    om = jnp.linalg.solve(A + ridge, b[..., None]).squeeze(-1)
    nrm = jnp.linalg.norm(om, axis=1, keepdims=True)
    om = jnp.where(nrm > max_step, om * (max_step / (nrm + 1e-30)), om)
    om = jnp.where(jnp.isfinite(om), om, 0.0)

    th = jnp.linalg.norm(om, axis=1, keepdims=True)
    k = om / (th + 1e-30)
    K = (jnp.zeros((len(om), 3, 3))
         .at[:, 0, 1].set(-k[:, 2]).at[:, 0, 2].set(k[:, 1])
         .at[:, 1, 0].set(k[:, 2]).at[:, 1, 2].set(-k[:, 0])
         .at[:, 2, 0].set(-k[:, 1]).at[:, 2, 1].set(k[:, 0]))
    R = (jnp.eye(3)[None] + jnp.sin(th)[:, :, None] * K
         + (1 - jnp.cos(th))[:, :, None] * jnp.matmul(K, K))
    return jnp.matmul(R, U_block)


def refine_candidates(q_obs, U_cand, p_cry, q_mags, allowed, schedule,
                      block=32):
    """Run the tracker's staged correspondence schedule on a whole shortlist.

    Refinement is a discriminator as much as a polish step: a trial near the
    truth tightens onto the peaks and its score rises, a wrong one has nothing
    to tighten onto. That only holds when the peak list is precise -- with the
    sparsifier's thresholded blobs, wrong candidates refine to HIGHER scores
    than the truth, which is what motivates EventPeakFinder.
    """
    q_j = jnp.asarray(np.asarray(q_obs, np.float32))
    p_j = jnp.asarray(np.asarray(p_cry, np.float32))
    a_j = jnp.asarray(allowed)
    out = np.empty_like(np.asarray(U_cand))
    for i in range(0, len(U_cand), block):
        Ub = jnp.asarray(np.asarray(U_cand[i:i + block], np.float32))
        for d_min, tau_deg, n_iter in schedule:
            res = jnp.asarray(q_mags <= 1.0 / d_min)
            inv2t2 = np.float32(1.0 / (0.5 * np.radians(tau_deg) ** 2))
            step = np.float32(np.radians(0.5 * tau_deg))
            for _ in range(n_iter):
                Ub = _refine_step(q_j, Ub, p_j, a_j, res, inv2t2, step)
        out[i:i + block] = np.asarray(Ub)
    return out


def lattice_rotations(B):
    """Cartesian rotations B R B^-1 for integer R preserving the lattice.

    The holohedry -- 12 proper rotations for a hexagonal cell. U is only ever
    determined up to one of these, so any comparison of orientations has to be
    reduced over them.
    """
    import itertools
    B = np.asarray(B, float)
    Binv = np.linalg.inv(B)
    out = []
    for vals in itertools.product((-1, 0, 1), repeat=9):
        R = np.array(vals, float).reshape(3, 3)
        if abs(np.linalg.det(R) - 1) > 1e-9:
            continue
        M = B @ R @ Binv
        if np.abs(M.T @ M - np.eye(3)).max() < 1e-8:
            out.append(M)
    return np.array(out)


def orientation_error(U_a, U_b, ops=None):
    """Geodesic angle in degrees, minimised over the reindexing group."""
    def ang(X, Y):
        t = np.clip((np.trace(np.asarray(X).T @ np.asarray(Y)) - 1) / 2, -1, 1)
        return float(np.degrees(np.arccos(t)))
    if ops is None:
        return ang(U_a, U_b)
    return min(ang(np.asarray(U_a) @ M, U_b) for M in ops)


def search(q_obs, ki_obs, B, wl_band, d_min=5.0, n_grid=200_000, tau_deg=1.0,
           refine_top=200, rescore_tau=0.5, schedule=None, block=64):
    """Grid search, refine the shortlist, rank by rescore.

    Returns a dict with the winning orientation, its score, the runner-up
    score, and the separation ratio between them. The ratio is the confidence
    readout: below ~1.1 the answer is not distinguishable from the null tail.
    """
    from spectral_tracker.tracker import DEFAULT_REFINE_SCHEDULE
    schedule = schedule or DEFAULT_REFINE_SCHEDULE

    q_obs = np.asarray(q_obs, float)
    s0 = np.asarray(ki_obs, float).mean(0)
    s0 /= np.linalg.norm(s0)
    p_cry, q_mags, hkl = reflection_shell(B, d_min)
    allowed = band_mask(q_obs, s0, q_mags, wl_band[0], wl_band[1])

    U_all = quat_to_mat(super_fibonacci(n_grid))
    sc = score_orientations(q_obs, U_all, p_cry, allowed, tau_deg, block)
    order = np.argsort(-sc)[:refine_top]
    U_ref = refine_candidates(q_obs, U_all[order], p_cry, q_mags, allowed,
                              schedule)
    sc2 = score_orientations(q_obs, U_ref, p_cry, allowed, rescore_tau, block)
    rank = np.argsort(-sc2)
    best, second = rank[0], rank[1]
    return dict(U=U_ref[best], score=float(sc2[best]),
                runner_up=float(sc2[second]),
                separation=float(sc2[best] / max(sc2[second], 1e-12)),
                candidates=U_ref, candidate_scores=sc2,
                grid_scores=sc, grid_order=order, n_reflections=len(p_cry))
