"""
Route B, Step 1: learn the incident spectrum phi(lambda) from reflection
populations (standalone, pure-numpy; no tracker / JAX / e3x dependency).

Idea
----
Events are directionless in wavelength, but a (rough) orientation U assigns each
PREDICTED reflection a single geometric Bragg wavelength

        lambda_j(U) = -2 * (k_i_hat . (U @ q_hat_j)) / |q_j|,   |q_j| = 1/d_j.

So U turns a directionless stream into (reflection, lambda) tags. The observed
population on reflection j factorizes as

        count_j  ~  I_j * phi(lambda_j) * geom_j,

with I_j the |F|^2 prior (from the MTZ) and geom_j the Ewald/Lorentz/efficiency
geometry. Dividing those out leaves a noisy estimate of the incident spectrum
sampled at the reflections' wavelengths:

        phi_hat_j = count_j / (I_j * geom_j).

We then fit a 1-2 parameter family to {(lambda_j, phi_hat_j)}, weighted by counts
and restricted to SINGLES (rays with no collinear harmonic partner, whose count
is unambiguous). Only the SHAPE is identifiable; the amplitude is a free nuisance.

Both supported families linearize after the right transform, so the shape fit is
a closed-form weighted least squares (no iterative solver, deterministic):

  log-normal   phi = A * exp(-(ln(lam/lam0))^2 / (2 s^2)) / lam
               ln(phi*lam) = a + b*u + c*u^2,  u=ln(lam)
               -> s = sqrt(-1/(2c)),  lam0 = exp(-b/(2c))

  maxwellian   phi = A * lam^-5 * exp(-(lam_T/lam)^2)
               ln(phi*lam^5) = ln A - lam_T^2 * (1/lam^2)
               -> lam_T = sqrt(-slope) of a line in z=1/lam^2
"""

import math
import numpy as np


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------
def reciprocal_B(a, b, c, alpha=90.0, beta=90.0, gamma=90.0):
    """Busing-Levy reciprocal-lattice B (columns = reciprocal basis), 1/Angstrom."""
    al, be, ga = np.radians([alpha, beta, gamma])
    ca, cb, cg = np.cos([al, be, ga])
    sa, sb, sg = np.sin([al, be, ga])
    V = a * b * c * math.sqrt(max(1 - ca**2 - cb**2 - cg**2 + 2 * ca * cb * cg, 1e-12))
    # Reciprocal lengths and reciprocal-angle cosines.
    ar, br, cr = b * c * sa / V, a * c * sb / V, a * b * sg / V
    cas = (cb * cg - ca) / (sb * sg)
    cbs = (ca * cg - cb) / (sa * sg)
    cgs = (ca * cb - cg) / (sa * sb)
    sgs = math.sqrt(max(1 - cgs**2, 1e-12))
    sbs = math.sqrt(max(1 - cbs**2, 1e-12))
    # Canonical Busing-Levy B (reduces to diag(1/a,1/b,1/c) for orthorhombic cells).
    B = np.array([
        [ar, br * cgs, cr * cbs],
        [0.0, br * sgs, -cr * sbs * ca],
        [0.0, 0.0, 1.0 / c],
    ])
    return B


def bragg_wavelengths(q_theo_cryst, U, ki_hat):
    """lambda_j = -2 (ki_hat . (U @ q_hat_j)) / |q_j|, with q_hat the unit direction."""
    q = np.asarray(q_theo_cryst, dtype=float)             # (3, M) reciprocal vectors
    q_norms = np.linalg.norm(q, axis=0)                   # (M,) == 1/d
    q_hat = q / np.where(q_norms == 0, 1.0, q_norms)
    proj = np.asarray(ki_hat, float) @ (np.asarray(U, float) @ q_hat)   # (M,)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = -2.0 * proj / np.where(q_norms == 0, np.nan, q_norms)
    return lam, q_norms


def _reduce_hkl(hkl):
    """Primitive ray index: divide each (h,k,l) by gcd(|h|,|k|,|l|), keeping sign."""
    hkl = np.asarray(hkl, dtype=int)
    out = np.zeros_like(hkl)
    for j in range(hkl.shape[1]):
        h, k, l = hkl[:, j]
        g = math.gcd(math.gcd(abs(int(h)), abs(int(k))), abs(int(l)))
        out[:, j] = hkl[:, j] // g if g > 0 else hkl[:, j]
    return out


def singles_mask(hkl, in_band):
    """
    True for reflections that are the ONLY in-band member of their ray (no
    collinear harmonic partner). Counts on multi-occupancy rays blend several
    wavelengths and cannot be apportioned, so they are excluded.
    """
    hkl = np.asarray(hkl, dtype=int)
    in_band = np.asarray(in_band, dtype=bool)
    reduced = _reduce_hkl(hkl)
    from collections import defaultdict
    occ = defaultdict(int)
    for j in range(hkl.shape[1]):
        if in_band[j]:
            occ[tuple(reduced[:, j])] += 1
    mask = np.zeros(hkl.shape[1], dtype=bool)
    for j in range(hkl.shape[1]):
        if in_band[j] and occ[tuple(reduced[:, j])] == 1:
            mask[j] = True
    return mask


# ----------------------------------------------------------------------------
# Parametric families (closed-form shape fits)
# ----------------------------------------------------------------------------
def _wls(X, y, w):
    """Weighted least squares: returns coefficients beta minimizing sum w (y - X beta)^2."""
    W = w[:, None]
    XtWX = X.T @ (W * X)
    XtWy = X.T @ (w * y)
    return np.linalg.solve(XtWX + 1e-12 * np.eye(X.shape[1]), XtWy)


def fit_lognormal(lam, phi_hat, weights):
    u = np.log(lam)
    y = np.log(phi_hat * lam)
    X = np.stack([np.ones_like(u), u, u**2], axis=1)
    a, b, c = _wls(X, y, weights)
    if c >= 0:
        raise ValueError("log-normal fit failed: non-concave (c >= 0); too little signal?")
    s = math.sqrt(-1.0 / (2.0 * c))
    lam0 = math.exp(-b / (2.0 * c))
    params = {"lam0": lam0, "s": s}

    def phi(x):
        x = np.asarray(x, float)
        out = np.zeros_like(x)
        m = x > 0
        out[m] = np.exp(-0.5 * (np.log(x[m] / lam0) / s) ** 2) / x[m]
        return out

    return params, phi


def fit_maxwellian(lam, phi_hat, weights):
    z = 1.0 / lam**2
    y = np.log(phi_hat) + 5.0 * np.log(lam)
    X = np.stack([np.ones_like(z), z], axis=1)
    lnA, slope = _wls(X, y, weights)
    if slope >= 0:
        raise ValueError("maxwellian fit failed: non-negative slope; wrong family or too little signal?")
    lam_T = math.sqrt(-slope)
    params = {"lam_T": lam_T}

    def phi(x):
        x = np.asarray(x, float)
        out = np.zeros_like(x)
        m = x > 0
        out[m] = x[m] ** (-5.0) * np.exp(-((lam_T / x[m]) ** 2))
        return out

    return params, phi


_FAMILIES = {"lognormal": fit_lognormal, "maxwellian": fit_maxwellian}


# ----------------------------------------------------------------------------
# Top-level estimator
# ----------------------------------------------------------------------------
def learn_spectrum(
    lambdas, counts, intensities, geom=None, singles=None,
    family="lognormal", lam_band=None, min_count=5.0,
):
    """
    Fit incident-spectrum SHAPE from reflection populations.

    Parameters
    ----------
    lambdas, counts, intensities : (M,) per-reflection arrays.
        lambdas    : geometric Bragg wavelength lambda_j(U).
        counts     : observed (gated) population on reflection j.
        intensities: |F|^2 prior I_j (from the MTZ); used to divide out structure.
    geom    : (M,) Ewald/Lorentz/efficiency factor to divide out (default ones).
    singles : (M,) bool. If None, all finite reflections are eligible -- but you
              should pass singles_mask(hkl, in_band) for real data.
    family  : 'lognormal' or 'maxwellian'.
    lam_band: (lo, hi) wavelength window to fit within (avoids the empty tails).
    min_count: drop reflections below this count (log-space noise control).

    Returns
    -------
    params : dict of fitted shape parameters.
    phi    : callable lambda->weight (unnormalized; only shape is meaningful).
    info   : dict with the points used and diagnostics.
    """
    lam = np.asarray(lambdas, float)
    cnt = np.asarray(counts, float)
    I = np.asarray(intensities, float)
    geom = np.ones_like(lam) if geom is None else np.asarray(geom, float)
    elig = np.ones_like(lam, bool) if singles is None else np.asarray(singles, bool)

    good = (
        elig & np.isfinite(lam) & (lam > 0) & (cnt >= min_count)
        & (I > 0) & (geom > 0)
    )
    if lam_band is not None:
        good &= (lam >= lam_band[0]) & (lam <= lam_band[1])
    if good.sum() < 5:
        raise ValueError(f"only {int(good.sum())} usable singles; need >= 5 to fit a spectrum")

    lam_g = lam[good]
    phi_hat = cnt[good] / (I[good] * geom[good])      # divide out |F|^2 and geometry
    weights = cnt[good]                               # ~ inverse log-space variance

    params, phi = _FAMILIES[family](lam_g, phi_hat, weights)
    info = {
        "n_singles_used": int(good.sum()),
        "lam_used": lam_g,
        "phi_hat_used": phi_hat,
        "weights": weights,
        "family": family,
    }
    return params, phi, info
