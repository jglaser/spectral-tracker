"""
Route B, Step 1 -- unit tests for the parametric spectrum-learning estimator.

These validate the estimator in ISOLATION (no tracker / JAX / e3x): given a known
orientation, hidden incident spectrum, non-trivial |F|^2 priors and a
lambda-correlated geometry factor, it must recover the spectral SHAPE from
reflection populations restricted to singles. Pure numpy/scipy, fully runnable.

Covered:
  * singles_mask correctly isolates rays with no in-band collinear harmonic.
  * log-normal recovery (params + normalized shape).
  * Maxwellian recovery.
  * the |F|^2 / geometry division is load-bearing (skipping geometry biases the fit).
"""

import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from spectral_tracker import spectrum_learning as sl


def _normalized_shape_l2(phi_a, phi_b, lo, hi, n=400):
    x = np.linspace(lo, hi, n)
    a, b = phi_a(x), phi_b(x)
    a = a / np.trapezoid(a, x)
    b = b / np.trapezoid(b, x)
    return float(np.sqrt(np.trapezoid((a - b) ** 2, x) / np.trapezoid(a ** 2, x)))


class TestSinglesMask(unittest.TestCase):
    def test_isolates_harmonic_free_rays(self):
        # Columns: 3 collinear (1,0,0) harmonics; a lone (0,1,0); 2 collinear
        # (1,1,0) harmonics; a lone (1,2,3); and a (1,0,0)-ray member that is the
        # ONLY in-band one of its ray (others marked out-of-band -> becomes single).
        hkl = np.array([
            [1, 2, 3, 0, 1, 2, 1, 5],   # h
            [0, 0, 0, 1, 1, 2, 2, 0],   # k
            [0, 0, 0, 0, 0, 0, 3, 0],   # l
        ])
        #        idx:  0  1  2  3  4  5  6  7
        # rays:  (100)(100)(100)(010)(110)(110)(123)(500)->(100)
        in_band = np.array([True, True, False, True, True, True, True, True])
        # ray (1,0,0) in-band members among idx {0,1,2,7}: idx0,idx1,idx7 -> occ 3 -> none single
        # ray (0,1,0): idx3 -> single
        # ray (1,1,0): idx4,idx5 -> occ 2 -> none single
        # ray (1,2,3): idx6 -> single
        mask = sl.singles_mask(hkl, in_band)
        expected = np.array([False, False, False, True, False, False, True, False])
        np.testing.assert_array_equal(mask, expected)

    def test_partial_outofband_creates_single(self):
        # (1,0,0),(2,0,0),(3,0,0) collinear; only (1,0,0) in band -> it is a single.
        hkl = np.array([[1, 2, 3], [0, 0, 0], [0, 0, 0]])
        in_band = np.array([True, False, False])
        np.testing.assert_array_equal(sl.singles_mask(hkl, in_band),
                                      np.array([True, False, False]))


class TestSpectrumRecovery(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.B = sl.reciprocal_B(10.0, 10.0, 10.0)
        hv = np.arange(-6, 7)
        H, K, L = np.meshgrid(hv, hv, hv, indexing="ij")
        hkl = np.stack([H.ravel(), K.ravel(), L.ravel()])
        self.hkl = hkl[:, ~((hkl[0] == 0) & (hkl[1] == 0) & (hkl[2] == 0))]
        self.q = self.B @ self.hkl
        self.qn = np.linalg.norm(self.q, axis=0)
        self.ki = np.array([0.0, 0.0, 1.0])
        self.U = Rotation.from_euler("xyz", [12, 45, -8], degrees=True).as_matrix()
        self.lam, _ = sl.bragg_wavelengths(self.q, self.U, self.ki)
        self.WL = (0.8, 6.0)
        self.in_band = (np.isfinite(self.lam) & (self.lam > self.WL[0]) & (self.lam < self.WL[1])
                        & (self.qn > 1 / 9.0) & (self.qn < 1 / 1.2))
        self.singles = sl.singles_mask(self.hkl, self.in_band)
        self.I = np.exp(self.rng.normal(0, 0.6, self.hkl.shape[1]))   # |F|^2 ~1 decade spread

    def _populations(self, phi_true, geom):
        p = np.zeros(self.hkl.shape[1])
        p[self.in_band] = (self.I * phi_true(self.lam) * geom)[self.in_band]
        p = p / p.sum()
        return self.rng.multinomial(1_000_000, p).astype(float)

    def test_recover_lognormal(self):
        LAM0, S = 2.3, 0.30

        def phi_true(x):
            x = np.asarray(x, float); o = np.zeros_like(x); m = x > 0
            o[m] = np.exp(-0.5 * (np.log(x[m] / LAM0) / S) ** 2) / x[m]; return o

        geom = np.where(self.in_band, self.lam, 1.0) ** 2   # lambda-correlated (adversarial)
        counts = self._populations(phi_true, geom)
        params, phi_fit, info = sl.learn_spectrum(
            self.lam, counts, self.I, geom=geom, singles=self.singles,
            family="lognormal", lam_band=self.WL, min_count=5)

        self.assertGreater(info["n_singles_used"], 50)
        self.assertLess(abs(params["lam0"] - LAM0) / LAM0, 0.03,
                        f"lam0 off: {params['lam0']:.3f} vs {LAM0}")
        self.assertLess(abs(params["s"] - S) / S, 0.08,
                        f"s off: {params['s']:.3f} vs {S}")
        self.assertLess(_normalized_shape_l2(phi_fit, phi_true, *self.WL), 0.02)

    def test_recover_maxwellian(self):
        LAM_T = 2.0

        def phi_true(x):
            x = np.asarray(x, float); o = np.zeros_like(x); m = x > 0
            o[m] = x[m] ** -5 * np.exp(-((LAM_T / x[m]) ** 2)); return o

        counts = self._populations(phi_true, np.ones(self.hkl.shape[1]))
        params, phi_fit, info = sl.learn_spectrum(
            self.lam, counts, self.I, geom=None, singles=self.singles,
            family="maxwellian", lam_band=self.WL, min_count=5)

        self.assertLess(abs(params["lam_T"] - LAM_T) / LAM_T, 0.03,
                        f"lam_T off: {params['lam_T']:.3f} vs {LAM_T}")
        self.assertLess(_normalized_shape_l2(phi_fit, phi_true, *self.WL), 0.02)

    def test_geometry_division_is_load_bearing(self):
        # A lambda-correlated geometry, if not divided out, must bias the recovered
        # spectrum -- confirming the |F|^2/geometry bookkeeping matters.
        LAM0, S = 2.3, 0.30

        def phi_true(x):
            x = np.asarray(x, float); o = np.zeros_like(x); m = x > 0
            o[m] = np.exp(-0.5 * (np.log(x[m] / LAM0) / S) ** 2) / x[m]; return o

        geom = np.where(self.in_band, self.lam, 1.0) ** 2
        counts = self._populations(phi_true, geom)

        with_geom, _, _ = sl.learn_spectrum(
            self.lam, counts, self.I, geom=geom, singles=self.singles,
            family="lognormal", lam_band=self.WL, min_count=5)
        no_geom, _, _ = sl.learn_spectrum(
            self.lam, counts, self.I, geom=None, singles=self.singles,
            family="lognormal", lam_band=self.WL, min_count=5)

        self.assertLess(abs(with_geom["lam0"] - LAM0) / LAM0, 0.03)         # corrected: accurate
        self.assertGreater(abs(no_geom["lam0"] - LAM0) / LAM0, 0.10)        # uncorrected: biased


if __name__ == "__main__":
    unittest.main(verbosity=2)
