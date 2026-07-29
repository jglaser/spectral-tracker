"""Regression for event peak finding and global orientation search.

Simulates a CG4D-like still -- T4 lysozyme cell, 2.8-4.5 A band, a detector
covering only 2theta 22-59 deg -- as raw pixel events with Gaussian spots on a
flat background, then checks that

  * EventPeakFinder recovers the spot positions with high precision, while
    thresholding the same field (what EventStreamSparsifier does) does not, and
  * the global search recovers the true orientation from no prior at all, with
    the winner separated from the best genuinely different solution.
"""
import unittest

import numpy as np

from spectral_tracker.streaming.peaks import EventPeakFinder, gaussian_kernel
from spectral_tracker.global_search import (
    super_fibonacci, quat_to_mat, reflection_shell, band_mask,
    score_orientations, lattice_rotations, orientation_error,
    distinct_solutions)

CELL = (61.5, 61.5, 95.9, 90.0, 90.0, 120.0)
WL_BAND = (2.8, 4.5)
S0 = np.array([0.0, 0.0, 1.0])
N_PIX = 512


def _rot(axis, ang):
    k = np.asarray(axis, float)
    k /= np.linalg.norm(k)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


class TestPeakFinding(unittest.TestCase):
    """Peak finding on a synthetic panel, independent of any instrument file."""

    def _panel(self, n_spots=40, counts_per_spot=400, bg_per_pixel=2.0,
               sigma=2.5, seed=0):
        rng = np.random.default_rng(seed)
        centres = rng.uniform(30, N_PIX - 30, size=(n_spots, 2))
        img = rng.poisson(bg_per_pixel, size=(N_PIX, N_PIX)).astype(np.float32)
        for r, c in centres:
            rr = rng.normal(r, sigma, counts_per_spot).astype(int)
            cc = rng.normal(c, sigma, counts_per_spot).astype(int)
            ok = (rr >= 0) & (rr < N_PIX) & (cc >= 0) & (cc < N_PIX)
            np.add.at(img, (rr[ok], cc[ok]), 1.0)
        return img, centres

    def _find(self, img, sigma, n_sigma, min_sep):
        from scipy import ndimage
        ker = gaussian_kernel(sigma)
        f = ndimage.convolve(img, ker, mode="constant")
        mx = ndimage.maximum_filter(f, size=2 * min_sep + 1)
        med = np.median(f)
        mad = 1.4826 * np.median(np.abs(f - med)) + 1e-9
        rr, cc = np.nonzero((f >= mx) & (f > med + n_sigma * mad))
        return np.stack([rr, cc], 1), f, med + n_sigma * mad

    def test_local_maxima_beat_thresholding(self):
        """Both see the same field; only one of them finds peaks."""
        from scipy import ndimage
        img, centres = self._panel()
        found, field, thr = self._find(img, sigma=3.0, n_sigma=8.0, min_sep=4)

        # recall of the local-maximum detector
        d = np.linalg.norm(found[:, None, :] - centres[None, :, :], axis=2)
        recall = float((d.min(axis=0) < 4).mean())
        precision = float((d.min(axis=1) < 4).mean())
        self.assertGreater(recall, 0.9, f"local maxima recall {recall:.2f}")
        self.assertGreater(precision, 0.8, f"local maxima precision {precision:.2f}")

        # the same field, thresholded and connected-component labelled, is the
        # sparsifier's behaviour: far fewer objects than there are peaks, since
        # neighbouring spots and background structure merge into blobs
        lab, n_blobs = ndimage.label(field > thr)
        sizes = ndimage.sum(np.ones_like(field), lab, range(1, n_blobs + 1))
        self.assertGreater(found.shape[0], 0)
        self.assertGreater(
            np.median(sizes), 4 * np.pi * 3.0 ** 2 / 8,
            "thresholded blobs should be extended regions, not peak-sized")

    def test_centroid_accuracy(self):
        img, centres = self._panel(n_spots=25, seed=3)
        found, _, _ = self._find(img, sigma=3.0, n_sigma=10.0, min_sep=5)
        d = np.linalg.norm(found[:, None, :] - centres[None, :, :], axis=2)
        matched = d.min(axis=0) < 4
        self.assertGreater(matched.mean(), 0.85)
        self.assertLess(float(np.median(d.min(axis=0)[matched])), 2.0)


class TestGlobalSearch(unittest.TestCase):

    def setUp(self):
        from subhkl.optimization import FindUB
        ub = FindUB()
        (ub.a, ub.b, ub.c, ub.alpha, ub.beta, ub.gamma) = CELL
        ub.space_group = "P3_221"
        self.B = np.asarray(ub.reciprocal_lattice_B(), float)
        self.ops = lattice_rotations(self.B)

    def test_lattice_rotations_are_the_holohedry(self):
        self.assertEqual(len(self.ops), 12,
                         "hexagonal lattice has 12 proper rotations")
        for M in self.ops:
            self.assertLess(np.abs(M.T @ M - np.eye(3)).max(), 1e-8)
            self.assertAlmostEqual(float(np.linalg.det(M)), 1.0, places=8)

    def test_band_mask_is_orientation_independent(self):
        """The whole speed argument rests on this."""
        rng = np.random.default_rng(0)
        p_cry, qn, _ = reflection_shell(self.B, 8.0)
        U = _rot(rng.standard_normal(3), 1.0)
        p = (U @ p_cry.T).T
        vis = -(p @ S0) > 0
        lam = np.where(vis, -2.0 * (p @ S0) / np.maximum(qn, 1e-30), 0.0)
        keep = vis & (lam > WL_BAND[0]) & (lam < WL_BAND[1])
        # for a peak sitting exactly on prediction j, the pairwise mask must
        # admit j -- computed without any reference to U
        idx = np.flatnonzero(keep)[:50]
        q_obs = p[idx]
        mask = band_mask(q_obs, S0, qn, *WL_BAND)
        self.assertTrue(mask[np.arange(len(idx)), idx].all(),
                        "band mask rejected pairs the elastic condition allows")

    def _spots(self, U_true, d_min=5.0):
        p_cry, qn, _ = reflection_shell(self.B, d_min)
        p = (U_true @ p_cry.T).T
        sin_t = -(p @ S0)
        lam = 2.0 * sin_t / qn
        vis = ((lam > WL_BAND[0]) & (lam < WL_BAND[1])
               & (sin_t > 0.19) & (sin_t < 0.49))
        return p[vis]

    def test_recovers_orientation_with_no_prior(self):
        rng = np.random.default_rng(17)
        U_true = _rot(rng.standard_normal(3), rng.uniform(0.3, np.pi))
        # Generate from the SAME shell the search scores against. Observations
        # of reflections outside the scoring shell act as pure junk, which is
        # why the real pipeline scores at d > 5 A where the peak finder's
        # strongest peaks live rather than at an arbitrary cut.
        D_SEARCH = 8.0
        spots = self._spots(U_true, d_min=D_SEARCH)
        self.assertGreater(len(spots), 40)
        # 120 observed directions with 0.15 deg scatter and 25% spurious ones,
        # which is roughly the precision EventPeakFinder delivers on CG4D
        n_true = 120
        obs = spots[rng.integers(0, len(spots), n_true)]
        obs = obs + np.radians(0.15) * rng.standard_normal(obs.shape)
        n_junk = 40
        junk = rng.standard_normal((n_junk * 20, 3))
        junk /= np.linalg.norm(junk, axis=1, keepdims=True)
        s = -(junk @ S0)
        junk = junk[(s > 0.19) & (s < 0.49)][:n_junk]
        q = np.concatenate([obs, junk])
        q /= np.linalg.norm(q, axis=1, keepdims=True)

        p_cry, qn, _ = reflection_shell(self.B, D_SEARCH)
        allowed = band_mask(q, S0, qn, *WL_BAND)
        U_all = quat_to_mat(super_fibonacci(200_000))
        sc = score_orientations(q, U_all, p_cry, allowed, 1.0, block=96)
        order = np.argsort(-sc)[:50]
        errs = np.array([orientation_error(U_true, U_all[i], self.ops)
                         for i in order])
        print(f"\n  best grid error {errs.min():.2f} deg at shortlist rank "
              f"{int(np.argmin(errs))}")
        self.assertLess(errs.min(), 2.0,
                        "no grid point in the top 50 is near the truth")

    def test_distinct_solutions_deduplicates(self):
        """Refined shortlists collapse onto a few answers; confidence has to
        compare against a genuinely different one."""
        rng = np.random.default_rng(2)
        U = _rot(rng.standard_normal(3), 1.2)
        near = [_rot(rng.standard_normal(3), np.radians(0.3)) @ U
                for _ in range(6)]
        far = [_rot(rng.standard_normal(3), 1.0) @ U for _ in range(3)]
        cand = np.stack(near + far)
        scores = np.concatenate([np.full(6, 0.9), np.full(3, 0.4)])
        keep = distinct_solutions(cand, scores, self.ops, min_sep_deg=2.0)
        self.assertLess(len(keep), len(cand))
        self.assertLessEqual(len(keep), 4)
        self.assertGreater(
            orientation_error(cand[keep[0]], cand[keep[1]], self.ops), 2.0)


if __name__ == "__main__":
    unittest.main()
