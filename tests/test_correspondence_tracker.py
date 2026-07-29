"""Regression for the correspondence measurement at macromolecular scale.

The rest of tests/ uses a 10 A cubic cell, which puts ~150 reflections in band
and makes the degree <= L_max moments of the event directions strongly
anisotropic -- the regime the spectral measurement was designed for. A protein
cell is the opposite: T4 lysozyme (61.5, 61.5, 95.9, P3_221) excites ~10^4
reflections between 2.8 and 4.5 A, whose l <= 8 moments are isotropic to ~0.1%,
so the spectral innovation carries no orientation information and the tracker
walks away from a correct seed. This file pins the case down with a simulation
matched to CG4D_1808: same cell, same band, a detector that only covers
2theta 22-59 deg, 0.1 deg spots and 80% background.
"""
import os
import tempfile
import unittest

import h5py
import numpy as np

from spectral_tracker.tracker import tracker

CELL = (61.5, 61.5, 95.9, 90.0, 90.0, 120.0)
WL_BAND = (2.8, 4.5)
SIN_THETA_RANGE = (0.19, 0.49)          # CG4D_1808 detector acceptance
S0 = np.array([0.0, 0.0, 1.0])


def _rot(axis, ang):
    k = np.asarray(axis, float)
    k /= np.linalg.norm(k)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


class TestCorrespondenceTracker(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.finder_file = os.path.join(self.dir.name, "mock_finder.h5")
        from subhkl.optimization import FindUB
        ub = FindUB()
        (ub.a, ub.b, ub.c, ub.alpha, ub.beta, ub.gamma) = CELL
        ub.space_group = "P3_221"
        # Same B the tracker builds, so the test cannot pass on a convention
        # mismatch that would hide a real one.
        self.B = np.asarray(ub.reciprocal_lattice_B(), float)

    def tearDown(self):
        self.dir.cleanup()

    def _spots(self, U_true, d_min=3.0, d_max=40.0):
        """In-band reflections that the modelled detector can actually see."""
        A = np.linalg.inv(self.B).T
        bounds = np.ceil(np.linalg.norm(A, axis=0) / d_min).astype(int)
        H = np.stack(np.meshgrid(*[np.arange(-b, b + 1) for b in bounds],
                                 indexing="ij"), -1).reshape(-1, 3).astype(float)
        H = H[np.any(H != 0, axis=1)]
        q = (self.B @ H.T).T
        qn = np.linalg.norm(q, axis=1)
        H, q, qn = (H[(qn >= 1 / d_max) & (qn <= 1 / d_min)],
                    q[(qn >= 1 / d_max) & (qn <= 1 / d_min)],
                    qn[(qn >= 1 / d_max) & (qn <= 1 / d_min)])
        p = (U_true @ q.T).T / qn[:, None]
        sin_t = -(p @ S0)
        lam = 2.0 * sin_t / qn
        vis = ((lam > WL_BAND[0]) & (lam < WL_BAND[1])
               & (sin_t > SIN_THETA_RANGE[0]) & (sin_t < SIN_THETA_RANGE[1]))
        return p[vis]

    def _events(self, U_true, n_events=40000, bg_fraction=0.8,
                spot_sigma_deg=0.1, seed=0):
        rng = np.random.default_rng(seed)
        spots = self._spots(U_true)
        self.assertGreater(len(spots), 200, "simulation produced too few spots")
        n_bg = int(n_events * bg_fraction)
        n_sig = n_events - n_bg

        idx = rng.integers(0, len(spots), n_sig)
        sig = spots[idx] + np.radians(spot_sigma_deg) * rng.standard_normal((n_sig, 3))

        # Background fills the same detector cap, so the observed direction
        # distribution has the real coverage anisotropy in it.
        bg = []
        while sum(len(x) for x in bg) < n_bg:
            v = rng.standard_normal((n_bg, 3))
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            s = -(v @ S0)
            bg.append(v[(s > SIN_THETA_RANGE[0]) & (s < SIN_THETA_RANGE[1])])
        bg = np.concatenate(bg)[:n_bg]

        q = np.concatenate([sig, bg])
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        return q[rng.permutation(len(q))], len(spots)

    def _batches(self, q, batch, repeat):
        n = len(q)
        for r in range(repeat):
            for s in range(0, n, batch):
                e = min(s + batch, n)
                m = e - s
                yield (q[s:e].astype(np.float32),
                       (r + np.linspace(s, e, m, endpoint=False) / n).astype(np.float32),
                       np.zeros(m, np.int16), np.zeros(m, np.int16), np.zeros(m, np.int16),
                       np.zeros((m, 1), np.float32), np.zeros((m, 3), np.float32),
                       np.tile(S0, (m, 1)).astype(np.float32), e)

    def _write_finder(self, U_seed):
        with h5py.File(self.finder_file, "w") as f:
            (f["sample/a"], f["sample/b"], f["sample/c"]) = CELL[:3]
            (f["sample/alpha"], f["sample/beta"], f["sample/gamma"]) = CELL[3:]
            f["sample/space_group"] = b"P3_221"
            f["beam/ki_vec"] = S0
            f["sample/U"] = U_seed.astype(np.float32)
            f["instrument/wavelength"] = np.array(WL_BAND)

    def _err(self, U_true, U):
        t = np.clip((np.trace(np.asarray(U_true).T @ np.asarray(U)) - 1) / 2, -1, 1)
        return float(np.degrees(np.arccos(t)))

    def test_hkl_enumeration_is_complete(self):
        """h_max=None must cover the cell; the old cubic default did not."""
        A = np.linalg.inv(self.B).T
        bounds = np.ceil(np.linalg.norm(A, axis=0) / 2.0).astype(int)
        self.assertTrue((bounds >= np.array([30, 30, 47])).all(),
                        f"expected ~(31,31,48) for this cell at 2 A, got {bounds}")

    def test_converges_from_perturbed_seed(self):
        rng = np.random.default_rng(11)
        U_true = _rot(rng.standard_normal(3), rng.uniform(0, np.pi))
        q, n_spots = self._events(U_true)
        for perturb in (1.0, 2.0):
            U_seed = _rot(np.array([0.3, -0.5, 0.8]), np.radians(perturb)) @ U_true
            self._write_finder(U_seed)
            U_final = tracker(
                finder_file=self.finder_file,
                event_batches=self._batches(q, 20000, 120),
                measurement="correspondence", d_min=6.0, d_max=1000.0,
            )
            err = self._err(U_true, U_final)
            print(f"  {n_spots} visible spots | seed {perturb:.1f} deg -> "
                  f"{err:.3f} deg")
            self.assertLess(err, 0.5,
                            f"correspondence tracker did not converge from "
                            f"{perturb} deg: final {err:.3f} deg")

    def test_stable_on_a_correct_seed(self):
        rng = np.random.default_rng(5)
        U_true = _rot(rng.standard_normal(3), rng.uniform(0, np.pi))
        q, _ = self._events(U_true, seed=1)
        self._write_finder(U_true)
        U_final = tracker(
            finder_file=self.finder_file,
            event_batches=self._batches(q, 20000, 120),
            measurement="correspondence", d_min=6.0, d_max=1000.0,
        )
        err = self._err(U_true, U_final)
        print(f"  seed exact -> {err:.3f} deg")
        self.assertLess(err, 0.5,
                        f"tracker drifted off a correct seed by {err:.3f} deg")


if __name__ == "__main__":
    unittest.main()
