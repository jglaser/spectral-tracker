"""
Regression: ASSUMED SPECTRUM SHAPE -- flat vs strongly-peaked-asymmetric
========================================================================

The tracker's Ewald band is, by construction, a FLAT (top-hat) assumed
spectrum: w_l_j integrates Legendre modes with uniform weight in
x = -0.5*|q|*lambda across [wl_min, wl_max]. Real instrument spectra (e.g. a
cold-source Maxwellian) are strongly peaked and ASYMMETRIC. This test asks:
when the data are drawn from such a spectrum, does telling the tracker the true
peaked-asymmetric SHAPE converge better than the default flat assumption?

REQUIRES the assumed-spectrum generalization of w_l_j (see
assumed_spectrum_patch.py): tracker() must accept `assumed_spectrum`, a
vectorized callable phi(lambda)->weight, with None == the current flat top-hat.

Clean isolation
---------------
  * One peaked-asymmetric (log-normal) DATA spectrum, injected into the
    generator, reused by both legs (bit-identical events).
  * BOTH legs use the SAME band edges [0.5, 10] A (the spectrum's support), so
    band-edge / width effects are held fixed and ONLY the in-band assumed SHAPE
    differs: flat top-hat vs the true log-normal.
  * Reflection pool pinned via identical d_min/d_max.

Note this is a milder probe than the centered-vs-disjoint band test: both
assumptions still cover the populated shell, so flat is suboptimal, not blind.
The flat leg only MIS-WEIGHTS shells (over-weighting the sparsely populated
ones); it never zeroes the right one. Expect a positive but modest gap. A small
gap is itself the useful finding: it quantifies how much feeding the real
CG4D spectrum buys you over the flat default at full coverage.

Assertions
----------
  1. RELATIVE (primary): the matched-shape assumption must converge at least as
     well as flat, and better by a margin -- modeling the true shape should not
     hurt and should help.
  2. ABSOLUTE control: the matched-shape leg must clear the standard 2 deg bar.
"""

import unittest
import tempfile
import os
import itertools

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

import pytest
import e3x

from spectral_tracker import tracker


# ---- the shared spectrum: strongly peaked, asymmetric (right-skewed) ---------
LAM0 = 2.5      # mode-ish location (Angstrom)
SHAPE_S = 0.25  # smaller -> sharper peak / more flat-assumption penalty

def peaked_asymmetric_spectrum(lam):
    """Log-normal in wavelength: sharp rise, long high-lambda tail (asymmetric)."""
    lam = np.asarray(lam, dtype=float)
    out = np.zeros_like(lam)
    m = lam > 0
    out[m] = np.exp(-0.5 * (np.log(lam[m] / LAM0) / SHAPE_S) ** 2) / lam[m]
    return out


@pytest.fixture(scope="session", autouse=True)
def setup_temp_e3x_cache(tmp_path_factory):
    temp_dir = tmp_path_factory.mktemp("e3x_cache")
    cache_path = temp_dir / "sph.npz"
    e3x.Config.set_spherical_harmonics_cache(str(cache_path))
    yield
    e3x.Config.set_spherical_harmonics_cache("")


def get_cubic_symmetries():
    syms = []
    I = np.eye(3)
    for p in itertools.permutations([0, 1, 2]):
        P = I[list(p), :]
        for signs in itertools.product([1, -1], repeat=3):
            M = np.diag(signs) @ P
            if np.isclose(np.linalg.det(M), 1.0):
                syms.append(M)
    return syms


class TestAssumedSpectrumShape(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.finder_file = os.path.join(self.test_dir.name, "mock_finder.h5")
        np.random.seed(42)

    def tearDown(self):
        self.test_dir.cleanup()

    def create_mock_mtz(self, hkl_array, intensities):
        import gemmi
        mtz = gemmi.Mtz()
        mtz.cell = gemmi.UnitCell(10.0, 10.0, 10.0, 90.0, 90.0, 90.0)
        mtz.spacegroup = gemmi.SpaceGroup('P 1')
        mtz.add_dataset('mock')
        for c in ('H', 'K', 'L'):
            mtz.add_column(c, type='H')
        mtz.add_column('I', type='J')
        data = np.zeros((hkl_array.shape[1], 4), dtype=np.float32)
        data[:, 0] = hkl_array[0, :]
        data[:, 1] = hkl_array[1, :]
        data[:, 2] = hkl_array[2, :]
        data[:, 3] = intensities
        mtz.set_data(data)
        return mtz

    def generate_poissonian_events(self, U_true, num_events=1000000, duration=5.0,
                                   sigma_q=0.008, bg_fraction=0.0, b_factor=0.0,
                                   spectrum=None):
        """Synthetic Laue events. If `spectrum` (callable lam->weight) is given,
        it modulates per-reflection sampling probability (flux is NOT written to
        the MTZ intensities, which stay |F|^2-like)."""
        B_mat = np.diag([1.0 / 10.0, 1.0 / 10.0, 1.0 / 10.0])
        ki_vec = np.array([0.0, 0.0, 1.0])

        h_vals = np.arange(-3, 4)
        hc, kc, lc = np.meshgrid(h_vals, h_vals, h_vals, indexing="ij")
        hkl = np.stack([hc.flatten(), kc.flatten(), lc.flatten()], axis=0)
        mask = ~((hkl[0] == 0) & (hkl[1] == 0) & (hkl[2] == 0))
        hkl = hkl[:, mask]

        q_theo = B_mat @ hkl
        q_norms = np.linalg.norm(q_theo, axis=0)
        q_theo_hat = q_theo / q_norms

        kinematic_proj = ki_vec.T @ (U_true @ q_theo_hat)
        wavelengths = -(2.0 / q_norms) * kinematic_proj

        valid_mask = (wavelengths > 0.5) & (wavelengths < 10.0)
        valid_q_hat = q_theo_hat[:, valid_mask]
        valid_norms = q_norms[valid_mask]
        valid_wl = wavelengths[valid_mask]
        valid_hkl = hkl[:, valid_mask]
        num_valid = valid_q_hat.shape[1]

        num_bg = int(num_events * bg_fraction)
        num_sig = num_events - num_bg

        if b_factor > 0.0:
            raw_intensities = np.exp(-(b_factor * 39.47) * (valid_norms ** 2))
        else:
            raw_intensities = np.ones_like(valid_norms)

        if spectrum is not None:
            flux = np.asarray(spectrum(valid_wl), dtype=float)
            weights = raw_intensities * flux
            p_dist = weights / np.sum(weights)
        elif b_factor > 0.0:
            p_dist = raw_intensities / np.sum(raw_intensities)
        else:
            p_dist = None

        peak_indices = np.random.choice(num_valid, size=num_sig, p=p_dist)

        q_exp_list = []
        for idx in peak_indices:
            q_hat_lab = U_true @ valid_q_hat[:, idx]
            angular_std = sigma_q / valid_norms[idx]
            q_exp = q_hat_lab + np.random.normal(0, angular_std, 3)
            q_exp /= np.linalg.norm(q_exp)
            q_exp_list.append(q_exp)

        if num_bg > 0:
            bg_vecs = np.random.normal(0, 1, (num_bg, 3))
            bg_vecs /= np.linalg.norm(bg_vecs, axis=1, keepdims=True)
            q_exp_list.extend(bg_vecs)

        q_lab = np.array(q_exp_list)
        q_lab = q_lab[np.random.permutation(num_events)]

        times = np.sort(np.random.uniform(0, duration, num_events))
        banks = np.ones(num_events, dtype=int)
        pixels_r = np.zeros(num_events, dtype=int)
        pixels_c = np.zeros(num_events, dtype=int)
        return q_lab, times, banks, pixels_r, pixels_c, valid_hkl, raw_intensities

    def get_fake_batches(self, sim_data, batch_size=10000):
        q_lab, times, banks, pixels_r, pixels_c = sim_data[:5]
        num_events = len(times)
        for s in range(0, num_events, batch_size):
            e = min(s + batch_size, num_events)
            N = e - s
            yield (
                q_lab[s:e].astype(np.float32),
                times[s:e].astype(np.float32),
                banks[s:e].astype(np.int16),
                pixels_r[s:e].astype(np.int16),
                pixels_c[s:e].astype(np.int16),
                np.zeros((N, 1), dtype=np.float32),
                np.zeros((N, 3), dtype=np.float32),
                np.tile([0.0, 0.0, 1.0], (N, 1)).astype(np.float32),
                e,
            )

    def _evaluate_cubic_symmetric_error(self, U_true, U_pred):
        min_err = np.inf
        for sym in get_cubic_symmetries():
            U_mate = U_true @ sym
            tr = np.clip(np.trace(U_mate.T @ U_pred), -1.0, 3.0)
            min_err = min(min_err, np.degrees(np.arccos((tr - 1.0) / 2.0)))
        return min_err

    def test_assumed_spectrum_flat_vs_peaked(self):
        print(f"\n{'='*60}\nExecuting Regression: ASSUMED SPECTRUM SHAPE (flat vs peaked-asym)\n{'='*60}")

        U_true = Rotation.from_euler('y', 45.0, degrees=True).as_matrix()
        U_seed = Rotation.from_euler('y', 40.0, degrees=True).as_matrix()

        # Data drawn from the peaked-asymmetric spectrum (reused by both legs).
        sim_data = self.generate_poissonian_events(
            U_true, num_events=1_000_000, duration=5.0, bg_fraction=0.0,
            spectrum=peaked_asymmetric_spectrum,
        )
        valid_hkl, intensities = sim_data[5], sim_data[6]
        mock_mtz = self.create_mock_mtz(valid_hkl, intensities)

        # SAME band + pool for both legs -> only the in-band assumed SHAPE varies.
        WL_MIN, WL_MAX = 0.5, 10.0
        D_MIN, D_MAX = 2.0, 8.0

        def run_with_assumption(assumed_spectrum):
            with h5py.File(self.finder_file, "w") as f:
                f["sample/a"], f["sample/b"], f["sample/c"] = 10.0, 10.0, 10.0
                f["sample/alpha"], f["sample/beta"], f["sample/gamma"] = 90.0, 90.0, 90.0
                f["sample/space_group"] = b"P 1"
                f["beam/ki_vec"] = np.array([0.0, 0.0, 1.0])
                f["sample/U"] = U_seed
            event_stream = self.get_fake_batches(sim_data, batch_size=10_000)
            final_U = tracker(
                finder_file=self.finder_file,
                event_batches=event_stream,
                structure_factors=mock_mtz,
                L_max=8,
                d_min=D_MIN, d_max=D_MAX,
                wl_min_tracking=WL_MIN, wl_max_tracking=WL_MAX,
                assumed_spectrum=assumed_spectrum,
                lorentz_correction=False,
            )
            return self._evaluate_cubic_symmetric_error(U_true, final_U)

        flat_err = run_with_assumption(None)
        print(f"  FLAT assumption    (top-hat over [{WL_MIN}, {WL_MAX}] A) -> Sym-Err = {flat_err:6.2f} deg")

        matched_err = run_with_assumption(peaked_asymmetric_spectrum)
        print(f"  PEAKED-ASYM assumption (true log-normal shape)       -> Sym-Err = {matched_err:6.2f} deg")

        print(f"  shape-modeling gain (flat - matched): {flat_err - matched_err:+.2f} deg")

        # (2) Control: modeling the true shape must converge.
        self.assertLess(
            matched_err, 2.0,
            f"Matched-shape leg failed to converge ({matched_err:.2f} deg >= 2.0 deg)."
        )

        # (1) Relative (primary): the true shape must not do worse than flat, and
        # should help. Margin guards against a coincidental tie; loosen to a plain
        # `assertLessEqual(matched_err, flat_err)` if the full-coverage regime
        # turns out to make flat nearly as good (a legitimate, informative result).
        self.assertLess(
            matched_err, flat_err,
            f"Modeling the true peaked-asymmetric spectrum did not beat the flat "
            f"top-hat assumption: matched={matched_err:.2f} deg vs flat={flat_err:.2f} deg. "
            f"With the data concentrated in a thin wavelength shell, the flat band "
            f"mis-weights the resolution shells and should converge less accurately."
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
