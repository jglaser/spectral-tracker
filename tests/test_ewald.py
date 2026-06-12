"""
Regression: EWALD WINDOW WAVELENGTH MISMATCH (spectrum-aware, width-controlled)
===============================================================================

Probes the effect of the prediction-side wavelength band
(wl_min_tracking / wl_max_tracking -> Legendre band-weights w_l_j) on
convergence, in the presence of a realistic incident spectrum.

Why the naive version was invalid
---------------------------------
The loader normalizes q = k_f_hat - k_i_hat, so the data carry no resolution
coordinate: wavelength re-enters ONLY through w_l_j (which predicted reflections
the band lights up and how it weights them). A first attempt compared a band
matched to the wavelength SUPPORT (0.5-10) against a subset (5-12) and the
"mismatched" band WON. Two confounds were tangled together:

  (1) support != density. With a spectral curve the events concentrate in a
      thin wavelength shell; a band covering the full support is matched to
      where events COULD be, not where they ARE.

  (2) band WIDTH sets prediction sharpness (Fisher information) independent of
      centering. A wide band spreads w_l_j across many |q|-shells -> low-
      contrast predicted moments -> weak gradient. A narrower band concentrates
      weight -> sharper moments -> more information (seen directly as a ~4x
      larger Coherent-Mass = trace(P^-1)). Width alone can beat centering.

This test removes both confounds:
  * The incident spectrum is injected explicitly (wl_center / wl_sigma), so the
    TRUE optimal band is known and the test owns it.
  * The two legs are EQUAL WIDTH. Only the band CENTER (overlap with the
    populated wavelength shell) differs, so centering -- not the width/sharpness
    nuisance -- decides the outcome. A shifted lambda-window also selects a
    different |q|-shell via x = -0.5*|q|*lambda inside the pinned d_min/d_max
    pool, so the off-center band lights genuinely different DIRECTIONS.

Design
------
  spectrum       : narrow Gaussian at SPECTRUM_CENTER (thin |q|-shell of data).
  matched band   : [center - HALF_W, center + HALF_W]  (sits on the spectrum).
  mismatched band: equal width, shifted onto an essentially unpopulated shell.
  reflection pool: pinned via identical d_min/d_max so only w_l_j varies.

Assertions
----------
  1. RELATIVE (primary, robust to absolute re-tuning): the on-spectrum band
     must converge better than the equal-width off-spectrum band.
  2. ABSOLUTE control: the on-spectrum band must clear the standard 2 deg bar.

NOTE: flux is deliberately NOT folded into the mock MTZ intensities. In reality
the intensity prior is |F|^2 (crystallography) while the spectral weighting is
exactly what w_l_j is meant to model -- so the spectrum lives only in the data
density, and the band is what must capture it.
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


@pytest.fixture(scope="session", autouse=True)
def setup_temp_e3x_cache(tmp_path_factory):
    """Session-scoped temporary e3x spherical-harmonics cache (mirrors the main suite)."""
    temp_dir = tmp_path_factory.mktemp("e3x_cache")
    cache_path = temp_dir / "sph.npz"
    e3x.Config.set_spherical_harmonics_cache(str(cache_path))
    yield
    e3x.Config.set_spherical_harmonics_cache("")


def get_cubic_symmetries():
    """The 24 proper rotations of the cubic point group."""
    syms = []
    I = np.eye(3)
    for p in itertools.permutations([0, 1, 2]):
        P = I[list(p), :]
        for signs in itertools.product([1, -1], repeat=3):
            S = np.diag(signs)
            M = S @ P
            if np.isclose(np.linalg.det(M), 1.0):
                syms.append(M)
    return syms


class TestEwaldWavelengthWindow(unittest.TestCase):
    # ---- harness (self-contained; does not re-collect the parent suite) ------

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
        mtz.add_column('H', type='H')
        mtz.add_column('K', type='H')
        mtz.add_column('L', type='H')
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
                                   wl_center=None, wl_sigma=None):
        """
        Synthetic Laue events. True bandpass SUPPORT is fixed at 0.5-10 A
        (valid_mask). If wl_center/wl_sigma are given, a Gaussian incident
        spectrum modulates the per-reflection sampling probability, so the
        populated reflections concentrate in a thin wavelength shell -- the
        density the tracking band must match. Flux is NOT written into the
        returned intensities (those stay |F|^2-like).
        """
        B_mat = np.array([[1.0 / 10.0, 0, 0],
                          [0, 1.0 / 10.0, 0],
                          [0, 0, 1.0 / 10.0]])
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

        # |F|^2-like prior (returned to the MTZ); flux is kept separate.
        if b_factor > 0.0:
            raw_intensities = np.exp(-(b_factor * 39.47) * (valid_norms ** 2))
        else:
            raw_intensities = np.ones_like(valid_norms)

        # Sampling density = |F|^2 prior * incident spectrum (if provided).
        if wl_center is not None:
            if wl_sigma is None:
                raise ValueError("wl_sigma required when wl_center is given")
            flux = np.exp(-0.5 * ((valid_wl - wl_center) / wl_sigma) ** 2)
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
            noise_vec = np.random.normal(0, angular_std, 3)
            q_exp = q_hat_lab + noise_vec
            q_exp /= np.linalg.norm(q_exp)
            q_exp_list.append(q_exp)

        if num_bg > 0:
            bg_vecs = np.random.normal(0, 1, (num_bg, 3))
            bg_vecs /= np.linalg.norm(bg_vecs, axis=1, keepdims=True)
            q_exp_list.extend(bg_vecs)

        q_lab = np.array(q_exp_list)
        shuffle_idx = np.random.permutation(num_events)
        q_lab = q_lab[shuffle_idx]

        times = np.sort(np.random.uniform(0, duration, num_events))
        banks = np.ones(num_events, dtype=int)
        pixels_r = np.zeros(num_events, dtype=int)
        pixels_c = np.zeros(num_events, dtype=int)

        return q_lab, times, banks, pixels_r, pixels_c, valid_hkl, raw_intensities

    def get_fake_batches(self, sim_data, batch_size=10000):
        q_lab, times, banks, pixels_r, pixels_c = sim_data[:5]
        num_events = len(times)
        for start_idx in range(0, num_events, batch_size):
            end_idx = min(start_idx + batch_size, num_events)
            N = end_idx - start_idx
            yield (
                q_lab[start_idx:end_idx].astype(np.float32),
                times[start_idx:end_idx].astype(np.float32),
                banks[start_idx:end_idx].astype(np.int16),
                pixels_r[start_idx:end_idx].astype(np.int16),
                pixels_c[start_idx:end_idx].astype(np.int16),
                np.zeros((N, 1), dtype=np.float32),                   # angles
                np.zeros((N, 3), dtype=np.float32),                   # s_lab
                np.tile([0.0, 0.0, 1.0], (N, 1)).astype(np.float32),  # ki_sample
                end_idx,                                              # cumulative count
            )

    def _evaluate_cubic_symmetric_error(self, U_true, U_pred):
        min_err_deg = np.inf
        for sym in get_cubic_symmetries():
            U_mate = U_true @ sym
            trace_val = np.clip(np.trace(U_mate.T @ U_pred), -1.0, 3.0)
            err_deg = np.degrees(np.arccos((trace_val - 1.0) / 2.0))
            min_err_deg = min(min_err_deg, err_deg)
        return min_err_deg

    # ---- the test ------------------------------------------------------------

    def test_ewald_window_wavelength_mismatch(self):
        print(f"\n{'='*60}\nExecuting Regression: EWALD WINDOW WAVELENGTH MISMATCH\n{'='*60}")

        U_true = Rotation.from_euler('y', 45.0, degrees=True).as_matrix()
        U_seed = Rotation.from_euler('y', 40.0, degrees=True).as_matrix()  # 5 deg local capture

        # Known incident spectrum -> the band's "truth". Narrow, so the populated
        # reflections form a thin |q|-shell and an off-center band lights a
        # genuinely different set of directions.
        SPECTRUM_CENTER = 2.5   # Angstrom
        SPECTRUM_SIGMA = 0.35

        sim_data = self.generate_poissonian_events(
            U_true, num_events=1_000_000, duration=5.0, bg_fraction=0.0,
            wl_center=SPECTRUM_CENTER, wl_sigma=SPECTRUM_SIGMA,
        )
        valid_hkl, intensities = sim_data[5], sim_data[6]
        mock_mtz = self.create_mock_mtz(valid_hkl, intensities)

        # Pin the reflection POOL so only w_l_j (the band weighting) varies.
        D_MIN, D_MAX = 2.0, 8.0

        # EQUAL-WIDTH bands: only the center differs. This removes the band-width
        # -> prediction-sharpness (Fisher-information) confound that let the
        # off-center band win in the naive support-vs-subset version.
        HALF_W = 0.9
        matched_band = (SPECTRUM_CENTER - HALF_W, SPECTRUM_CENTER + HALF_W)  # ~[1.60, 3.40]
        MISMATCH_CENTER = 5.5
        mismatched_band = (MISMATCH_CENTER - HALF_W, MISMATCH_CENTER + HALF_W)  # ~[4.60, 6.40]
        # Flux at the near edge of the mismatched band (4.60 A) relative to the
        # spectrum: exp(-0.5*((4.60-2.5)/0.35)^2) ~ 1e-8  -> essentially no data
        # in that wavelength shell, yet the band still lights reflections there.

        def run_with_band(wl_min, wl_max):
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
                wl_min_tracking=wl_min, wl_max_tracking=wl_max,
            )
            return self._evaluate_cubic_symmetric_error(U_true, final_U)

        width = matched_band[1] - matched_band[0]
        matched_err = run_with_band(*matched_band)
        print(f"  matched    band [{matched_band[0]:.2f}, {matched_band[1]:.2f}] A "
              f"(width {width:.2f}, ON spectrum @ {SPECTRUM_CENTER} A) -> Sym-Err = {matched_err:6.2f} deg")

        mismatched_err = run_with_band(*mismatched_band)
        print(f"  mismatched band [{mismatched_band[0]:.2f}, {mismatched_band[1]:.2f}] A "
              f"(width {width:.2f}, OFF spectrum)        -> Sym-Err = {mismatched_err:6.2f} deg")

        print(f"  band-induced degradation (mismatched - matched): {mismatched_err - matched_err:+.2f} deg")

        # (2) Control: on-spectrum band must converge (5 deg local capture).
        self.assertLess(
            matched_err, 2.0,
            f"On-spectrum control failed to converge ({matched_err:.2f} deg >= 2.0 deg); "
            f"fix the matched leg before trusting the comparison."
        )

        # (1) Primary, relative (tuning-robust): at EQUAL width, the band centered
        # on the populated wavelength shell must beat the band shifted off it.
        # If this fails, w_l_j is not actually steering the prediction toward the
        # data density -- which is the whole point of the Ewald band.
        self.assertLess(
            matched_err, mismatched_err,
            f"Ewald band centering had no benefit at equal width: "
            f"on-spectrum={matched_err:.2f} deg vs off-spectrum={mismatched_err:.2f} deg. "
            f"Expected the band overlapping the populated wavelength shell to win, "
            f"since w_l_j is the sole channel steering the prediction onto the data density."
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
