import unittest
import tempfile
import os
import h5py
import numpy as np
import itertools
from scipy.spatial.transform import Rotation

def get_cubic_symmetries():
    """Generates the 24 valid rotation matrices for a Cubic point group."""
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

class BaseTrackerTest(unittest.TestCase):
    """Base class for Tracker tests containing shared simulation and evaluation utilities."""
    
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.finder_file = os.path.join(self.test_dir.name, "mock_finder.h5")
        
        # Ensure deterministic testing environment
        np.random.seed(42)

    def tearDown(self):
        self.test_dir.cleanup()

    def create_mock_mtz(self, hkl_array, intensities):
        """ Wraps exact test-generated intensities into a valid Gemmi object. """
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

    def generate_poissonian_events(self, U_true, num_events=1000000, duration=5.0, sigma_q=0.008, bg_fraction=0.0, b_factor=0.0):
        # Busing-Levy convention (1/d) to match the tracker's geometry exactly
        B_mat = np.array([
            [1.0/10.0, 0, 0],
            [0, 1.0/10.0, 0],
            [0, 0, 1.0/10.0]
        ])
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
        
        # True kinematics in Busing-Levy space
        wavelengths = -(2.0 / q_norms) * kinematic_proj

        valid_mask = (wavelengths > 0.5) & (wavelengths < 10.0)
        valid_q_hat = q_theo_hat[:, valid_mask]
        valid_norms = q_norms[valid_mask]
        valid_hkl = hkl[:, valid_mask] 
        num_valid = valid_q_hat.shape[1]

        num_bg = int(num_events * bg_fraction)
        num_sig = num_events - num_bg

        # --- THE WILSON PRIOR (Intensity Decay) ---
        if b_factor > 0.0:
            raw_intensities = np.exp(-(b_factor * 39.47) * (valid_norms**2))
            p_dist = raw_intensities / np.sum(raw_intensities)
        else:
            raw_intensities = np.ones_like(valid_norms)
            p_dist = None

        peak_indices = np.random.choice(num_valid, size=num_sig, p=p_dist)

        q_exp_list = []
        # 1. Generate Physical Signal Events
        for idx in peak_indices:
            q_hat_lab = U_true @ valid_q_hat[:, idx]
            
            angular_std = sigma_q / valid_norms[idx]
            noise_vec = np.random.normal(0, angular_std, 3)
            q_exp = q_hat_lab + noise_vec
            q_exp /= np.linalg.norm(q_exp)
            q_exp_list.append(q_exp)

        # 2. Generate Uniform Background Noise Events
        if num_bg > 0:
            bg_vecs = np.random.normal(0, 1, (num_bg, 3))
            bg_vecs /= np.linalg.norm(bg_vecs, axis=1, keepdims=True)
            q_exp_list.extend(bg_vecs)

        q_lab = np.array(q_exp_list)

        # Shuffle to mix background and signal evenly across time
        shuffle_idx = np.random.permutation(num_events)
        q_lab = q_lab[shuffle_idx]

        times = np.sort(np.random.uniform(0, duration, num_events)) 
        banks = np.ones(num_events, dtype=int)
        pixels_r = np.zeros(num_events, dtype=int)
        pixels_c = np.zeros(num_events, dtype=int)

        return q_lab, times, banks, pixels_r, pixels_c, valid_hkl, raw_intensities

    def get_fake_batches(self, sim_data, batch_size=10000):
        """Yields streaming tuples exactly matching the EventStreamLoader signature."""
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
                np.zeros((N, 1), dtype=np.float32),  # angles
                np.zeros((N, 3), dtype=np.float32),  # s_lab
                np.tile([0.0, 0.0, 1.0], (N, 1)).astype(np.float32), # ki_sample
                end_idx # cumulative count
            )

    def _evaluate_cubic_symmetric_error(self, U_true, U_pred):
        min_err_deg = np.inf
        for sym in get_cubic_symmetries():
            U_mate = U_true @ sym
            trace_val = np.clip(np.trace(U_mate.T @ U_pred), -1.0, 3.0)
            err_deg = np.degrees(np.arccos((trace_val - 1.0) / 2.0))
            min_err_deg = min(min_err_deg, err_deg)
        return min_err_deg

    def generate_anisotropic_background(self, num_bg, duration=5.0,
                                        axis=np.array([0.0, 0.0, 1.0]),
                                        spread=0.35):
        axis = axis / np.linalg.norm(axis)
        vecs = axis[None, :] + np.random.normal(0.0, spread, size=(num_bg, 3))
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        times = np.sort(np.random.uniform(0.0, duration, num_bg))
        return vecs.astype(np.float32), times.astype(np.float32)

    def apply_detector_coverage(self, q, times, coverage_fraction,
                                axis=np.array([1.0, 0.0, 0.0])):
        axis = axis / np.linalg.norm(axis)
        cos_a = 1.0 - 2.0 * coverage_fraction
        cos_theta = q @ axis
        keep = cos_theta >= cos_a
        return q[keep], times[keep]
