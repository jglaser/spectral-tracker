import h5py
import numpy as np
import scipy.spatial
from scipy.spatial.transform import Rotation

from spectral_tracker.tracker import tracker
from spectral_tracker import spectrum_learning as sl
from tests.testing_utils import BaseTrackerTest

class TestSpectrumLearningIntegration(BaseTrackerTest):
    
    def test_spectrum_learning_integration(self):
        print(f"\n{'='*60}\nExecuting Regression: SPECTRUM LEARNING & LORENTZ WIRING\n{'='*60}")
        
        U_true = Rotation.from_euler('y', 45.0, degrees=True).as_matrix()
        U_seed = Rotation.from_euler('y', 20.0, degrees=True).as_matrix() # 25 degree trap!
        
        # 1. Generate strongly skewed physical data (Maxwellian + Lorentz)
        B_mat = np.diag([1/10., 1/10., 1/10.])
        ki_vec = np.array([0.0, 0.0, 1.0])

        h_vals = np.arange(-3, 4)
        hc, kc, lc = np.meshgrid(h_vals, h_vals, h_vals, indexing="ij")
        hkl = np.stack([hc.flatten(), kc.flatten(), lc.flatten()], axis=0)
        hkl = hkl[:, ~((hkl[0] == 0) & (hkl[1] == 0) & (hkl[2] == 0))]

        q_theo = B_mat @ hkl
        q_norms = np.linalg.norm(q_theo, axis=0)
        q_theo_hat = q_theo / q_norms

        lam_true = -(2.0 / q_norms) * (ki_vec.T @ (U_true @ q_theo_hat))
        valid = (lam_true > 0.5) & (lam_true < 10.0)

        q_valid = q_theo_hat[:, valid]
        q_norms_valid = q_norms[valid]
        lam_valid = lam_true[valid]
        valid_hkl = hkl[:, valid]

        # Heavy Maxwellian skew (peaks at ~2.5 A)
        lam_T = 2.5
        phi_true = lam_valid**(-5) * np.exp(-(lam_T/lam_valid)**2)
        L_true = 4.0 * lam_valid**2 / q_norms_valid**2
        
        # The true relative intensities combining structure, spectrum, and Lorentz geometry
        intensities = phi_true * L_true
        p_dist = intensities / np.sum(intensities)

        num_events = 250_000
        peak_indices = np.random.choice(len(lam_valid), size=num_events, p=p_dist)

        q_exp_list = []
        for idx in peak_indices:
            q_hat_lab = U_true @ q_valid[:, idx]
            noise = np.random.normal(0, 0.008 / q_norms_valid[idx], 3)
            q_exp = q_hat_lab + noise
            q_exp /= np.linalg.norm(q_exp)
            q_exp_list.append(q_exp)

        q_lab = np.array(q_exp_list)
        times = np.linspace(0, 5.0, num_events)

        sim_data = (q_lab, times, np.ones(num_events, dtype=int), 
                    np.zeros(num_events, dtype=int), np.zeros(num_events, dtype=int))
        
        # Reset H5 helper
        def reset_h5():
            with h5py.File(self.finder_file, "w") as f:
                f["sample/a"], f["sample/b"], f["sample/c"] = 10.0, 10.0, 10.0
                f["sample/alpha"], f["sample/beta"], f["sample/gamma"] = 90.0, 90.0, 90.0
                f["sample/space_group"] = b"P 1"
                f["beam/ki_vec"] = np.array([0.0, 0.0, 1.0])
                f["sample/U"] = U_seed

        # --- PART 1: TRACKER FAILS WITHOUT SPECTRUM LEARNING ---
        reset_h5()
        final_U_flat = tracker(
            finder_file=self.finder_file,
            event_batches=self.get_fake_batches(sim_data, batch_size=10000),
            L_max=8,
            assumed_spectrum=None  # Standard flat top-hat assumption
        )
        err_flat = self._evaluate_cubic_symmetric_error(U_true, final_U_flat)
        print(f"  [Flat Assumption] Final Error: {err_flat:.2f}°")
        self.assertGreater(err_flat, 5.0, "Expected tracker to fail escaping trap with flat spectrum")

        # --- PART 2: LEARN SPECTRUM AND SUCCEED ---
        reset_h5()
        
        # 1. Grab first 20k events to estimate counts near the seed U
        q_pred_seed = U_seed @ q_valid
        lam_seed = -(2.0 / q_norms_valid) * (ki_vec.T @ q_pred_seed)
        tree = scipy.spatial.cKDTree(q_pred_seed.T)
        dist, nn = tree.query(q_lab[:20000])
        keep = dist < 0.05
        counts = np.bincount(nn[keep], minlength=len(lam_valid))

        # 2. Extract pure spectrum by dividing out the Lorentz factor in the learner
        geom_seed = np.where(q_norms_valid > 0, 4.0 * lam_seed**2 / q_norms_valid**2, 1.0)
        params, learned_phi, info = sl.learn_spectrum(
            lambdas=lam_seed,
            counts=counts,
            intensities=np.ones_like(counts), # No MTZ provided in this test
            geom=geom_seed,                   # Learner divides out Lorentz
            family="maxwellian",
            lam_band=(1.0, 8.0)
        )

        # 3. Wire the learned closure into the tracking loop
        final_U_learned = tracker(
            finder_file=self.finder_file,
            event_batches=self.get_fake_batches(sim_data, batch_size=10000),
            L_max=8,
            assumed_spectrum=learned_phi,  # Inject the learned function
            lorentz_correction=True        # Re-apply Lorentz natively inside Ewald integrations
        )

        err_learned = self._evaluate_cubic_symmetric_error(U_true, final_U_learned)
        print(f"  [Learned Spectrum] Final Error: {err_learned:.2f}°")
        self.assertLess(err_learned, 2.0, "Tracker failed to converge even with the correct wired spectrum")
