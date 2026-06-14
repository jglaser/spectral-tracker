import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from subhkl.instrument.goniometer import lab_to_sample
from spectral_tracker.tracker import tracker, _sample_to_lab_matrix
from tests.testing_utils import BaseTrackerTest

class TestBinghamTracker(BaseTrackerTest):
    
    def test_wilson_intensity_modulation(self):
        print(f"\n{'='*60}\nExecuting Regression: WILSON INTENSITY MODULATION (Low-Q Preference)\n{'='*60}")

        U_true = Rotation.from_euler('y', 45.0, degrees=True).as_matrix()
        U_seed = Rotation.from_euler('y', 40.0, degrees=True).as_matrix()

        with h5py.File(self.finder_file, "w") as f:
            f["sample/a"], f["sample/b"], f["sample/c"] = 10.0, 10.0, 10.0
            f["sample/alpha"], f["sample/beta"], f["sample/gamma"] = 90.0, 90.0, 90.0
            f["sample/space_group"] = b"P 1"
            f["beam/ki_vec"] = np.array([0.0, 0.0, 1.0])
            f["sample/U"] = U_seed

        sim_data = self.generate_poissonian_events(U_true, num_events=1000000, duration=5.0, b_factor=0.5)
        event_stream = self.get_fake_batches(sim_data, batch_size=10000)

        valid_hkl = sim_data[5]
        intensities = sim_data[6]
        mock_mtz = self.create_mock_mtz(valid_hkl, intensities)

        def streaming_callback(time, U_preds, losses, best_idx, neutron_count, new_events, metrics):
            err = self._evaluate_cubic_symmetric_error(U_true, U_preds[best_idx])
            print(f"  -> [t={time:4.2f}s | {neutron_count:6d} evts] Sym-Err={err:6.2f}° | Norm-Gap={metrics['eigengap']:.2f}")

        final_U = tracker(
            finder_file=self.finder_file,
            event_batches=event_stream,
            structure_factors=mock_mtz,
            streaming_callback=streaming_callback,
        )

        final_err = self._evaluate_cubic_symmetric_error(U_true, final_U)
        self.assertLess(final_err, 2.0, f"Wilson Modulation failed to converge: Final Error {final_err:.2f}° >= 2.0°")

    def test_local_capture(self):
        print(f"\n{'='*60}\nExecuting Regression: LOCAL CAPTURE (Seed Err: 5.0°)\n{'='*60}")
        
        U_true = Rotation.from_euler('y', 45.0, degrees=True).as_matrix()
        U_seed = Rotation.from_euler('y', 40.0, degrees=True).as_matrix()
        
        with h5py.File(self.finder_file, "w") as f:
            f["sample/a"], f["sample/b"], f["sample/c"] = 10.0, 10.0, 10.0
            f["sample/alpha"], f["sample/beta"], f["sample/gamma"] = 90.0, 90.0, 90.0
            f["sample/space_group"] = b"P 1"
            f["beam/ki_vec"] = np.array([0.0, 0.0, 1.0])
            f["sample/U"] = U_seed

        sim_data = self.generate_poissonian_events(U_true, num_events=1000000, duration=5.0)
        event_stream = self.get_fake_batches(sim_data, batch_size=10000)

        valid_hkl = sim_data[5]
        intensities = sim_data[6]
        mock_mtz = self.create_mock_mtz(valid_hkl, intensities)
        
        def streaming_callback(time, U_preds, losses, best_idx, neutron_count, new_events, metrics):
            err = self._evaluate_cubic_symmetric_error(U_true, U_preds[best_idx])
            print(f"  -> [t={time:4.2f}s | {neutron_count:6d} evts] Sym-Err={err:6.2f}° | Norm-Gap={metrics['eigengap']:.2f}")

        final_U = tracker(
            finder_file=self.finder_file,
            event_batches=event_stream,
            structure_factors=mock_mtz,
            streaming_callback=streaming_callback,
            L_max=8,
        )
        
        final_err = self._evaluate_cubic_symmetric_error(U_true, final_U)
        self.assertLess(final_err, 2.0, f"Local Capture failed to converge: Final Error {final_err:.2f}° >= 2.0°")

    def test_global_aliasing(self):
        print(f"\n{'='*60}\nExecuting Regression: GLOBAL ALIASING (Seed Err: 30.0°)\n{'='*60}")
        
        U_true = Rotation.from_euler('y', 45.0, degrees=True).as_matrix()
        U_seed = Rotation.from_euler('y', 15.0, degrees=True).as_matrix()
        
        with h5py.File(self.finder_file, "w") as f:
            f["sample/a"], f["sample/b"], f["sample/c"] = 10.0, 10.0, 10.0
            f["sample/alpha"], f["sample/beta"], f["sample/gamma"] = 90.0, 90.0, 90.0
            f["sample/space_group"] = b"P 1"
            f["beam/ki_vec"] = np.array([0.0, 0.0, 1.0])
            f["sample/U"] = U_seed

        sim_data = self.generate_poissonian_events(U_true, num_events=1000000, duration=5.0)
        event_stream = self.get_fake_batches(sim_data, batch_size=10000)

        valid_hkl = sim_data[5]
        intensities = sim_data[6]
        mock_mtz = self.create_mock_mtz(valid_hkl, intensities)
        
        def streaming_callback(time, U_preds, losses, best_idx, neutron_count, new_events, metrics):
            err = self._evaluate_cubic_symmetric_error(U_true, U_preds[best_idx])
            print(f"  -> [t={time:4.2f}s | {neutron_count:6d} evts] Sym-Err={err:6.2f}° | Norm-Gap={metrics['eigengap']:.2f}")

        final_U = tracker(
            finder_file=self.finder_file,
            event_batches=event_stream,
            structure_factors=mock_mtz,
            streaming_callback=streaming_callback,
            L_max=8,
            prior_ridge=0.5,
            meas_weight_2nd=2000.0,
            ridge_inflation=1e-4,
        )
        
        final_err = self._evaluate_cubic_symmetric_error(U_true, final_U)
        self.assertLess(final_err, 2.0, f"Global Aliasing failed to escape trap: Final Error {final_err:.2f}° >= 2.0°")

    def test_background_robustness(self):
        print(f"\n{'='*60}\nExecuting Regression: BACKGROUND ROBUSTNESS (80% Noise, Seed Err: 5.0°)\n{'='*60}")
        
        U_true = Rotation.from_euler('y', 45.0, degrees=True).as_matrix()
        U_seed = Rotation.from_euler('y', 40.0, degrees=True).as_matrix()
        
        with h5py.File(self.finder_file, "w") as f:
            f["sample/a"], f["sample/b"], f["sample/c"] = 10.0, 10.0, 10.0
            f["sample/alpha"], f["sample/beta"], f["sample/gamma"] = 90.0, 90.0, 90.0
            f["sample/space_group"] = b"P 1"
            f["beam/ki_vec"] = np.array([0.0, 0.0, 1.0])
            f["sample/U"] = U_seed

        # Generates a massive 80% uniform random spherical noise!
        sim_data = self.generate_poissonian_events(U_true, num_events=1000000, duration=5.0, bg_fraction=0.80)
        event_stream = self.get_fake_batches(sim_data, batch_size=10000)

        valid_hkl = sim_data[5]
        intensities = sim_data[6]
        mock_mtz = self.create_mock_mtz(valid_hkl, intensities)

        def streaming_callback(time, U_preds, losses, best_idx, neutron_count, new_events, metrics):
            err = self._evaluate_cubic_symmetric_error(U_true, U_preds[best_idx])
            print(f"  -> [t={time:4.2f}s | {neutron_count:6d} evts] Sym-Err={err:6.2f}° | Norm-Gap={metrics['eigengap']:.2f}")

        final_U = tracker(
            finder_file=self.finder_file,
            event_batches=event_stream,
            structure_factors=mock_mtz,
            streaming_callback=streaming_callback,
            L_max=8,
        )
        
        final_err = self._evaluate_cubic_symmetric_error(U_true, final_U)
        self.assertLess(final_err, 2.0, f"Background test failed: Tracker derailed by noise (Final Error {final_err:.2f}° >= 2.0°)")

    def test_soc_background_flash(self):
        print(f"\n{'='*60}\nExecuting Regression: SELF-ORGANIZED CRITICALITY (Dynamic Flash)\n{'='*60}")
        
        U_true = Rotation.from_euler('y', 45.0, degrees=True).as_matrix()
        U_seed = Rotation.from_euler('y', 40.0, degrees=True).as_matrix()

        with h5py.File(self.finder_file, "w") as f:
            f["sample/a"], f["sample/b"], f["sample/c"] = 10.0, 10.0, 10.0
            f["sample/alpha"], f["sample/beta"], f["sample/gamma"] = 90.0, 90.0, 90.0
            f["sample/space_group"] = b"P 1"
            f["beam/ki_vec"] = np.array([0.0, 0.0, 1.0])
            f["sample/U"] = U_seed

        # --- GENERATE A MULTI-PHASE DYNAMIC EVENT STREAM ---
        # Phase 1: Calm Approach (1.5s, 20% Noise)
        data_p1 = self.generate_poissonian_events(U_true, num_events=300000, duration=1.5, bg_fraction=0.20)
        
        # Phase 2: The Flash (2.0s, 98% Noise - Would shatter a static tracker)
        data_p2 = self.generate_poissonian_events(U_true, num_events=400000, duration=2.0, bg_fraction=0.98)
        
        # Phase 3: Recovery (1.5s, 20% Noise)
        data_p3 = self.generate_poissonian_events(U_true, num_events=300000, duration=1.5, bg_fraction=0.20)

        # Concatenate the streams and shift times to be continuous
        q_lab = np.concatenate([data_p1[0], data_p2[0], data_p3[0]])
        
        times_p2 = data_p2[1] + data_p1[1][-1]
        times_p3 = data_p3[1] + times_p2[-1]
        times = np.concatenate([data_p1[1], times_p2, times_p3])
        
        banks = np.concatenate([data_p1[2], data_p2[2], data_p3[2]])
        pixels_r = np.concatenate([data_p1[3], data_p2[3], data_p3[3]])
        pixels_c = np.concatenate([data_p1[4], data_p2[4], data_p3[4]])
        
        sim_data_flash = (q_lab, times, banks, pixels_r, pixels_c)
        event_stream = self.get_fake_batches(sim_data_flash, batch_size=10000)
        
        valid_hkl = data_p1[5]
        intensities = data_p1[6]
        mock_mtz = self.create_mock_mtz(valid_hkl, intensities)
        
        # Telemetry storage for assertions
        recorded_errors = []

        def streaming_callback(time, U_preds, losses, best_idx, neutron_count, new_events, metrics):
            err = self._evaluate_cubic_symmetric_error(U_true, U_preds[best_idx])
            recorded_errors.append((time, err))
            print(f"  -> [t={time:4.2f}s | {neutron_count:6d} evts] Sym-Err={err:6.2f}°")

        final_U = tracker(
            finder_file=self.finder_file,
            event_batches=event_stream,
            structure_factors=mock_mtz,
            streaming_callback=streaming_callback,
        )

        # --- SURVIVAL ASSERTIONS ---
        times_arr = np.array([t[0] for t in recorded_errors])
        errs_arr = np.array([e[1] for e in recorded_errors])

        # Extract Phase 2 slice (The Flash)
        phase2_mask = (times_arr > 1.5) & (times_arr <= 3.5)

        max_err_during_flash = np.max(errs_arr[phase2_mask])
        final_err = errs_arr[-1]

        # 1. Did the tracker maintain topological lock during the flash? (Didn't shatter)
        self.assertLess(max_err_during_flash, 15.0, 
                        f"Tracking Failure: The flash shattered the tracker (Max Error {max_err_during_flash:.2f}° >= 15.0°)")

        # 2. Did it recover absolute precision?
        self.assertLess(final_err, 2.0, 
                        f"Tracking Failure: Failed to regain precision after flash (Final Error {final_err:.2f}° >= 2.0°)")

    def test_thermodynamic_entropy_stabilization(self):
        print(f"\n{'='*60}\nExecuting Regression: THERMODYNAMIC ENTROPY STABILIZATION\n{'='*60}")
        
        U_true = Rotation.from_euler('y', 45.0, degrees=True).as_matrix()
        U_seed = Rotation.from_euler('y', 40.0, degrees=True).as_matrix()
        
        with h5py.File(self.finder_file, "w") as f:
            f["sample/a"], f["sample/b"], f["sample/c"] = 10.0, 10.0, 10.0
            f["sample/alpha"], f["sample/beta"], f["sample/gamma"] = 90.0, 90.0, 90.0
            f["sample/space_group"] = b"P 1"
            f["beam/ki_vec"] = np.array([0.0, 0.0, 1.0])
            f["sample/U"] = U_seed

        # Simulate the "4-panel" scenario: Moderate background, but plenty of time to overfit.
        sim_data = self.generate_poissonian_events(U_true, num_events=10000000, duration=5.0, bg_fraction=0.98)
        event_stream = self.get_fake_batches(sim_data, batch_size=100000)

        valid_hkl = sim_data[5]
        intensities = sim_data[6]
        mock_mtz = self.create_mock_mtz(valid_hkl, intensities)
        
        def streaming_callback(time, U_preds, losses, best_idx, neutron_count, new_events, metrics):
            err = self._evaluate_cubic_symmetric_error(U_true, U_preds[best_idx])
            print(f"  -> [t={time:4.2f}s | {neutron_count:6d} evts] Best-Idx={best_idx:3d} | Sym-Err={err:6.2f}° | Free-Energy={metrics['loss']:.2f}")

        final_U = tracker(
            finder_file=self.finder_file,
            event_batches=event_stream,
            structure_factors=mock_mtz,
            annealing_rate=5,    # Smooth time-driven cooling funnel
            streaming_callback=streaming_callback,
            L_max=8,
        ) 

        final_err = self._evaluate_cubic_symmetric_error(U_true, final_U)
        
        self.assertLess(
            final_err, 
            2.0, 
            f"Thermodynamic Collapse: The tracker overfit to a noise trap. (Final Error {final_err:.2f}° >= 2.0°)"
        )

    def test_resolution_dependent_narrowing(self):
        print(f"\n{'='*60}\nExecuting Regression: RESOLUTION-DEPENDENT NARROWING (Lever Arm)\n{'='*60}")
        
        U_true = Rotation.from_euler('xyz', [15.0, 25.0, 35.0], degrees=True).as_matrix()
        U_seed = Rotation.from_euler('xyz', [16.5, 23.8, 36.2], degrees=True).as_matrix()
        
        with h5py.File(self.finder_file, "w") as f:
            f["sample/a"], f["sample/b"], f["sample/c"] = 8.0, 8.0, 8.0
            f["sample/alpha"], f["sample/beta"], f["sample/gamma"] = 90.0, 90.0, 90.0
            f["sample/space_group"] = b"P 1"
            f["beam/ki_vec"] = np.array([0.0, 0.0, 1.0])
            f["sample/U"] = U_seed

        # Generate a dataset extending deep into the high-Q shell (d_min = 1.5 Angstroms)
        sim_data = self.generate_poissonian_events(
            U_true, num_events=200000, duration=1.0, sigma_q=0.008, bg_fraction=0.50
        )
        event_stream = self.get_fake_batches(sim_data, batch_size=20000)

        valid_hkl = sim_data[5]
        intensities = sim_data[6]
        mock_mtz = self.create_mock_mtz(valid_hkl, intensities)

        recorded_gaps = []
        recorded_errors = []

        def streaming_callback(time, U_preds, losses, best_idx, neutron_count, new_events, metrics):
            err = self._evaluate_cubic_symmetric_error(U_true, U_preds[best_idx])
            recorded_gaps.append(metrics['eigengap'])
            recorded_errors.append(err)
            print(f"  -> [t={time:4.2f}s | {neutron_count:6d} evts] Sym-Err={err:6.2f}° | Eigengap={metrics['eigengap']:.2f}")

        final_U = tracker(
            finder_file=self.finder_file,
            event_batches=event_stream,
            structure_factors=mock_mtz,
            annealing_rate=1.0,      
            d_min=1.5,               
            d_max=8.0,
            L_max=16,                
            process_q_scale_start=1e-4, 
            process_q_scale_end=1e-9,   
            streaming_callback=streaming_callback,
        )

        final_err = self._evaluate_cubic_symmetric_error(U_true, final_U)
        
        self.assertLess(final_err, 0.5, f"Lever arm refinement failed to achieve sub-degree precision. Final Error: {final_err:.2f}°")
        self.assertGreater(recorded_gaps[-1], recorded_gaps[0] * 3.0, "Thermodynamic Failure: Eigengap curvature did not accelerate.")
        self.assertLess(recorded_errors[-1], recorded_errors[0], "Kinematic Failure: Crystalline funnel did not actively refine the seed error.")

    def test_anisotropic_background_rejection(self):
        print(f"\n{'='*60}\nExecuting Regression: ANISOTROPIC (STRUCTURED) BACKGROUND\n{'='*60}")

        U_true = Rotation.from_euler('y', 45.0, degrees=True).as_matrix()
        U_seed = Rotation.from_euler('y', 40.0, degrees=True).as_matrix()

        with h5py.File(self.finder_file, "w") as f:
            f["sample/a"], f["sample/b"], f["sample/c"] = 10.0, 10.0, 10.0
            f["sample/alpha"], f["sample/beta"], f["sample/gamma"] = 90.0, 90.0, 90.0
            f["sample/space_group"] = b"P 1"
            f["beam/ki_vec"] = np.array([0.0, 0.0, 1.0])
            f["sample/U"] = U_seed

        duration = 5.0
        n_sig = 400_000
        n_bg = 600_000

        sig = self.generate_poissonian_events(
            U_true, num_events=n_sig, duration=duration, bg_fraction=0.0
        )
        sig_q, sig_t = sig[0], sig[1]
        valid_hkl, intensities = sig[5], sig[6]

        bg_q, bg_t = self.generate_anisotropic_background(
            n_bg, duration=duration, axis=np.array([0.0, 0.0, 1.0]), spread=0.2
        )

        q_lab = np.concatenate([sig_q, bg_q], axis=0)
        times = np.concatenate([sig_t, bg_t])
        order = np.argsort(times, kind="stable")
        q_lab, times = q_lab[order], times[order]

        N = len(times)
        sim_data = (
            q_lab, times,
            np.ones(N, dtype=int),    
            np.zeros(N, dtype=int),   
            np.zeros(N, dtype=int),   
        )

        event_stream = self.get_fake_batches(sim_data, batch_size=10_000)
        mock_mtz = self.create_mock_mtz(valid_hkl, intensities)

        def streaming_callback(time, U_preds, losses, best_idx, neutron_count, new_events, metrics):
            err = self._evaluate_cubic_symmetric_error(U_true, U_preds[best_idx])
            print(f"  -> [t={time:4.2f}s | {neutron_count:7d} evts] Sym-Err={err:6.2f}° | Norm-Gap={metrics['eigengap']:.2f}")

        final_U = tracker(
            finder_file=self.finder_file,
            event_batches=event_stream,
            structure_factors=mock_mtz,
            streaming_callback=streaming_callback,
            L_max=8,
            lorentz_correction=False,
        )

        final_err = self._evaluate_cubic_symmetric_error(U_true, final_U)
        self.assertLess(final_err, 2.0, "Structured-background rejection failed")

    def test_partial_detector_coverage(self):
        print(f"\n{'='*60}\nExecuting Regression: PARTIAL DETECTOR S^2 COVERAGE\n{'='*60}")

        U_true = Rotation.from_euler('y', 45.0, degrees=True).as_matrix()
        U_seed = Rotation.from_euler('y', 40.0, degrees=True).as_matrix()

        duration = 5.0
        sig = self.generate_poissonian_events(
            U_true, num_events=2_000_000, duration=duration, bg_fraction=0.0
        )
        q_all, t_all = sig[0], sig[1]
        valid_hkl, intensities = sig[5], sig[6]
        mock_mtz = self.create_mock_mtz(valid_hkl, intensities)

        det_axis = np.array([1.0, 0.0, 0.0])
        coverages = [1.0, 0.5, 0.25]

        axis_n = det_axis / np.linalg.norm(det_axis)
        counts = {
            f: int(np.sum((q_all @ axis_n) >= (1.0 - 2.0 * f)))
            for f in coverages
        }
        common_n = min(counts.values())
        print(f"  Common event budget (all levels): {common_n:,}")

        rng = np.random.default_rng(0)
        results = {}

        for f in coverages:
            q_c, t_c = self.apply_detector_coverage(q_all, t_all, coverage_fraction=f, axis=det_axis)
            idx = rng.choice(len(t_c), size=common_n, replace=False)
            q_c, t_c = q_c[idx], t_c[idx]
            order = np.argsort(t_c, kind="stable")  
            q_c, t_c = q_c[order], t_c[order]

            N = len(t_c)
            sim_data = (q_c, t_c, np.ones(N, dtype=int), np.zeros(N, dtype=int), np.zeros(N, dtype=int))
            event_stream = self.get_fake_batches(sim_data, batch_size=10_000)

            with h5py.File(self.finder_file, "w") as fh:
                fh["sample/a"], fh["sample/b"], fh["sample/c"] = 10.0, 10.0, 10.0
                fh["sample/alpha"], fh["sample/beta"], fh["sample/gamma"] = 90.0, 90.0, 90.0
                fh["sample/space_group"] = b"P 1"
                fh["beam/ki_vec"] = np.array([0.0, 0.0, 1.0])
                fh["sample/U"] = U_seed

            final_U = tracker(
                finder_file=self.finder_file,
                event_batches=event_stream,
                structure_factors=mock_mtz,
                L_max=8,
            )
            err = self._evaluate_cubic_symmetric_error(U_true, final_U)
            results[f] = err
            print(f"  coverage={f*100:5.1f}% of 4pi | events={N:,} | final Sym-Err={err:6.2f} deg")

        self.assertLess(results[1.0], 2.0, "Full-coverage control failed")
        self.assertLess(results[0.5], 10.0, "Partial-coverage bias failed")

    def test_sample_lab_transform_roundtrip(self):
        print(f"\n{'='*60}\nUnit: SAMPLE<->LAB ROUND-TRIP (finite angles + offsets)\n{'='*60}")
        rng = np.random.default_rng(0)

        cases = [
            (np.array([[0.0, 1.0, 0.0]]), np.array([30.0]), None),
            (np.array([[0.0, 1.0, 0.0]]), np.array([30.0]), np.array([5.0])),
            (np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]), np.array([25.0, 15.0]), np.array([3.0, -2.0])),
            (np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), np.array([40.0, -20.0, 10.0]), np.array([1.0, 2.0, -3.0])),
        ]

        for axes, ang, offsets in cases:
            na = len(axes)
            R = np.asarray(_sample_to_lab_matrix(axes, ang, offsets), dtype=float)

            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-4)
            self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=4)

            v_lab = rng.normal(size=(64, 3))
            v_lab /= np.linalg.norm(v_lab, axis=1, keepdims=True)
            ang_full = np.tile(ang.reshape(na, 1), (1, v_lab.shape[0])) 
            v_sample = np.asarray(lab_to_sample(v_lab, axes, ang_full, None, offsets, is_vector=True))
            v_lab_rec = (R @ v_sample.T).T

            np.testing.assert_allclose(v_lab_rec, v_lab, atol=1e-4)

    def test_goniometer_finite_setting(self):
        print(f"\n{'='*60}\nIntegration: FINITE GONIOMETER SETTING (+offsets)\n{'='*60}")
     
        axes = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]) 
        ang = np.array([20.0, -12.0])                          
        offs = np.array([5.0, -3.0])                           
        na = len(axes)
     
        U_true = Rotation.from_euler("y", 45.0, degrees=True).as_matrix()
        U_seed = Rotation.from_euler("y", 40.0, degrees=True).as_matrix() 
     
        R = np.asarray(_sample_to_lab_matrix(axes, ang, offs), dtype=float)
        U_eff = R @ U_true
     
        self.assertGreater(self._evaluate_cubic_symmetric_error(U_true, U_eff), 5.0)
     
        sim = self.generate_poissonian_events(U_eff, num_events=1_000_000, duration=5.0, bg_fraction=0.0)
        q_lab, t = sim[0], sim[1]
        valid_hkl, intensities = sim[5], sim[6]
        mock_mtz = self.create_mock_mtz(valid_hkl, intensities)
     
        N = len(t)
        ang_full = np.tile(ang.reshape(na, 1), (1, N))                    
        q_sample = np.asarray(lab_to_sample(q_lab, axes, ang_full, None, offs, is_vector=True))
        ki_sample = np.asarray(lab_to_sample(np.tile([0.0, 0.0, 1.0], (N, 1)), axes, ang_full, None, offs, is_vector=True))
        angles_col = np.tile(ang.reshape(1, na), (N, 1))                  
     
        def emit(batch=10000):
            order = np.argsort(t, kind="stable")
            qs, ts, ac, ks = q_sample[order], t[order], angles_col[order], ki_sample[order]
            for s in range(0, N, batch):
                e = min(s + batch, N)
                n = e - s
                yield (
                    qs[s:e].astype(np.float32), ts[s:e].astype(np.float32),
                    np.ones(n, dtype=np.int16), np.zeros(n, dtype=np.int16), np.zeros(n, dtype=np.int16),
                    ac[s:e].astype(np.float32), np.zeros((n, 3), dtype=np.float32), ks[s:e].astype(np.float32), e
                )
     
        with h5py.File(self.finder_file, "w") as f:
            f["sample/a"], f["sample/b"], f["sample/c"] = 10.0, 10.0, 10.0
            f["sample/alpha"], f["sample/beta"], f["sample/gamma"] = 90.0, 90.0, 90.0
            f["sample/space_group"] = b"P 1"
            f["beam/ki_vec"] = np.array([0.0, 0.0, 1.0])
            f["sample/U"] = U_seed
     
        final_U = tracker(
            finder_file=self.finder_file,
            event_batches=emit(),
            structure_factors=mock_mtz,
            gonio_axes=axes,
            gonio_offsets=offs,
            L_max=8,
        )
     
        err = self._evaluate_cubic_symmetric_error(U_true, final_U)
        self.assertLess(err, 2.0, "Finite-setting recovery failed")


class TestTrackerInitialization:
    
    def test_local_capture_initialization_gauge(self, mock_reciprocal_h5):
        """
        Verifies that the tracking prior correctly imports the initial seed 
        without introducing any stride, layout, or transposition offsets.
        """
        h5_file, U_seed, _ = mock_reciprocal_h5
        
        # Mock an empty single-step batch array to isolate the initialization block
        mock_batch = [
            (
                np.zeros((0, 3), dtype=np.float32),  # q_batch
                np.zeros((0,), dtype=np.float32),    # t_batch
                np.zeros((0,), dtype=np.int16),      # banks
                np.zeros((0,), dtype=np.int16),      # pr
                np.zeros((0,), dtype=np.int16),      # pc
                np.zeros((0, 1), dtype=np.float32),  # angles
                np.zeros((0, 3), dtype=np.float32),  # slab
                np.zeros((0, 3), dtype=np.float32),  # ki_sample
                0,                                   # cumulative count
            )
        ]

        final_U = tracker(
            finder_file=h5_file,
            event_batches=mock_batch,
            L_max=8,
        )
        
        trace_val = np.clip((np.trace(final_U.T @ U_seed) - 1.0) / 2.0, -1.0, 1.0)
        angular_error_deg = np.degrees(np.arccos(trace_val))
        
        print(f"\n[Validation Test] Extracted Angle Error to Seed Matrix: {angular_error_deg:.6f}°")
        assert angular_error_deg < 0.05, f"Gauge error detected! Tracker scrambled the input matrix at startup."
