"""
Regression Test: EVENT STREAM LOADER & DYNAMIC SPARSIFICATION
=============================================================

Validates the chronological loader and the dynamic `EventStreamSparsifier`.
Constructs a mock HDF5 NeXus layout containing a 100x100 detector bank
flooded with uniform Poisson noise and a dense synthetic Bragg peak.

Assertions:
  1. Unfiltered Loader: Yields exactly 100% of the raw events.
  2. Sparsified Loader: Actively thresholds the 2D local field, 
     crushing the total event count while preserving the Bragg signal.
"""

import unittest
import os
import tempfile
import h5py
import numpy as np
from unittest.mock import patch, MagicMock

from spectral_tracker.streaming.loader import EventStreamLoader

class TestStreamingLoaderSparsification(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.nexus_file = os.path.join(self.test_dir.name, "mock_events.h5")
        
        # Detector dimensions
        self.nx = 100
        self.ny = 100
        
        # Event Statistics
        self.num_bg = 20000  # ~2 events per pixel uniformly
        self.num_sig = 2000  # Dense cluster
        self.total_events = self.num_bg + self.num_sig
        
        # Bragg Peak Location (r=50, c=50)
        self.peak_r = 50
        self.peak_c = 50
        
        np.random.seed(42)
        with h5py.File(self.nexus_file, 'w') as f:
            bank = f.create_group('/entry/bank1_events')
            
            # 1. Background: Uniformly distributed pixels
            bg_pr = np.random.randint(0, self.nx, size=self.num_bg)
            bg_pc = np.random.randint(0, self.ny, size=self.num_bg)
            
            # 2. Signal: Gaussian cluster tightly packed around the peak
            sig_pr = np.clip(np.random.normal(self.peak_r, 1.5, size=self.num_sig), 0, self.nx - 1).astype(int)
            sig_pc = np.clip(np.random.normal(self.peak_c, 1.5, size=self.num_sig), 0, self.ny - 1).astype(int)
            
            all_pr = np.concatenate([bg_pr, sig_pr])
            all_pc = np.concatenate([bg_pc, sig_pc])
            
            # Convert 2D (r,c) to 1D local ID (YAxisIsFastVaryingIndex)
            all_ids = (all_pc * self.nx + all_pr).astype(np.uint32)
            
            # Shuffle chronologically to emulate a real mixed stream
            shuffle_idx = np.random.permutation(self.total_events)
            all_ids = all_ids[shuffle_idx]
            
            # Synthetic pulse times in microseconds
            times = np.sort(np.random.uniform(0, 5e6, self.total_events)).astype(np.float32)
            
            bank.create_dataset('event_id', data=all_ids)
            bank.create_dataset('event_time_offset', data=times)
            bank.create_dataset('event_time_zero', data=np.array([0.0], dtype=np.float32))
            bank.create_dataset('event_index', data=np.array([0], dtype=np.uint32))

    def tearDown(self):
        self.test_dir.cleanup()

    @patch.dict('spectral_tracker.streaming.loader.beamlines', {"MOCK_INST": {"1": {"n": 100, "m": 100, "offset": 0}}})
    @patch.dict('spectral_tracker.streaming.loader.reduction_settings', {"MOCK_INST": {"YAxisIsFastVaryingIndex": True}})
    @patch('spectral_tracker.streaming.loader.Detector')
    def test_loader_with_and_without_sparsification(self, MockDetector):
        # 1. Mock the Detector instance properties
        def mock_detector_init(det_config):
            m = MagicMock()
            m.n = det_config.get('n', 100)
            m.m = det_config.get('m', 100)
            # Mock spatial mapping to bypass rigorous layout generation
            m.pixel_to_lab.side_effect = lambda pr, pc: np.column_stack([pr, pc, np.ones_like(pr, dtype=float)])
            return m
            
        MockDetector.side_effect = mock_detector_init
        
        ki_vec = np.array([0.0, 0.0, 1.0])
        sample_offset = np.array([0.0, 0.0, 0.0])
        
        print(f"\n{'='*60}\nTesting: UNFILTERED LOADER STREAM\n{'='*60}")
        loader_control = EventStreamLoader(
            event_nexus_filename=self.nexus_file,
            instrument_name="MOCK_INST",
            ki_vec=ki_vec,
            sample_offset=sample_offset,
        )
        
        self.assertEqual(loader_control.total_events, self.total_events)
        
        # Stream the full dataset as one massive batch
        batches_unfiltered = list(loader_control.get_batches(
            batch_size_events=50000, 
            use_sparsifier=False
        ))
        
        total_unfiltered = sum(len(b[1]) for b in batches_unfiltered)
        self.assertEqual(total_unfiltered, self.total_events, "Unfiltered stream dropped valid raw events.")
        print(f"  [Control] Successfully streamed 100% of raw events: {total_unfiltered:,}")


        print(f"\n{'='*60}\nTesting: SPARSIFIED LOADER STREAM\n{'='*60}")
        loader_sparsified = EventStreamLoader(
            event_nexus_filename=self.nexus_file,
            instrument_name="MOCK_INST",
            ki_vec=ki_vec,
            sample_offset=sample_offset,
        )
        
        # The sparsifier expects `lambda_bg` to be proportional to per_pixel_rate * batch_size.
        # Since we load all 22,000 events in one batch, the expected background lambda is 
        # (20,000 bg events / 10,000 pixels) = 2.0 events/pixel.
        # Therefore, per_pixel_rate = 2.0 / 22000.
        target_lambda = self.num_bg / (self.nx * self.ny)
        rate = target_lambda / self.total_events
        
        batches_filtered = list(loader_sparsified.get_batches(
            batch_size_events=50000, 
            use_sparsifier=True, 
            per_pixel_rate=rate
        ))
        
        total_filtered = sum(len(b[1]) for b in batches_filtered)
        
        # Accumulate retained pixels
        pr_filtered = np.concatenate([b[3] for b in batches_filtered])
        pc_filtered = np.concatenate([b[4] for b in batches_filtered])
        
        # Analyze Signal Retention (ROI defined as ±5 pixels around the peak)
        roi_mask = (np.abs(pr_filtered - self.peak_r) <= 5) & (np.abs(pc_filtered - self.peak_c) <= 5)
        signal_retained = np.sum(roi_mask)
        
        print(f"  [Sparsifier] Raw Event Input:  {total_unfiltered:,}")
        print(f"  [Sparsifier] Filtered Output:  {total_filtered:,}")
        print(f"  [Sparsifier] Background Crush: {100.0 * (1 - total_filtered / total_unfiltered):.1f}% volume reduction")
        print(f"  [Sparsifier] Bragg Retention:  {signal_retained:,} / {self.num_sig:,} ({100.0 * signal_retained / self.num_sig:.1f}%)")

        # Assertions
        # 1. Total volume must be crushed by at least 80% (dropping the uniform Poisson background)
        self.assertLess(total_filtered, self.total_events * 0.20, 
                        "Sparsifier failed to discard the expected volume of background noise.")
        
        # 2. Bragg Signal must be protected (at least 85% survival rate)
        self.assertGreater(signal_retained, self.num_sig * 0.85, 
                           "Sparsifier's heavy-tail threshold sheared off the physical Bragg peak.")

if __name__ == '__main__':
    unittest.main(verbosity=2)
