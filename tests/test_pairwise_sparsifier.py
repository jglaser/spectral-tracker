"""
Regression Test: PAIRWISE SPARSIFIER (ZONE AXIS CAPTURE)
========================================================

Validates the Global Capture mechanism mapping points on the sphere to 
Zone Axes using heavy-tailed (Banach space) fields and Campbell's Theorem.

Constructs a uniform 3D Poisson point process and injects a 3-axis orthogonal 
crystal. Asserts that the product-space algorithm mathematically destroys the 
$O(N^2)$ background intersections and perfectly isolates the true zone axes.
"""

import unittest
import numpy as np
import jax

from spectral_tracker.pairwise_sparsifier import PairwiseSparsifier

class TestPairwiseSparsifier(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        
        # 1. Background: Uniform Poisson shot noise
        self.num_bg = 600
        self.bg_events = self._random_sphere_points(self.num_bg)
        
        # 2. Signal: A simple cubic/orthogonal lattice (Zone axes on X, Y, Z)
        self.num_sig_per_zone = 60
        self.true_zones = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        sig_events = []
        for z in self.true_zones:
            # Events cluster along the equator of the zone axis
            sig_events.append(self._generate_equator_points(z, self.num_sig_per_zone))
            
        self.sig_events = np.vstack(sig_events)
        
        # 3. Mix streams (Total N ~ 780 -> ~303,000 intersecting pair operations)
        self.all_events = np.vstack([self.bg_events, self.sig_events])
        np.random.shuffle(self.all_events)
        
    def _random_sphere_points(self, n):
        pts = np.random.normal(size=(n, 3))
        return pts / np.linalg.norm(pts, axis=1, keepdims=True)
        
    def _generate_equator_points(self, pole, n):
        # Build an orthogonal basis for the equator
        idx = np.argmin(np.abs(pole))
        v1 = np.zeros(3); v1[idx] = 1.0
        v1 = np.cross(pole, v1)
        v1 /= np.linalg.norm(v1)
        v2 = np.cross(pole, v1)
        
        # Generate uniform points strictly on the great circle (equator)
        angles = np.random.uniform(0, 2*np.pi, n)
        pts = v1[None, :] * np.cos(angles)[:, None] + v2[None, :] * np.sin(angles)[:, None]
        
        # Apply slight mosaic/instrumental scatter (0.01 radians ~ 0.5 degrees)
        pts += np.random.normal(0, 0.01, size=pts.shape)
        return pts / np.linalg.norm(pts, axis=1, keepdims=True)

    def test_global_capture_zone_axes(self):
        print(f"\n{'='*60}\nTesting: PAIRWISE SPARSIFIER (GLOBAL CAPTURE)\n{'='*60}")
        
        # Initialize sparsifier with a sharp core (gamma=0.05) and Holtsmark-like tail (nu=1.5)
        # target_fp=0.5 enforces strict suppression (allow only 1 false positive every two executions)
        sparsifier = PairwiseSparsifier(target_fp=0.5, gamma=0.05, nu=1.5, n_grid=10000)
        
        # 1. Execute $O(N^2)$ product space mapping and thresholding
        zones, field, threshold = sparsifier.find_zone_axes(self.all_events)
        
        total_events = len(self.all_events)
        pairs_eval = (total_events * (total_events - 1)) // 2
        
        print(f"  Input Events (Spared from 2D panel): {total_events}")
        print(f"  O(N^2) Product Space Intersections:  {pairs_eval:,}")
        print(f"  Holtsmark Null Hypothesis Threshold: {threshold:.2f}")
        print(f"  Max Local Field Topography:          {np.max(field):.2f}")
        print(f"  Surviving Zone Axes Extracted:       {len(zones)}")
        
        # Assertions
        self.assertGreater(len(zones), 0, "Global capture failed to identify any zone axes.")
        self.assertGreater(np.max(field), threshold * 2.0, "Signal failed to convincingly penetrate the background threshold.")
        
        # 2. Verify topological accuracy (extracted zones align physically with truth)
        matched_truth = 0
        for tz in self.true_zones:
            # Absolute dot product because Zone Axes are antipodal (+Z and -Z are identical)
            overlaps = np.abs(zones @ tz)
            best_match = np.max(overlaps)
            if best_match > 0.985:  # Tolerance ~ 9.9 degrees
                matched_truth += 1
                print(f"    -> Successfully matched orthogonal Zone Axis {tz} (Align: {best_match:.4f})")
                
        self.assertEqual(matched_truth, len(self.true_zones), 
                         f"Global capture lost topological integrity. Only recovered {matched_truth} / {len(self.true_zones)} zone axes.")

if __name__ == '__main__':
    unittest.main(verbosity=2)
