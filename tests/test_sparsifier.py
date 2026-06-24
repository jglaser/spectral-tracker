"""
Unit tests for the 2D Local Field Sparsifier (Event Classification).

Evaluates non-linear thresholding of 2D neutron events using heavy-tailed 
Point Spread Functions (PSFs). Validates the Campbell/Gamma analytical 
thresholding strategy derived from segment fluid models.
"""

import unittest
import numpy as np
from scipy.signal import fftconvolve
import scipy.stats
from scipy.ndimage import binary_dilation

class TestLocalFieldSparsifier(unittest.TestCase):
    def setUp(self):
        """Initialize the 2D detector grid with a uniform background and dense Bragg peaks."""
        self.nx, self.ny = 256, 256
        self.rng = np.random.default_rng(42)
        
        # 1. Generate Uniform Poisson Background
        self.bg_rate = 0.05
        self.background = self.rng.poisson(lam=self.bg_rate, size=(self.nx, self.ny))
        
        # 2. Inject Bragg Peaks (Dense Clusters)
        self.signal = np.zeros((self.nx, self.ny))
        self.peak_locations = [
            (50, 50), (200, 200), (50, 200), (200, 50), (128, 128)
        ]
        self.peak_radius = 2
        
        for (cx, cy) in self.peak_locations:
            # Inject a 5x5 dense cluster for each peak
            self.signal[cx-self.peak_radius:cx+self.peak_radius+1, 
                        cy-self.peak_radius:cy+self.peak_radius+1] = self.rng.poisson(lam=4.0, size=(5, 5))
            
        self.events = self.background + self.signal
        
        # 3. Create Ground Truth Mask for Evaluation
        # Dilation is required because heavy-tailed PSFs physically spread the signal.
        # A strict 5x5 mask will penalize the valid kernel footprint as false positives.
        raw_mask = self.signal > 0
        self.truth_mask = binary_dilation(raw_mask, iterations=4)

        # 4. Initialize Spatial Grid for Kernels
        kx = np.linspace(-20, 20, 128)
        ky = np.linspace(-20, 20, 128)
        KX, KY = np.meshgrid(kx, ky)
        self.R_sq = KX**2 + KY**2
        self.R = np.sqrt(self.R_sq)

    def _generate_test_grid(self, bg_rate, signal_rate=4.0):
        """Dynamically generates a test grid for sweeping parameters."""
        background = self.rng.poisson(lam=bg_rate, size=(self.nx, self.ny))
        signal = np.zeros((self.nx, self.ny))
        for (cx, cy) in self.peak_locations:
            signal[cx-self.peak_radius:cx+self.peak_radius+1, 
                   cy-self.peak_radius:cy+self.peak_radius+1] = self.rng.poisson(lam=signal_rate, size=(5, 5))
        
        events = background + signal
        truth_mask = binary_dilation(signal > 0, iterations=4)
        return events, truth_mask

    def _calculate_production_threshold(self, kernel, bg_rate, batch_pixels, target_fp=0.5):
        """
        RECIPE FOR PRODUCTION SPARSIFICATION:
        Computes the optimal local field threshold as a function of the background 
        event rate (lambda) and the spatial batch size (number of pixels).

        Parameters:
        -----------
        kernel : ndarray
            The 2D Point Spread Function (must be L1 normalized so sum(kernel) == 1).
        bg_rate : float
            The average background events per pixel (lambda).
        batch_pixels : int
            The number of pixels evaluated in the current chunk/batch.
        target_fp : float
            The acceptable expected number of false positive background pixels per batch. 
            Default is 0.5 (one false positive every two batches).

        Returns:
        --------
        kernel : ndarray (L1 normalized)
        threshold : float (absolute local field threshold)
        """
        # L1 Normalization ensures sum(kernel) == 1
        kernel = kernel / np.sum(kernel)
        
        # 1. Exact Discrete Campbell Cumulants
        # kappa_1 = lambda * sum(V)
        kappa_1 = bg_rate * 1.0  
        # kappa_2 = lambda * sum(V^2)
        kappa_2 = bg_rate * np.sum(kernel**2)
        
        # 2. Method of Moments mapping to Gamma distribution
        theta = kappa_2 / kappa_1
        k = kappa_1 / theta
        
        # 3. Batch-Size Dependent Percentile
        # To strictly limit the expected false positives across the entire batch,
        # the survival probability (1 - CDF) must equal (target_fp / batch_pixels).
        percentile = 1.0 - (target_fp / batch_pixels)
        percentile = min(max(percentile, 0.0), 1.0 - 1e-15) # Numerical safety
        
        # 4. Determine analytical threshold via Percent Point Function (Inverse CDF)
        threshold = scipy.stats.gamma.ppf(percentile, a=k, scale=theta)
        return kernel, threshold

    def _evaluate_classifier(self, field, threshold, name, custom_truth_mask=None):
        """Calculates classification metrics against the ground truth mask."""
        detected = field > threshold
        mask = self.truth_mask if custom_truth_mask is None else custom_truth_mask
        
        true_positives = np.sum(detected & mask)
        false_positives = np.sum(detected & ~mask)
        false_negatives = np.sum(~detected & mask)
        
        precision = true_positives / max((true_positives + false_positives), 1)
        recall = true_positives / max((true_positives + false_negatives), 1)
        f1_score = 2 * (precision * recall) / max((precision + recall), 1e-9)
        
        print(f"  [{name}] Precision: {precision:.3f} | Recall: {recall:.3f} | F1-Score: {f1_score:.3f}")
        return precision, recall, f1_score

    def test_moffat_levy_sparsifier(self):
        """Evaluates the 2D Moffat kernel (nu=2) which yields a Lévy tail."""
        print(f"\n{'='*60}\nTesting: MOFFAT (LÉVY TAIL) SPARSIFIER\n{'='*60}")
        
        gamma = 1.5
        raw_kernel = 1.0 / (1.0 + (self.R_sq / gamma**2))**2
        
        # Get analytical threshold based on background rate and total detector pixels
        batch_pixels = self.nx * self.ny
        kernel, threshold = self._calculate_production_threshold(
            raw_kernel, self.bg_rate, batch_pixels, target_fp=5.0
        )
        
        # Compute local field
        field = fftconvolve(self.events, kernel, mode='same')
        
        # Evaluate
        precision, recall, f1 = self._evaluate_classifier(field, threshold, "Moffat (nu=2)")
        
        self.assertGreater(f1, 0.55, "Moffat classifier failed to achieve expected F1-Score.")
        self.assertGreater(precision, 0.40, "Moffat classifier yielded too many false positives.")

    def test_generalized_moffat_nu_sweep(self):
        """
        Sweeps the exponent (nu) of the Generalized Moffat PSF.
        Demonstrates the 2D dimensional tension:
         - Low nu (e.g. 1.1) fails due to the r^2 area penalty dragging in background noise.
         - High nu (e.g. 10.0) fails as it approaches the Gaussian limit, losing its tail.
         - Optimal separation requires an intermediate power-law tail (nu=1.5 to 3.0).
        """
        print(f"\n{'='*60}\nTesting: GENERALIZED MOFFAT EXPONENT (NU) SWEEP\n{'='*60}")
        
        gamma = 1.5
        nus = [1.1, 1.5, 2.0, 3.0, 10.0]
        f1_scores = []
        batch_pixels = self.nx * self.ny
        
        for nu in nus:
            raw_kernel = 1.0 / (1.0 + (self.R_sq / gamma**2))**nu
            
            # Get analytical threshold based on background rate and total detector pixels
            kernel, threshold = self._calculate_production_threshold(
                raw_kernel, self.bg_rate, batch_pixels, target_fp=5.0
            )
            
            # Compute local field
            field = fftconvolve(self.events, kernel, mode='same')
            
            # Evaluate
            precision, recall, f1 = self._evaluate_classifier(field, threshold, f"Gen. Moffat (nu={nu:>4.1f})")
            f1_scores.append(f1)
            
        # Physics-based assertions:
        # 1. The Gaussian limit (nu=10.0) should underperform the optimal heavy tail
        self.assertLess(f1_scores[-1], max(f1_scores), "Gaussian limit unexpectedly beat the optimal heavy tail.")
        
        # 2. The nearly non-integrable Cauchy limit (nu=1.1) should suffer from the 2D area penalty
        self.assertLess(f1_scores[0], f1_scores[2], "nu=1.1 unexpectedly beat nu=2.0 despite severe area penalty.")

    def test_cgmy_tempered_stable_sparsifier(self):
        """Evaluates the Tempered Stable (CGMY Proxy) kernel."""
        print(f"\n{'='*60}\nTesting: TEMPERED STABLE (CGMY) SPARSIFIER\n{'='*60}")
        
        gamma = 1.5
        nu_heavy = 1.5   # Power-law core
        rc = 3.0         # Exponential cutoff
        
        raw_kernel = (1.0 / (1.0 + (self.R_sq / gamma**2))**nu_heavy) * np.exp(-self.R / rc)
        
        # Get analytical threshold
        batch_pixels = self.nx * self.ny
        kernel, threshold = self._calculate_production_threshold(
            raw_kernel, self.bg_rate, batch_pixels, target_fp=5.0
        )
        
        # Compute local field
        field = fftconvolve(self.events, kernel, mode='same')
        
        # Evaluate
        precision, recall, f1 = self._evaluate_classifier(field, threshold, "Tempered Stable")
        
        self.assertGreater(f1, 0.60, "CGMY classifier failed to achieve expected F1-Score.")
        self.assertGreater(precision, 0.45, "CGMY classifier yielded too many false positives.")

    def test_baseline_gaussian_failure(self):
        """Demonstrates that a baseline Gaussian kernel underperforms the heavy-tailed kernels."""
        print(f"\n{'='*60}\nTesting: BASELINE GAUSSIAN (Control)\n{'='*60}")
        
        sigma = 1.5
        raw_kernel = np.exp(-self.R_sq / (2 * sigma**2))
        kernel = raw_kernel / np.sum(raw_kernel)
        
        # 1. Numerical Validation of Method of Moments (MoM)
        # For a Gaussian PSF, Campbell's Theorem dictates the local field converges 
        # to a Normal distribution. We validate the Gamma MoM threshold against 
        # the exact analytical Normal distribution threshold.
        batch_pixels = self.nx * self.ny
        _, gamma_thresh = self._calculate_production_threshold(
            raw_kernel, self.bg_rate, batch_pixels, target_fp=5.0
        )
        
        mu = self.bg_rate
        sigma_field = np.sqrt(self.bg_rate * np.sum(kernel**2))
        
        # Normal percentile equivalent to target_fp / batch_pixels
        percentile = 1.0 - (5.0 / batch_pixels)
        normal_thresh = scipy.stats.norm.ppf(percentile, loc=mu, scale=sigma_field)
        
        error = abs(gamma_thresh - normal_thresh) / normal_thresh
        print(f"  [Validation] Gamma Thresh: {gamma_thresh:.4f} | Normal Thresh: {normal_thresh:.4f} | Error: {error:.2%}")
        
        # Normal distribution strictly underestimates the threshold for low-count 
        # Poisson processes due to skewness. The Gamma distribution correctly models 
        # this skewness, but they remain mathematically linked.
        self.assertLess(error, 0.35, "Gamma approximation deviates beyond expected skewness bounds from Normal.")
        
        # 2. Evaluate Classifier
        field = fftconvolve(self.events, kernel, mode='same')
        precision, recall, f1 = self._evaluate_classifier(field, gamma_thresh, "2D Gaussian")
        
        # The Gaussian kernel lacks the heavy tails to pull the signal out of the noise.
        # It should fail the task, resulting in a significantly lower F1 score compared 
        # to the heavy-tailed CGMY/Moffat kernels (which hit > 0.60).
        self.assertLess(f1, 0.55, "Gaussian classifier unexpectedly achieved high F1-Score.")

    def test_lambda_dependency_sweep(self):
        """
        Sweeps the background rate (lambda) to demonstrate how signal 
        recovery degrades as the noise variance increases relative to the signal.
        """
        print(f"\n{'='*60}\nTesting: LAMBDA (BACKGROUND RATE) DEPENDENCY SWEEP\n{'='*60}")
        
        gamma = 1.5
        nu_heavy = 1.5
        rc = 3.0
        raw_kernel = (1.0 / (1.0 + (self.R_sq / gamma**2))**nu_heavy) * np.exp(-self.R / rc)
        
        # Sweep lambda from extremely sparse to densely noisy
        bg_rates = [0.01, 0.05, 0.25, 0.5, 1.0, 2.0]
        recalls = []
        batch_pixels = self.nx * self.ny
        
        for bg_rate in bg_rates:
            events, truth_mask = self._generate_test_grid(bg_rate, signal_rate=4.0)
            kernel, threshold = self._calculate_production_threshold(
                raw_kernel, bg_rate, batch_pixels, target_fp=5.0
            )
            
            field = fftconvolve(events, kernel, mode='same')
            
            _, recall, _ = self._evaluate_classifier(field, threshold, f"Lambda = {bg_rate:4.2f}", custom_truth_mask=truth_mask)
            recalls.append(recall)
            
        # Assertions to validate the physics of the degradation:
        # At high noise variance, the threshold is forced upwards, physically 
        # shearing the edges off the Bragg peak and crashing the Recall.
        self.assertGreater(recalls[0], recalls[-1], "Recall failed to degrade at high noise levels.")
        self.assertGreater(recalls[0], 0.95, "Failed to recover signal core in extremely low background (Lambda=0.01).")
        self.assertLess(recalls[-1], 0.60, "Unrealistic signal retention in extremely high background (Lambda=2.0).")

    def test_batch_size_threshold_scaling(self):
        """
        Validates the production recipe: how the optimal threshold scales
        logarithmically with the spatial batch size (detector size) to strictly 
        control the False Discovery Rate (FDR).
        """
        print(f"\n{'='*60}\nTesting: BATCH SIZE THRESHOLD SCALING RECIPE\n{'='*60}")
        
        gamma = 1.5
        nu_heavy = 1.5
        rc = 3.0
        raw_kernel = (1.0 / (1.0 + (self.R_sq / gamma**2))**nu_heavy) * np.exp(-self.R / rc)
        
        bg_rate = 0.1
        
        # Simulate streaming batches of varying pixel counts 
        # (e.g. single ASIC vs full panel vs massive detector array)
        batch_sizes = [10_000, 100_000, 1_000_000, 10_000_000]
        target_false_positives = 1.0 # strictly 1 expected FP per batch
        
        thresholds = []
        for b_size in batch_sizes:
            _, thresh = self._calculate_production_threshold(
                raw_kernel, bg_rate, b_size, target_fp=target_false_positives
            )
            thresholds.append(thresh)
            print(f"  [Batch Size {b_size:10,d} px] Threshold to maintain 1.0 FP: {thresh:.4f}")
        
        # The threshold MUST increase as the batch size increases to suppress the 
        # higher absolute number of expected outliers in the heavy tail.
        for i in range(1, len(thresholds)):
            self.assertGreater(thresholds[i], thresholds[i-1], 
                               "Threshold failed to scale up with increased batch size.")

if __name__ == '__main__':
    unittest.main(verbosity=2)
