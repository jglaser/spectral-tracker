"""
Regression: LEARNED SPECTRUM WIRING — flat fails, learned succeeds
==================================================================
Demonstrates the full Route B handoff: a hidden incident spectrum is LEARNED
(sl.learn_spectrum) and wired into the tracker via assumed_spectrum, converging
where the flat top-hat default does not.

Discriminating regime (mirrors the proven test_wl_spectrum.py)
--------------------------------------------------------------
A SHARP log-normal spectrum (LAM0=2.5 Å, SHAPE_S=0.25), NO Lorentz, band
[0.5,10] Å, 5° seed. test_wl_spectrum already establishes that flat converges
poorly here (~4.5°, fails the 2° bar) while the matched shape converges (<2°),
because the top-hat mis-weights the resolution shells when the data concentrate
in a thin wavelength shell. The spectrum decides PRECISION at a capturable
seed — it does NOT enlarge the capture basin (a 25° seed stays trapped for any
spectrum; that is a global-search problem, see test_global_aliasing).

Why population counts, not the event-assignment diagnostic
----------------------------------------------------------
The spectrum is learned from PER-REFLECTION POPULATIONS (Multinomial counts),
which sl.learn_spectrum recovers near-exactly even under a 5° seed
(lam0/s to <1%, shape L2 ~ 7e-4; validated in sandbox). The diagnostic's
event-assignment path (tracker_diagnostics._spectrum_block) instead BROADENS
sharp spectra: with σ_ang = σ_q/|q| comparable to the gate radius, low-|q|
long-λ events spill out of the gate, inflating the fitted width (s: 0.25 → ~0.58
on this data). That is a documented limitation of the diagnostic ("width = upper
bound"); the diagnostic is for eyeballing the PEAK before wiring, not for
recovering a sharp width. This test isolates the estimator + wiring from that
angular bias by feeding accurate populations — the count-extraction quality is a
separate concern tracked against the diagnostic.

Pipeline under test
-------------------
    Multinomial populations  →  sl.learn_spectrum (log-normal)  →  phi callable
    →  tracker(assumed_spectrum=phi)  →  build_band_weights  →  w_l_j  →  <2°.
"""

import unittest
from collections import defaultdict

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from spectral_tracker.tracker import tracker
from spectral_tracker import spectrum_learning as sl
from tests.testing_utils import BaseTrackerTest


# ── hidden data spectrum: sharp log-normal (same as test_wl_spectrum) ──────────
LAM0, SHAPE_S = 2.5, 0.25


def _phi_lognormal(lam, lam0=LAM0, s=SHAPE_S):
    lam = np.asarray(lam, float)
    out = np.zeros_like(lam)
    m = lam > 0
    out[m] = np.exp(-0.5 * (np.log(lam[m] / lam0) / s) ** 2) / lam[m]
    return out


class TestLearnedSpectrumWiring(BaseTrackerTest):

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _lognormal_data(U_true, N=1_000_000, duration=5.0, seed=42):
        """Events from a sharp log-normal spectrum, NO Lorentz (flux=spectrum)."""
        B = np.diag([0.1, 0.1, 0.1])
        ki = np.array([0.0, 0.0, 1.0])
        hv = np.arange(-3, 4)
        hc, kc, lc = np.meshgrid(hv, hv, hv, indexing='ij')
        hkl = np.stack([hc.ravel(), kc.ravel(), lc.ravel()])
        hkl = hkl[:, ~((hkl[0] == 0) & (hkl[1] == 0) & (hkl[2] == 0))]
        q = B @ hkl
        qn = np.linalg.norm(q, axis=0)
        qh = q / qn
        lam = -(2.0 / qn) * (ki @ (U_true @ qh))
        valid = (lam > 0.5) & (lam < 10.0)
        qh_v, qn_v, lam_v = qh[:, valid], qn[valid], lam[valid]
        flux = _phi_lognormal(lam_v)                  # NO Lorentz
        p = flux / flux.sum()
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(lam_v), size=N, p=p)
        sa = (0.008 / qn_v[idx])[None, :]
        dirs = U_true @ qh_v[:, idx] + rng.normal(0.0, sa, (3, N))
        dirs /= np.linalg.norm(dirs, axis=0, keepdims=True)
        times = np.sort(rng.uniform(0.0, duration, N))
        return (dirs.T, times,
                np.ones(N, dtype=int), np.zeros(N, dtype=int), np.zeros(N, dtype=int))

    @staticmethod
    def _learn_from_populations(U_ref, N=1_000_000,
                                h_max=6, d_min=2.0, d_max=8.0, wl=(0.5, 10.0)):
        """
        Learn the log-normal from per-reflection Multinomial populations tagged
        with U_ref. Recovers the sharp shape near-exactly even at a 5° seed
        (validated: lam0/s < 1%, shape L2 ~ 7e-4), because the Lorentz-free
        geometry is self-consistent and the WLS is well-conditioned on singles.
        """
        ki = np.array([0.0, 0.0, 1.0])
        B = sl.reciprocal_B(10.0, 10.0, 10.0)
        hv = np.arange(-h_max, h_max + 1)
        H, K, L = np.meshgrid(hv, hv, hv, indexing='ij')
        hkl = np.stack([H.ravel(), K.ravel(), L.ravel()])
        hkl = hkl[:, ~((hkl[0] == 0) & (hkl[1] == 0) & (hkl[2] == 0))]
        q = B @ hkl
        qn = np.linalg.norm(q, axis=0)
        res = (qn > 1.0 / d_max) & (qn < 1.0 / d_min)
        hkl, q, qn = hkl[:, res], q[:, res], qn[res]

        lam_ref, _ = sl.bragg_wavelengths(q, U_ref, ki)
        in_band = np.isfinite(lam_ref) & (lam_ref > wl[0]) & (lam_ref < wl[1])

        raw = np.where(in_band, _phi_lognormal(lam_ref), 0.0)    # no Lorentz
        p = raw / raw.sum()
        counts = np.random.default_rng(13).multinomial(N, p).astype(float)

        # background floor via out-of-band monitor reflections (standard pipeline)
        reduced = sl._reduce_hkl(hkl)
        occ = defaultdict(int)
        for j in range(hkl.shape[1]):
            if in_band[j]:
                occ[tuple(reduced[:, j])] += 1
        monitor = np.array([(not in_band[j]) and occ[tuple(reduced[:, j])] == 0
                            for j in range(hkl.shape[1])])
        c_bg = float(np.median(counts[monitor])) if monitor.any() else 0.0
        counts = np.clip(counts - c_bg, 0.0, None)

        singles = sl.singles_mask(hkl, in_band)
        params, phi, info = sl.learn_spectrum(
            lam_ref, counts, np.ones_like(qn), geom=None, singles=singles,
            family="lognormal", lam_band=wl, min_count=5)
        return params, phi, info

    def _reset_h5(self, U_seed):
        with h5py.File(self.finder_file, 'w') as f:
            f['sample/a'] = f['sample/b'] = f['sample/c'] = 10.0
            f['sample/alpha'] = f['sample/beta'] = f['sample/gamma'] = 90.0
            f['sample/space_group'] = b'P 1'
            f['beam/ki_vec'] = np.array([0.0, 0.0, 1.0])
            f['sample/U'] = U_seed

    def _run(self, sim_data, U_seed, assumed_spectrum):
        self._reset_h5(U_seed)
        return tracker(
            finder_file=self.finder_file,
            event_batches=self.get_fake_batches(sim_data),
            L_max=8,
            d_min=2.0, d_max=8.0,
            wl_min_tracking=0.5, wl_max_tracking=10.0,
            assumed_spectrum=assumed_spectrum,
            lorentz_correction=False,        # data has no Lorentz
        )

    # ── test ─────────────────────────────────────────────────────────────────

    def test_flat_fails_learned_lognormal_succeeds(self):
        print(
            f"\n{'='*60}\n"
            "Regression: FLAT FAILS / LEARNED LOG-NORMAL SUCCEEDS\n"
            f"{'='*60}"
        )

        U_true = Rotation.from_euler('y', 45.0, degrees=True).as_matrix()
        U_seed = Rotation.from_euler('y', 40.0, degrees=True).as_matrix()  # 5° capturable

        sim_data = self._lognormal_data(U_true)

        # ── Part 1: flat top-hat — converges poorly on the thin λ-shell ────────
        U_flat = self._run(sim_data, U_seed, assumed_spectrum=None)
        err_flat = self._evaluate_cubic_symmetric_error(U_true, U_flat)
        print(f"  flat top-hat:        {err_flat:.2f}°  (expect > 2°)")
        self.assertGreater(
            err_flat, 2.0,
            f"Expected flat to converge poorly on the sharp log-normal "
            f"({err_flat:.2f}° < 2°). If flat now passes, sharpen SHAPE_S or "
            f"widen the band so the top-hat mis-weights more shells."
        )

        # ── Part 2: learn the log-normal from populations (5° seed) ────────────
        params, learned_phi, info = self._learn_from_populations(U_seed)
        l0_err = abs(params['lam0'] - LAM0) / LAM0 * 100
        s_err  = abs(params['s'] - SHAPE_S) / SHAPE_S * 100
        print(f"  learned: lam0={params['lam0']:.3f} (true {LAM0}, {l0_err:.1f}%)  "
              f"s={params['s']:.3f} (true {SHAPE_S}, {s_err:.1f}%)  "
              f"singles={info['n_singles_used']}")
        self.assertLess(l0_err, 5.0, f"learned lam0 off: {params['lam0']:.3f}")
        self.assertLess(s_err, 15.0, f"learned width off: {params['s']:.3f}")

        # ── Part 3: tracker with the learned spectrum ──────────────────────────
        U_lrn = self._run(sim_data, U_seed, assumed_spectrum=learned_phi)
        err_lrn = self._evaluate_cubic_symmetric_error(U_true, U_lrn)
        print(f"  learned log-normal:  {err_lrn:.2f}°  (expect < 2°)")
        print(f"  precision gain (flat − learned): {err_flat - err_lrn:+.2f}°")

        self.assertLess(
            err_lrn, err_flat,
            f"Learned spectrum must beat flat: {err_lrn:.2f}° vs {err_flat:.2f}°."
        )
        self.assertLess(
            err_lrn, 2.0,
            f"Tracker with learned spectrum failed to converge "
            f"({err_lrn:.2f}° >= 2°). Check assumed_spectrum → build_band_weights "
            f"→ w_l_j wiring."
        )


if __name__ == '__main__':
    unittest.main(verbosity=2)
