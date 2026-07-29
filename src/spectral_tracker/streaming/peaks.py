"""Peak finding directly from the detector event stream.

WHY THIS EXISTS

`EventStreamSparsifier` answers "is this pixel above background?", which is the
right question for thinning an event stream and the wrong one for producing
observations to index against. It convolves the per-panel count image with a
smoothing kernel and keeps every pixel whose field exceeds a Gamma quantile, so
what comes out is REGIONS. Measured on CG4D_1808 against the 562 peaks a subhkl
finder run reports for the same still:

    threshold + connected components   precision  2-6%   recall <= 22%
                                       centroid agreement 1.5-2.1 px
                                       connected blobs of ~113 px

and a global orientation search built on those cannot separate the true
orientation from a random one (margin 0.79-0.94x, i.e. worse than chance).
Sweeping the sparsifier's own knobs does not fix it: `target_fp` has no useful
range below ~1e-2, because `percentile = 1 - target_fp/n_pixels` saturates and
the Gamma quantile sends the threshold to where nothing at all survives.

The fix is not a better threshold, it is a different question. A peak is a
LOCAL MAXIMUM of the same field. One extra maximum-filter turns region
detection into peak detection:

    local maxima on the same field    precision  79%    centroid agreement 0.24 px

and the same global search then separates the true orientation cleanly (1.89x
on 112 peaks, better than the 1.27x the 562 subhkl finder peaks give, because
precision matters more than recall here -- every spurious peak is a direction a
wrong orientation gets to explain).

Recall is deliberately low at the recommended settings. Global search does not
need every reflection; it needs directions it can trust.

HOW FEW EVENTS THIS NEEDS

Running the whole chain -- events, peaks, 200k orientation grid, refine,
rescore -- with no prior orientation at all, on CG4D_1808:

    250,000 raw events (79 s of beam)    4.6 deg, 1.13x   not indexed
    500,000 raw events (159 s)           0.68 deg, 1.80x  INDEXED
      5,000,000 raw events (26 min)      0.71 deg, 1.84x  INDEXED
    272,902,295 raw events (24 h)        0.74 deg, 1.89x  INDEXED

So half a million events -- under three minutes of beam on this instrument --
is enough to index from nothing, and the remaining 272 million buy 0.06 deg of
accuracy and 0.09x of confidence.
"""
from __future__ import annotations

import numpy as np

from subhkl.config import beamlines
from subhkl.instrument.detector import Detector


def gaussian_kernel(sigma: float, size: int | None = None) -> np.ndarray:
    size = size or (int(6 * sigma) | 1)
    x = np.arange(size) - size // 2
    X, Y = np.meshgrid(x, x)
    k = np.exp(-(X ** 2 + Y ** 2) / (2.0 * sigma ** 2))
    return (k / k.sum()).astype(np.float32)


class EventPeakFinder:
    """Accumulate raw events into per-panel images and extract Bragg peaks.

    Parameters
    ----------
    sigma
        Smoothing width in pixels; match it to the spot size. 3-5 px covers the
        CG4D spots. Larger is more selective and localises better (centroid
        agreement 0.34 px at sigma=3, 0.24 px at sigma=5).
    n_sigma
        Threshold on the convolved field in robust standard deviations above
        its median (MAD-based, so a few bright panels do not move it). This is
        the precision/recall knob and, unlike the sparsifier's `target_fp`, it
        has a usable range: on CG4D_1808 precision runs 28% at n_sigma=8 to 79%
        at n_sigma=12 with sigma=5.

        The default of 6 is chosen for the event-starved end rather than the
        best precision, because that is where the knob actually matters.
        Separation of the global-search winner from the best different
        solution, CG4D_1808:

            raw events     n_sigma=5   n_sigma=6   n_sigma=8   n_sigma=12
              250,000       1.13x(no)   too few     too few     too few
              500,000       1.38x       1.80x       1.34x       too few
            1,000,000       1.78x       1.65x       1.95x       too few
            5,000,000       1.81x       1.84x       1.79x       1.75x
          272,902,295         --          --          --        1.89x

        Raise it when events are plentiful and precision is what you want;
        lower it when they are not.
    min_separation
        Peaks closer than this (pixels) are merged by the maximum filter.
        Defaults to 1.5 * sigma.
    max_peaks_per_bank
        Keep only the strongest this many per panel.
    """

    def __init__(self, instrument_name: str, sigma: float = 5.0,
                 n_sigma: float = 6.0, min_separation: int | None = None,
                 max_peaks_per_bank: int = 200):
        self.instrument_name = instrument_name
        self.sigma = float(sigma)
        self.n_sigma = float(n_sigma)
        self.min_separation = (int(min_separation) if min_separation is not None
                               else max(2, int(1.5 * sigma)))
        self.max_peaks_per_bank = int(max_peaks_per_bank)
        self.kernel = gaussian_kernel(sigma)

        self.shape = {}
        for bank_str, cfg in beamlines[instrument_name].items():
            if bank_str.isdigit():
                det = Detector(cfg)
                self.shape[int(bank_str)] = (det.n, det.m)
        self.images: dict[int, np.ndarray] = {}
        self.n_events = 0

    # -- accumulation ------------------------------------------------------

    def accumulate(self, banks, pixel_r, pixel_c):
        """Add one raw batch. Safe to call with the loader's raw batches."""
        banks = np.asarray(banks)
        pr = np.asarray(pixel_r)
        pc = np.asarray(pixel_c)
        self.n_events += len(banks)
        for b in np.unique(banks):
            b = int(b)
            if b not in self.shape:
                continue
            nx, ny = self.shape[b]
            img = self.images.get(b)
            if img is None:
                img = self.images[b] = np.zeros((nx, ny), np.float32)
            m = banks == b
            np.add.at(img, (np.clip(pr[m], 0, nx - 1).astype(np.intp),
                            np.clip(pc[m], 0, ny - 1).astype(np.intp)), 1.0)

    def consume(self, raw_stream, max_events: int | None = None):
        """Drain a loader raw-batch stream into the accumulator."""
        for t_b, b_b, pr_b, pc_b, end_idx in raw_stream:
            self.accumulate(b_b, pr_b, pc_b)
            if max_events is not None and end_idx >= max_events:
                break
        return self

    # -- detection ---------------------------------------------------------

    def find(self):
        """Return the peak list as a dict of parallel arrays.

        Keys: bank, pixel_r, pixel_c (sub-pixel centroids), intensity, n_pixels.
        """
        from scipy import ndimage

        bank, rr, cc, inten, npx = [], [], [], [], []
        half = max(2, int(round(1.5 * self.sigma)))
        for b, img in self.images.items():
            nx, ny = img.shape
            field = ndimage.convolve(img, self.kernel, mode="constant")
            peak_max = ndimage.maximum_filter(
                field, size=2 * self.min_separation + 1)
            med = np.median(field)
            # MAD rather than the standard deviation: the Bragg peaks are
            # themselves outliers of the field, and letting them inflate the
            # scale is what makes a plain sigma-clip threshold miss them.
            mad = 1.4826 * np.median(np.abs(field - med)) + 1e-9
            r0, c0 = np.nonzero((field >= peak_max)
                                & (field > med + self.n_sigma * mad))
            if len(r0) == 0:
                continue
            strength = field[r0, c0]
            if len(r0) > self.max_peaks_per_bank:
                keep = np.argsort(-strength)[:self.max_peaks_per_bank]
                r0, c0, strength = r0[keep], c0[keep], strength[keep]
            for r, c in zip(r0, c0):
                r1, r2 = max(0, r - half), min(nx, r + half + 1)
                c1, c2 = max(0, c - half), min(ny, c + half + 1)
                w = img[r1:r2, c1:c2]
                tot = float(w.sum())
                if tot <= 0:
                    continue
                gr, gc = np.mgrid[r1:r2, c1:c2]
                bank.append(b)
                rr.append(float((gr * w).sum() / tot))
                cc.append(float((gc * w).sum() / tot))
                inten.append(tot)
                npx.append(int((w > 0).sum()))
        return dict(bank=np.array(bank, np.int32),
                    pixel_r=np.array(rr), pixel_c=np.array(cc),
                    intensity=np.array(inten), n_pixels=np.array(npx, np.int32))


def peaks_to_sample_frame(loader, peaks, angles):
    """Sample-frame (q_hat, ki_hat) for a peak list, via the loader geometry.

    `angles` is the goniometer setting the peaks were accumulated at, shape
    (num_axes,). Reuses EventStreamLoader._project_events so the peaks go
    through exactly the same detector and goniometer path as the events.
    """
    n = len(peaks["bank"])
    ang = np.asarray(angles, float)
    q_lab, q_sample, ki_sample, _, _ = loader._project_events(
        np.zeros(n), peaks["bank"], peaks["pixel_r"], peaks["pixel_c"])
    q = np.asarray(q_sample, float)
    k = np.asarray(ki_sample, float)
    ok = ((np.linalg.norm(q, axis=1) > 1e-9)
          & (np.linalg.norm(k, axis=1) > 1e-9))
    q, k = q[ok], k[ok]
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    k /= np.linalg.norm(k, axis=1, keepdims=True)
    return q, k, ok
