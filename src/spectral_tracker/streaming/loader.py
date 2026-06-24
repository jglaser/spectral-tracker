import h5py
import numpy as np
import re
import multiprocessing
import concurrent.futures
from scipy.signal import fftconvolve
import scipy.stats

from subhkl.instrument.detector import Detector
from subhkl.config import beamlines, reduction_settings
from subhkl.instrument.goniometer import sample_to_lab, lab_to_sample

def _extract_raw_bank(args):
    """Parallel worker to parse a single detector bank into raw events only."""
    nexus_filename, key, instrument_name = args

    with h5py.File(nexus_filename, 'r') as f:
        match = re.match(r"bank(\d+)_events", key)
        if not match: return None
        bank_id = int(match.group(1))
        bank_str = str(bank_id)

        folder = f'/entry/{key}'
        if folder+'/event_id' not in f: return None

        event_id = f[folder+'/event_id'][:]
        event_index = f[folder+'/event_index'][:]
        event_time_offset = f[folder+'/event_time_offset'][:]
        event_time_zero = f[folder+'/event_time_zero'][:]

    if len(event_id) == 0: return None

    counts_per_pulse = np.diff(np.append(event_index, len(event_time_offset))).astype(int)
    absolute_time = np.repeat(event_time_zero, counts_per_pulse) + (event_time_offset * 1e-6)

    det_config = beamlines[instrument_name][bank_str]
    det = Detector(det_config)
    settings = reduction_settings.get(instrument_name, {})

    offset = det_config.get("offset", 0)
    local_id = event_id - offset

    if settings.get("YAxisIsFastVaryingIndex"):
        pixel_c = local_id // det.n
        pixel_r = local_id % det.n
    else:
        pixel_c = local_id % det.m
        pixel_r = local_id // det.m

    banks = np.full(len(absolute_time), bank_id, dtype=np.int16)

    return (
        absolute_time,
        banks,
        pixel_r.astype(np.int16),
        pixel_c.astype(np.int16)
    )


class EventStreamSparsifier:
    """
    Applies non-linear outlier detection to raw pixel events prior to 3D mapping.
    Uses the analytical Campbell framework to isolate Bragg peaks from the noise field.
    """
    def __init__(self, instrument_name, per_pixel_rate=5e-6, target_fp=1.0, gamma=1.5, nu=1.5, rc=3.0):
        self.instrument_name = instrument_name
        self.per_pixel_rate = per_pixel_rate
        self.target_fp = target_fp
        
        # Precompute the Tempered Stable (CGMY Proxy) spatial kernel
        kx = np.linspace(-20, 20, 128)
        ky = np.linspace(-20, 20, 128)
        KX, KY = np.meshgrid(kx, ky)
        R_sq = KX**2 + KY**2
        R = np.sqrt(R_sq)
        
        raw_kernel = (1.0 / (1.0 + (R_sq / gamma**2))**nu) * np.exp(-R / rc)
        self.kernel = raw_kernel / np.sum(raw_kernel)
        self.kernel_sq_sum = np.sum(self.kernel**2)
        
        # Map out grid sizes for configured banks
        self.nx_ny = {}
        for bank_str, det_config in beamlines[instrument_name].items():
            if bank_str.isdigit():
                b_id = int(bank_str)
                det = Detector(det_config)
                self.nx_ny[b_id] = (det.n, det.m)
                
    def filter_batch(self, banks, pr, pc, batch_size):
        keep_mask = np.zeros(len(banks), dtype=bool)
        unique_banks = np.unique(banks)
        
        # Calculate dynamic lambda based on the event batch throughput
        lambda_bg = self.per_pixel_rate * batch_size
        if lambda_bg < 1e-9:
            return np.ones(len(banks), dtype=bool)
            
        kappa_1 = lambda_bg * 1.0  
        kappa_2 = lambda_bg * self.kernel_sq_sum
        theta = kappa_2 / kappa_1
        k = kappa_1 / theta
        
        for b_id in unique_banks:
            b_mask = (banks == b_id)
            if b_id not in self.nx_ny:
                keep_mask[b_mask] = True
                continue
            
            nx, ny = self.nx_ny[b_id]
            pr_b = pr[b_mask]
            pc_b = pc[b_mask]
            if len(pr_b) == 0:
                continue
                
            batch_pixels = nx * ny
            percentile = 1.0 - (self.target_fp / batch_pixels)
            percentile = min(max(percentile, 0.0), 1.0 - 1e-15)
            
            # Map Moments to Gamma for closed-form thresholding
            threshold = scipy.stats.gamma.ppf(percentile, a=k, scale=theta)
            
            # Project events to 2D field and convolve
            events, _, _ = np.histogram2d(pr_b, pc_b, bins=(nx, ny), range=[[0, nx], [0, ny]])
            field = fftconvolve(events, self.kernel, mode='same')
            
            # Non-linear detection
            detected_mask = field > threshold
            
            pr_safe = np.clip(pr_b, 0, nx - 1)
            pc_safe = np.clip(pc_b, 0, ny - 1)
            keep_mask[b_mask] = detected_mask[pr_safe, pc_safe]
            
        return keep_mask


class EventStreamLoader:
    def __init__(
        self,
        event_nexus_filename: str,
        instrument_name: str,
        ki_vec: np.ndarray,
        sample_offset: np.ndarray,
        gonio_axes=None,
        gonio_names=None,
        gonio_offsets=None,
    ):
        self.event_nexus_filename = event_nexus_filename
        self.instrument_name = instrument_name
        self.ki_vec = ki_vec
        self.sample_offset = sample_offset
        self.gonio_axes = gonio_axes
        self.gonio_names = gonio_names
        self.gonio_offsets = gonio_offsets
        self.gonio_continuous_logs = None
        
        print(f"  > Initializing Event Stream Loader from: {event_nexus_filename}")
        self._load_and_sort_events()

    def _load_and_sort_events(self):
        gonio_continuous_logs = []
        with h5py.File(self.event_nexus_filename, 'r') as f:
            keys = [k for k in f['entry'].keys() if k.endswith('_events')]
            
            if self.gonio_names is not None and self.gonio_axes is not None:
                for name in self.gonio_names:
                    try:
                        log_path = f'entry/DASlogs/{name}'
                        if log_path in f and 'time' in f[log_path] and 'value' in f[log_path]:
                            t = f[f'{log_path}/time'][:]
                            v = f[f'{log_path}/value'][:]
                            gonio_continuous_logs.append((t, v))
                        else:
                            gonio_continuous_logs.append((np.array([0.0]), np.array([0.0])))
                    except Exception as e:
                        gonio_continuous_logs.append((np.array([0.0]), np.array([0.0])))
            else:
                gonio_continuous_logs = None
                
        self.gonio_continuous_logs = gonio_continuous_logs

        args_list = [
            (self.event_nexus_filename, k, self.instrument_name)
            for k in keys
        ]

        all_times, all_banks, all_pixels_r, all_pixels_c = [], [], [], []
        
        print(f"  > Extracting {len(keys)} detector banks via Multiprocessing...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            for result in executor.map(_extract_raw_bank, args_list):
                if result is not None:
                    t, b, pr, pc = result
                    all_times.append(t)
                    all_banks.append(b)
                    all_pixels_r.append(pr)
                    all_pixels_c.append(pc)

        if not all_times:
            self.total_events = 0
            return

        all_times = np.concatenate(all_times)
        all_banks = np.concatenate(all_banks)
        all_pixels_r = np.concatenate(all_pixels_r)
        all_pixels_c = np.concatenate(all_pixels_c)

        print("  > Performing global chronological sort...")
        
        sort_idx = np.argsort(all_times, kind='stable')
        
        def apply_sort(arr):
            return arr[sort_idx]

        arrays_to_sort = [all_times, all_banks, all_pixels_r, all_pixels_c]

        print("  > Applying sorted indices to arrays...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(arrays_to_sort)) as executor:
            (
                self.all_times, 
                self.all_banks, 
                self.all_pixels_r, 
                self.all_pixels_c
            ) = list(executor.map(apply_sort, arrays_to_sort))

        self.total_events = len(self.all_times)
        print(f"  > EventStreamLoader Ready. Cached {self.total_events:,} raw events.")

    def _project_events(self, absolute_time, banks, pixel_r, pixel_c):
        """Lazily executed 3D transformations for events that survive the filter."""
        num_events = len(absolute_time)
        num_axes = len(self.gonio_axes) if self.gonio_axes is not None else 1
        
        xyz = np.zeros((num_events, 3), dtype=np.float32)
        unique_banks = np.unique(banks)

        for b_id in unique_banks:
            mask = banks == b_id
            bank_str = str(b_id)
            if bank_str in beamlines[self.instrument_name]:
                det_config = beamlines[self.instrument_name][bank_str]
                det = Detector(det_config)
                xyz[mask] = det.pixel_to_lab(pixel_r[mask], pixel_c[mask])

        if self.gonio_axes is not None and self.gonio_continuous_logs is not None:
            interpolated_angles = np.zeros((num_axes, num_events), dtype=np.float32)
            for i in range(num_axes):
                g_times, g_vals = self.gonio_continuous_logs[i]
                if len(g_times) <= 1:
                    interpolated_angles[i, :] = g_vals[0] if len(g_vals) > 0 else 0.0
                else:
                    interpolated_angles[i, :] = np.interp(absolute_time, g_times, g_vals)

            s_lab_dynamic = sample_to_lab(
                np.zeros((num_events, 3)), 
                self.gonio_axes, 
                interpolated_angles, 
                self.sample_offset, 
                zero_offsets=self.gonio_offsets
            )
            kf_lab = xyz - s_lab_dynamic
        else:
            interpolated_angles = np.zeros((num_axes, num_events), dtype=np.float32)
            if self.sample_offset is not None:
                s_lab_static = np.atleast_2d(self.sample_offset)[-1][:3]
            else:
                s_lab_static = np.zeros(3)
            s_lab_dynamic = np.tile(s_lab_static, (num_events, 1)).astype(np.float32)
            kf_lab = xyz - s_lab_dynamic

        kf_norm = np.sqrt(np.sum(kf_lab**2, axis=1, keepdims=True))
        kf_lab /= np.where(kf_norm == 0, 1.0, kf_norm)
        q_lab = kf_lab - self.ki_vec[None, :]

        if self.gonio_axes is not None:
            q_sample = lab_to_sample(
                q_lab, self.gonio_axes, interpolated_angles, self.sample_offset, self.gonio_offsets, is_vector=True
            )
            ki_sample = lab_to_sample(
                np.tile(self.ki_vec, (num_events, 1)), self.gonio_axes, interpolated_angles, self.sample_offset, self.gonio_offsets, is_vector=True
            )
        else:
            q_sample = q_lab
            ki_sample = np.tile(self.ki_vec, (num_events, 1))
            
        return q_lab, q_sample.astype(np.float32), ki_sample.astype(np.float32), interpolated_angles.T.astype(np.float32), s_lab_dynamic

    def get_batches(self, batch_size_events: int = 10000, min_batch_size: int = 1, use_sparsifier: bool = False, per_pixel_rate: float = 5e-6):
        if self.total_events == 0:
            return

        if use_sparsifier:
            sparsifier = EventStreamSparsifier(self.instrument_name, per_pixel_rate=per_pixel_rate)
        else:
            sparsifier = None

        pack_times, pack_banks, pack_pr, pack_pc = [], [], [], []
        packed_count = 0
        read_chunk_size = batch_size_events

        for start_idx in range(0, self.total_events, read_chunk_size):
            end_idx = min(start_idx + read_chunk_size, self.total_events)
            
            t_b = self.all_times[start_idx:end_idx]
            b_b = self.all_banks[start_idx:end_idx]
            pr_b = self.all_pixels_r[start_idx:end_idx]
            pc_b = self.all_pixels_c[start_idx:end_idx]
            
            n_read = len(t_b)

            if sparsifier:
                keep_mask = sparsifier.filter_batch(b_b, pr_b, pc_b, batch_size=n_read)
                pack_times.append(t_b[keep_mask])
                pack_banks.append(b_b[keep_mask])
                pack_pr.append(pr_b[keep_mask])
                pack_pc.append(pc_b[keep_mask])
                packed_count += keep_mask.sum()
            else:
                pack_times.append(t_b)
                pack_banks.append(b_b)
                pack_pr.append(pr_b)
                pack_pc.append(pc_b)
                packed_count += n_read

            # Emit new packed batches
            while packed_count >= batch_size_events:
                merged_t = np.concatenate(pack_times)
                merged_b = np.concatenate(pack_banks)
                merged_pr = np.concatenate(pack_pr)
                merged_pc = np.concatenate(pack_pc)

                yield_t = merged_t[:batch_size_events]
                yield_b = merged_b[:batch_size_events]
                yield_pr = merged_pr[:batch_size_events]
                yield_pc = merged_pc[:batch_size_events]

                # Project the surviving slice lazily
                q_lab, q_sample, ki_sample, angles, s_lab = self._project_events(
                    yield_t, yield_b, yield_pr, yield_pc
                )
                yield (q_sample, yield_t, yield_b, yield_pr, yield_pc, angles, s_lab, ki_sample, end_idx)

                # Keep the remainder
                pack_times = [merged_t[batch_size_events:]]
                pack_banks = [merged_b[batch_size_events:]]
                pack_pr = [merged_pr[batch_size_events:]]
                pack_pc = [merged_pc[batch_size_events:]]
                packed_count = len(pack_times[0])

        # Yield any remaining packed events that satisfy min_batch_size
        if packed_count >= min_batch_size:
            yield_t = np.concatenate(pack_times)
            yield_b = np.concatenate(pack_banks)
            yield_pr = np.concatenate(pack_pr)
            yield_pc = np.concatenate(pack_pc)
            
            q_lab, q_sample, ki_sample, angles, s_lab = self._project_events(
                yield_t, yield_b, yield_pr, yield_pc
            )
            yield (q_sample, yield_t, yield_b, yield_pr, yield_pc, angles, s_lab, ki_sample, self.total_events)
