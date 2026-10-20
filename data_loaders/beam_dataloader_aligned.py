"""
Optimized Beam Dataset Dataloader for XJTU-DV-Beam
Performance improvements:
- Vectorized event processing using NumPy operations
- Memory-mapped file access for large files
- Parallel data loading with multiprocessing
- Optimized caching with compression

Supported event representations:
- 'frame': 2D frame with pos/neg channels (2, H, W)
- 'raw': raw event dict with x, y, t, p
- 'event_rate': ROI-based event rate timeseries (C, 2048)
- 'image_sequence': sequence of 2D images (n_frames, 2, H, W) for CNN3D/CNN-LSTM
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
from tqdm import tqdm
import sys
import pickle
import lz4.frame
from concurrent.futures import ThreadPoolExecutor
import warnings

# Add parent directory to path to import PSEELoader
sys.path.append(str(Path(__file__).parent.parent))
from utils.io.psee_loader import PSEELoader


class BeamDataset(Dataset):
    """
    Dataset class for XJTU-DV-Beam data.

    Loads pairs of:
    - Dynamic Vision (DV) data: Event-based camera data in .dat format
    - Laser Doppler Vibrometer (LDV) data: Vibration signals in .txt format

    Performance optimizations:
    - Vectorized operations for event processing
    - Efficient sparse representations
    - Optional preprocessing and caching
    - Parallel data loading
    """

    def __init__(
        self,
        data_root: str,
        event_duration: float = 4.8e6,  # Duration in microseconds
        event_window: Tuple[float, float] = (3.1e6, 4.8e6),  # Valid event window in microseconds
        image_size: Tuple[int, int] = (720, 1280),  # Height, Width of event camera
        transform=None,
        target_transform=None,
        mode: str = 'train',  # 'train', 'val', 'test', or 'all'
        max_vibration_samples: Optional[int] = None,  # Limit vibration samples for consistency
        event_representation: str = 'frame',  # 'frame', 'raw', 'event_rate', 'image_sequence' (voxel/sparse removed)
        # Image sequence parameters (for CNN/ViT models)
        image_seq_height: int = 720,  # Height for image_sequence representation (default: original camera resolution)
        image_seq_width: int = 1280,  # Width for image_sequence representation (default: original camera resolution)
        image_seq_n_frames: Optional[int] = None,  # Number of frames (defaults to n_event_rate)
        cache_events: bool = False,  # Whether to cache processed events in memory
        cache_dir: Optional[str] = None,  # Directory for disk caching
        use_sparse: bool = True,  # Use sparse matrices for event frames
        preprocess_all: bool = False,  # Preprocess all data at initialization
        num_workers_preprocess: int = 4,  # Number of workers for preprocessing
        # Alignment options
        align: bool = True,
        vib_duration_s: float = 8.0,
        vib_fs_hz: Optional[float] = 256.0,          # if LDV txt doesn't include time column in seconds, set this
        ev_bin_ms: float = 5.0,                       # event-rate binning (ms)
        env_smooth_ms: float = 50.0,                  # smoothing for correlation (ms)
        max_shift_s: float = 3.0,                     # search window for offset (seconds)
        offset_cache: bool = True,                    # store offsets to avoid recompute
        offset_cache_name: str = "offsets.pkl",
        rois: Optional[List[Tuple[int,int,int,int]]] = None,
        roi_grid: Optional[Tuple[int,int]] = None,
        use_polarity_event_rate: bool = True,
        n_event_rate: int = 2048,
        n_vibration_samples: int = 2048,  # Number of vibration samples (can downsample for Speck chip)
        roi_cache_tag: str = "",
        # Temporal augmentation (training only) - circular shift to prevent learning absolute time
        circular_shift_aug: bool = False,            # enable random circular shift augmentation
        # Noise filtering
        bandpass_filter: Optional[Tuple[float, float]] = None,  # (low_hz, high_hz) bandpass filter for vibration
        is_spiking: bool = False,
        # Windowing options
        use_windowing: bool = False,                  # enable windowing
        window_size: int = 8,                         # window size (must divide n_event_rate evenly)
        shuffle_windows: bool = False,                # shuffle windows (training only)
    ):
        """
        Initialize the Optimized Beam Dataset.

        The dataset uses a fixed stratified split based on trials:
        - Train: trials 1, 2, 3 (18 samples, 60%)
        - Val: trial 4 (6 samples, 20%)
        - Test: trial 5 (6 samples, 20%)

        This ensures each split has all 6 experimental configurations
        (2 illumination conditions × 3 camera settings).

        Args:
            data_root: Root directory containing DV and LDV folders
            event_duration: Duration of events to load in microseconds
            event_window: Time window to extract valid events (start, end) in microseconds
            image_size: Size of the event camera (height, width)
            transform: Optional transform to apply to event data
            target_transform: Optional transform to apply to vibration data
            mode: Dataset mode ('train', 'val', 'test', or 'all' for entire dataset)
            max_vibration_samples: Maximum number of vibration samples to use
            event_representation: How to represent events ('frame', 'raw', 'event_rate', 'image_sequence')
            cache_events: Whether to cache processed events in memory
            cache_dir: Directory for disk-based caching (compressed)
            use_sparse: Use sparse matrix representation for memory efficiency
            preprocess_all: Preprocess all data at initialization
            num_workers_preprocess: Number of parallel workers for preprocessing
        """
        self.data_root = Path(data_root)
        self.dv_dir = self.data_root / "DV"
        self.ldv_dir = self.data_root / "LDV"

        # Verify directories exist
        assert self.dv_dir.exists(), f"DV directory not found: {self.dv_dir}"
        assert self.ldv_dir.exists(), f"LDV directory not found: {self.ldv_dir}"

        self.event_duration = event_duration
        self.event_window = event_window
        self.image_size = image_size
        self.rois = rois
        self.roi_grid = roi_grid
        self.use_polarity_event_rate = use_polarity_event_rate
        self.n_event_rate = n_event_rate
        self.n_vibration_samples = n_vibration_samples
        self.roi_cache_tag = roi_cache_tag

        # Build ROI list (resolved) once
        self.rois_resolved = self._resolve_rois(self.rois, self.roi_grid)

        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode
        self.max_vibration_samples = max_vibration_samples
        self.event_representation = event_representation
        self.image_seq_height = image_seq_height
        self.image_seq_width = image_seq_width
        self.image_seq_n_frames = image_seq_n_frames
        self.cache_events = cache_events
        self.use_sparse = use_sparse
        self.event_cache = {} if cache_events else None

        self.align = align
        self.vib_duration_s = vib_duration_s
        self.vib_fs_hz = vib_fs_hz
        self.ev_bin_ms = ev_bin_ms
        self.env_smooth_ms = env_smooth_ms
        self.max_shift_s = max_shift_s
        self.offset_cache = offset_cache
        self.circular_shift_aug = circular_shift_aug
        self.bandpass_filter = bandpass_filter
        self.is_spiking = is_spiking
        self.use_windowing = use_windowing
        self.window_size = window_size
        self.shuffle_windows = shuffle_windows

        # Validate windowing parameters
        if self.use_windowing:
            if self.n_event_rate % self.window_size != 0:
                raise ValueError(f"window_size ({self.window_size}) must divide n_event_rate ({self.n_event_rate}) evenly")

        # Global normalization for event rate
        # Will be computed on first access if needed
        self._global_event_max = None
        self._global_max_computed = False

        # Setup disk caching
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None


        self.offsets = {}
        if self.offset_cache and self.cache_dir:
            self.offset_cache_path = Path(self.cache_dir) / offset_cache_name
            if self.offset_cache_path.exists():
                try:
                    with open(self.offset_cache_path, "rb") as f:
                        self.offsets = pickle.load(f)
                except Exception:
                    self.offsets = {}
        else:
            self.offset_cache_path = None


        # Get all data file pairs
        self.data_pairs = self._get_data_pairs()

        # Split data into train/val/test
        self._split_data()

        # Preprocess all data if requested
        if preprocess_all:
            self._preprocess_all_data(num_workers_preprocess)

    def _resolve_rois(
        self,
        rois: Optional[List[Tuple[int, int, int, int]]],
        roi_grid: Optional[Tuple[int, int]],
    ) -> List[Tuple[int, int, int, int]]:
        """
        Returns a list of (x0,y0,x1,y1) ROIs.
        Priority:
        - if rois provided: use them
        - elif roi_grid provided: generate grid ROIs
        - else: full-frame ROI
        """
        H, W = self.image_size  # (H,W) but stored as (720,1280)
        # Note: self.image_size = (H,W) in your code
        H = int(H)
        W = int(W)

        if rois is not None and len(rois) > 0:
            out = []
            for (x0, y0, x1, y1) in rois:
                x0 = int(max(0, min(W, x0)))
                x1 = int(max(0, min(W, x1)))
                y0 = int(max(0, min(H, y0)))
                y1 = int(max(0, min(H, y1)))
                if x1 > x0 and y1 > y0:
                    out.append((x0, y0, x1, y1))
            if len(out) == 0:
                return [(0, 0, W, H)]
            return out

        if roi_grid is not None:
            rows, cols = roi_grid
            rows = int(rows)
            cols = int(cols)
            rows = max(rows, 1)
            cols = max(cols, 1)

            out = []
            for r in range(rows):
                y0 = int(round(r * H / rows))
                y1 = int(round((r + 1) * H / rows))
                for c in range(cols):
                    x0 = int(round(c * W / cols))
                    x1 = int(round((c + 1) * W / cols))
                    if x1 > x0 and y1 > y0:
                        out.append((x0, y0, x1, y1))
            return out if len(out) > 0 else [(0, 0, W, H)]

        return [(0, 0, W, H)]


    def _roi_signature(self) -> str:
        """
        A stable signature string for cache keys.
        Keeps caching correct across different ROI configs.
        """
        parts = []
        parts.append(f"pol{int(self.use_polarity_event_rate)}")
        parts.append(f"n{int(self.n_event_rate)}")
        if self.roi_grid is not None:
            parts.append(f"grid{int(self.roi_grid[0])}x{int(self.roi_grid[1])}")
        else:
            parts.append(f"rois{len(self.rois_resolved)}")
        if getattr(self, "roi_cache_tag", ""):
            parts.append(f"tag{self.roi_cache_tag}")
        return "_".join(parts)


    def _moving_average(self, x: np.ndarray, win: int) -> np.ndarray:
        if win <= 1:
            return x
        win = min(win, len(x))
        kernel = np.ones(win, dtype=np.float32) / win
        return np.convolve(x, kernel, mode="same").astype(np.float32)

    def _apply_bandpass_filter(self, signal: np.ndarray, fs: float, low_hz: float, high_hz: float) -> np.ndarray:
        """Apply a bandpass filter to remove noise outside the frequency range of interest."""
        from scipy.signal import butter, filtfilt

        # Ensure valid frequency range
        nyq = fs / 2.0
        low = max(low_hz / nyq, 0.001)
        high = min(high_hz / nyq, 0.999)

        if low >= high:
            return signal

        # Design 4th order Butterworth bandpass filter
        b, a = butter(4, [low, high], btype='band')

        # Apply zero-phase filtering
        try:
            filtered = filtfilt(b, a, signal, padlen=min(len(signal) - 1, 3 * max(len(a), len(b))))
            return filtered.astype(np.float32)
        except ValueError:
            # If signal too short, return original
            return signal

    def _apply_circular_shift(self, events: np.ndarray, vibration: np.ndarray, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply random circular shift to both events and vibration.
        This prevents the model from learning absolute time positions.
        Both signals are shifted by the same amount to maintain alignment.

        np.roll(arr, shift) moves elements to higher indices (rightward/later in time).
        So we shift event timestamps in the same direction (add shift_us).
        """
        if n_samples <= 1:
            return events, vibration

        # Random shift amount (in samples for vibration, will be converted for events)
        shift = np.random.randint(0, n_samples)

        # Circular shift vibration: elements move to higher indices
        vibration_shifted = np.roll(vibration, shift)

        # Shift events: convert sample shift to microseconds
        # Events are in aligned time [0, vib_duration_s * 1e6] microseconds
        duration_us = self.vib_duration_s * 1e6
        shift_us = (shift / n_samples) * duration_us

        # Circular shift events in the SAME direction as vibration
        # np.roll moves to higher indices, so event times should also increase
        events_shifted = events.copy()
        t_shifted = (events_shifted["t"].astype(np.float64) + shift_us) % duration_us
        events_shifted["t"] = t_shifted.astype(np.int64)

        return events_shifted, vibration_shifted

    def _to_seconds_events(self, t: np.ndarray) -> np.ndarray:
        # Heuristic: detect us vs ns
        tmax = float(np.max(t)) if len(t) else 0.0
        if tmax > 1e12:      # ns range
            return (t.astype(np.float64) * 1e-9).astype(np.float32)
        elif tmax > 1e9:     # could still be ns
            return (t.astype(np.float64) * 1e-9).astype(np.float32)
        else:                # assume microseconds
            return (t.astype(np.float64) * 1e-6).astype(np.float32)

    def _get_vibration_time_and_signal(self, vib_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        vib_data can be:
          - Nx2 [time, amplitude]  (preferred)
          - Nx1 [amplitude]        (fallback uses vib_fs_hz)
        Returns:
          t_v (seconds), y_v (amplitude), fs (Hz)
        """
        if vib_data.ndim == 2 and vib_data.shape[1] >= 2:
            t = vib_data[:, 0].astype(np.float32)
            y = vib_data[:, 1].astype(np.float32)
            # infer fs from time if possible
            dt = np.diff(t)
            dt = dt[np.isfinite(dt) & (dt > 0)]
            fs = float(1.0 / np.median(dt)) if len(dt) else float(self.vib_fs_hz or 256.0)
            return t, y, fs

        # fallback: assume uniform sampling
        y = vib_data.reshape(-1).astype(np.float32)
        fs = float(self.vib_fs_hz or 256.0)
        t = (np.arange(len(y), dtype=np.float32) / fs).astype(np.float32)
        return t, y, fs

    def _compute_event_rate(self, t_sec: np.ndarray, bin_ms: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        t_sec: event timestamps in seconds for ROI-filtered events (already)
        Returns:
          t_centers (seconds), rate (events/sec)
        """
        if len(t_sec) == 0:
            return np.zeros(1, np.float32), np.zeros(1, np.float32)

        bin_w = bin_ms / 1000.0
        t0 = float(t_sec.min())
        t1 = float(t_sec.max())
        # ensure at least 2 bins
        nbins = max(int(np.ceil((t1 - t0) / bin_w)), 2)
        edges = (t0 + np.arange(nbins + 1, dtype=np.float32) * bin_w).astype(np.float32)

        counts, _ = np.histogram(t_sec, bins=edges)
        counts = counts.astype(np.float32)
        rate = counts / bin_w
        centers = (edges[:-1] + edges[1:]) * 0.5
        return centers.astype(np.float32), rate.astype(np.float32)

    def _estimate_offset_peak_alignment(
        self,
        t_ev: np.ndarray,
        ev_rate: np.ndarray,
        t_v: np.ndarray,
        vib: np.ndarray,
        smooth_ms: float,
    ) -> Tuple[float, float]:
        """
        Align signals by matching the highest peak in vibration with the highest peak in event rate.

        Returns:
          offset_s such that aligned_event_time = event_time - offset_s matches vibration time.
          score (confidence, always 1.0 for peak alignment)
        """
        # Smooth both signals to get cleaner peaks
        from scipy.ndimage import gaussian_filter1d

        # Convert smooth_ms to samples for each signal
        dt_ev = np.mean(np.diff(t_ev)) if len(t_ev) > 1 else 0.001
        dt_v = np.mean(np.diff(t_v)) if len(t_v) > 1 else 0.001

        sigma_ev = (smooth_ms / 1000.0) / dt_ev
        sigma_v = (smooth_ms / 1000.0) / dt_v

        # Use absolute value for vibration (envelope)
        vib_smooth = gaussian_filter1d(np.abs(vib).astype(np.float32), sigma=max(sigma_v, 1.0))
        ev_smooth = gaussian_filter1d(ev_rate.astype(np.float32), sigma=max(sigma_ev, 1.0))

        # Find the highest peak in each signal
        vib_peak_idx = np.argmax(vib_smooth)
        ev_peak_idx = np.argmax(ev_smooth)

        # Get the times of these peaks
        t_vib_peak = float(t_v[vib_peak_idx])
        t_ev_peak = float(t_ev[ev_peak_idx])

        # Calculate offset: how much to shift event time to align with vibration time
        # aligned_event_time = event_time - offset
        # We want: t_ev_peak - offset = t_vib_peak
        # Therefore: offset = t_ev_peak - t_vib_peak
        offset_s = t_ev_peak - t_vib_peak

        # Calculate confidence score based on peak prominence
        vib_peak_height = vib_smooth[vib_peak_idx]
        ev_peak_height = ev_smooth[ev_peak_idx]

        vib_mean = np.mean(vib_smooth)
        ev_mean = np.mean(ev_smooth)

        # Score based on how prominent the peaks are (SNR-like measure)
        vib_prominence = vib_peak_height / (vib_mean + 1e-6)
        ev_prominence = ev_peak_height / (ev_mean + 1e-6)

        # Geometric mean of prominences, normalized
        score = min(np.sqrt(vib_prominence * ev_prominence) / 10.0, 1.0)

        return float(offset_s), float(score)

    def _estimate_offset_onset_detection(
        self,
        t_ev: np.ndarray,
        ev_rate: np.ndarray,
        t_v: np.ndarray,
        vib: np.ndarray,
        threshold_percentile: float = 80,
    ) -> Tuple[float, float]:
        """
        Align signals by detecting when activity first rises above background noise.

        Finds the first time each signal exceeds a threshold (e.g., 80th percentile),
        then aligns those onset times.

        Returns:
          offset_s such that aligned_event_time = event_time - offset_s matches vibration time.
          score (confidence based on how clear the onset is)
        """
        # Find threshold for each signal (above background noise)
        ev_threshold = np.percentile(ev_rate, threshold_percentile)
        vib_threshold = np.percentile(np.abs(vib), threshold_percentile)

        # Find first time signal exceeds threshold
        ev_above = np.where(ev_rate > ev_threshold)[0]
        vib_above = np.where(np.abs(vib) > vib_threshold)[0]

        if len(ev_above) == 0 or len(vib_above) == 0:
            # Fall back to peak alignment
            return self._estimate_offset_peak_alignment(t_ev, ev_rate, t_v, vib, smooth_ms=50.0)

        # Get onset times
        ev_onset_idx = ev_above[0]
        vib_onset_idx = vib_above[0]

        t_ev_onset = float(t_ev[ev_onset_idx])
        t_vib_onset = float(t_v[vib_onset_idx])

        # Offset: how much to shift events to align with vibration
        offset_s = t_ev_onset - t_vib_onset

        # Score based on how sharp the onset is
        ev_max = ev_rate.max()
        vib_max = np.abs(vib).max()
        score = min((ev_max / (ev_threshold + 1e-8)) * (vib_max / (vib_threshold + 1e-8)) / 100, 1.0)

        return float(offset_s), float(score)

    def _events_to_eventrate_2048(
        self,
        events_aligned: np.ndarray,
        n: Optional[int] = None,
        duration_s: Optional[float] = None,
        use_polarity: Optional[bool] = None,
        rois: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> np.ndarray:
        """
        ROI-aware event-rate representation.

        events_aligned:
        - structured array with fields x,y,t,p
        - t is aligned and stored in microseconds (int64) spanning [0, duration_s*1e6]

        Returns:
        X shape (C, n):
            - if use_polarity: C = 2 * num_rois (pos, neg per ROI)
            - else:           C = 1 * num_rois (total per ROI)
        """
        if duration_s is None:
            duration_s = float(self.vib_duration_s)
        if n is None:
            n = int(self.n_event_rate)
        if use_polarity is None:
            use_polarity = bool(self.use_polarity_event_rate)
        if rois is None:
            rois = self.rois_resolved

        edges_us = np.linspace(0.0, duration_s * 1e6, n + 1, dtype=np.float64)
        bin_w_s = duration_s / n

        C = (2 if use_polarity else 1) * len(rois)
        if len(events_aligned) == 0:
            return np.zeros((C, n), dtype=np.float32)

        # Pull arrays
        x = events_aligned["x"].astype(np.int32)
        y = events_aligned["y"].astype(np.int32)
        t_us = events_aligned["t"].astype(np.float64)
        if use_polarity:
            p = events_aligned["p"]

        chans = []
        for (x0, y0, x1, y1) in rois:
            m = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)

            if not np.any(m):
                if use_polarity:
                    chans.append(np.zeros(n, dtype=np.float32))
                    chans.append(np.zeros(n, dtype=np.float32))
                else:
                    chans.append(np.zeros(n, dtype=np.float32))
                continue

            t_roi = t_us[m]

            if use_polarity:
                p_roi = p[m]
                pos_counts, _ = np.histogram(t_roi[p_roi == 1], bins=edges_us)
                neg_counts, _ = np.histogram(t_roi[p_roi != 1], bins=edges_us)

                chans.append((pos_counts.astype(np.float32) / bin_w_s))
                chans.append((neg_counts.astype(np.float32) / bin_w_s))
            else:
                counts, _ = np.histogram(t_roi, bins=edges_us)
                chans.append((counts.astype(np.float32) / bin_w_s))

        X = np.stack(chans, axis=0)  # (C, n)

        # Luminance accumulation preprocessing (removes low-frequency bias and stochastic noise)
        # 1. Mean removal (high-pass filter to remove DC bias)
        X = X - X.mean(axis=1, keepdims=True)

        # 2. Smoothing (moving average to reduce stochastic noise)
        from scipy.ndimage import uniform_filter1d
        smoothing_window = 5  # M in the formula
        X = uniform_filter1d(X, size=smoothing_window, axis=1, mode='nearest')

        # Per-sample normalization (keeps scale stable across trials)
        # Use abs max since we now have positive and negative values after mean removal
        X = X / (np.abs(X).max() + 1e-6)
        return X

    def _events_to_image_sequence(
        self,
        events_aligned: np.ndarray,
        n_frames: Optional[int] = None,
        duration_s: Optional[float] = None,
        image_height: int = 224,
        image_width: int = 224,
    ) -> np.ndarray:
        """
        Convert events to a sequence of 2-channel image frames for CNN3D/CNN-LSTM.

        Each time bin becomes a 2-channel image where polarity is encoded:
        - Channel 0: Positive polarity events (p == 1)
        - Channel 1: Negative polarity events (p != 1)

        This is suitable for CNN3D and CNN-LSTM models that process temporal sequences.

        Args:
            events_aligned: Structured array with fields x, y, t, p
                           (t in microseconds, aligned to [0, duration_s*1e6])
            n_frames: Number of frames (time bins). Defaults to self.n_event_rate (2048)
            duration_s: Total duration in seconds. Defaults to self.vib_duration_s (8.0)
            image_height: Height of output images
            image_width: Width of output images

        Returns:
            np.ndarray of shape (n_frames, 2, image_height, image_width)
            - Channel 0: Positive polarity events
            - Channel 1: Negative polarity events

        Note:
            For a reduced sequence (e.g., for temporal models), use a smaller n_frames.
            For example, n_frames=64 gives 64 images covering 8 seconds.
        """
        if duration_s is None:
            duration_s = float(self.vib_duration_s)
        if n_frames is None:
            n_frames = int(self.n_event_rate)

        H_orig, W_orig = self.image_size  # Original camera resolution (720, 1280)

        # Initialize output: (n_frames, 2, H, W) - 2 channels for pos/neg polarity
        frames = np.zeros((n_frames, 2, image_height, image_width), dtype=np.float32)

        if len(events_aligned) == 0:
            return frames

        # Time bin edges in microseconds
        edges_us = np.linspace(0.0, duration_s * 1e6, n_frames + 1, dtype=np.float64)

        # Extract event data
        x = events_aligned["x"].astype(np.float32)
        y = events_aligned["y"].astype(np.float32)
        t_us = events_aligned["t"].astype(np.float64)
        p = events_aligned["p"]

        # Scale coordinates from original resolution to target image size
        x_scaled = (x / W_orig * image_width).astype(np.int32)
        y_scaled = (y / H_orig * image_height).astype(np.int32)

        # Clip to valid range
        x_scaled = np.clip(x_scaled, 0, image_width - 1)
        y_scaled = np.clip(y_scaled, 0, image_height - 1)

        # Assign events to time bins
        bin_indices = np.searchsorted(edges_us[1:], t_us, side='right')
        bin_indices = np.clip(bin_indices, 0, n_frames - 1)

        # Separate positive and negative polarity masks
        pos_mask = p == 1
        neg_mask = ~pos_mask

        # Accumulate events into frames using vectorized operations
        for frame_idx in range(n_frames):
            time_mask = bin_indices == frame_idx
            if not np.any(time_mask):
                continue

            # Positive events -> Channel 0
            pos_frame_mask = time_mask & pos_mask
            if np.any(pos_frame_mask):
                np.add.at(
                    frames[frame_idx, 0],
                    (y_scaled[pos_frame_mask], x_scaled[pos_frame_mask]),
                    1
                )

            # Negative events -> Channel 1
            neg_frame_mask = time_mask & neg_mask
            if np.any(neg_frame_mask):
                np.add.at(
                    frames[frame_idx, 1],
                    (y_scaled[neg_frame_mask], x_scaled[neg_frame_mask]),
                    1
                )

        # Normalize each frame independently
        # Use global max across both channels for consistent scaling
        for frame_idx in range(n_frames):
            max_val = max(frames[frame_idx, 0].max(), frames[frame_idx, 1].max())
            if max_val > 0:
                frames[frame_idx, 0] = frames[frame_idx, 0] / max_val
                frames[frame_idx, 1] = frames[frame_idx, 1] / max_val

        return frames


    def _align_events_to_vibration(
        self,
        events: np.ndarray,
        vib_data: np.ndarray,
        sample_name: str,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Align event timestamps to vibration timeline, then crop events to [0, vib_duration_s].
        Returns:
          events_aligned (subset of events), meta dict (offset, score, etc.)
        """
        # If cached offset exists, reuse it
        if self.offset_cache and sample_name in self.offsets:
            meta = self.offsets[sample_name]
            offset_s = float(meta["offset_s"])
            score = float(meta.get("score", 0.0))
        else:
            # Convert to seconds
            t_sec = self._to_seconds_events(events["t"])

            t_v, y_v, fs_v = self._get_vibration_time_and_signal(vib_data)
            # Zero-base vibration time
            t_v = (t_v - float(t_v.min())).astype(np.float32)

            # Compute event rate
            t_ev, ev_rate = self._compute_event_rate(t_sec, bin_ms=self.ev_bin_ms)

            # Normalize both time bases to start at 0
            t_ev_normalized = t_ev - float(t_ev.min())

            # Find offset by detecting when activity first rises above noise
            offset_s, score = self._estimate_offset_onset_detection(
                t_ev=t_ev_normalized,
                ev_rate=ev_rate,
                t_v=t_v,
                vib=y_v,
                threshold_percentile=80,
            )

            # Account for the initial event time
            t_ev_start = float(t_sec.min())

            # Final offset to apply to raw event timestamps
            # aligned_time = event_time - t_ev_start - offset
            final_offset_s = t_ev_start + offset_s

            meta = {
                "offset_s": final_offset_s,
                "score": score,
                "raw_offset": offset_s,
                "event_start": t_ev_start,
            }

            # Save offset cache
            if self.offset_cache:
                self.offsets[sample_name] = meta
                if self.offset_cache_path:
                    try:
                        with open(self.offset_cache_path, "wb") as f:
                            pickle.dump(self.offsets, f)
                    except Exception:
                        pass

        # Apply offset: aligned_event_time = event_time - offset_s
        t_sec = self._to_seconds_events(events["t"])
        t_aligned = t_sec - offset_s

        # Crop to vib window [0, vib_duration_s]
        start_s = 0.0
        end_s = float(self.vib_duration_s)
        keep = (t_aligned >= start_s) & (t_aligned <= end_s)

        events_crop = events[keep].copy()

        # Important: replace timestamps with aligned timestamps in original unit (microseconds) for downstream binning
        # We keep as float seconds in meta only; for voxel/binning we can use aligned seconds in a separate path.
        # For consistency with your existing code (expects 't' numeric), store aligned time in microseconds as int64.
        t_crop_us = (t_aligned[keep].astype(np.float64) * 1e6).astype(np.int64)
        events_crop["t"] = t_crop_us

        meta["events_total"] = int(len(events))
        meta["events_in_window"] = int(len(events_crop))

        if meta.get("score", 0.0) < 0.05:
            warnings.warn(f"Low alignment score for {sample_name}: {meta['score']:.3f}")

        return events_crop, meta

    def _get_data_pairs(self) -> List[Dict[str, Path]]:
        """Get all matching pairs of DV and LDV files."""
        dv_files = sorted(self.dv_dir.glob("*.dat"))
        ldv_files = sorted(self.ldv_dir.glob("*.txt"))

        # Create a mapping based on filenames (without extension)
        dv_dict = {f.stem: f for f in dv_files}
        ldv_dict = {f.stem: f for f in ldv_files}

        # Find matching pairs
        pairs = []
        matched_dv = set()
        matched_ldv = set()

        for name in dv_dict.keys():
            if name in ldv_dict:
                pairs.append({
                    'name': name,
                    'dv': dv_dict[name],
                    'ldv': ldv_dict[name],
                    'label': self._extract_label(name)
                })
                matched_dv.add(name)
                matched_ldv.add(name)

        # Check for special case: typo in filename (On_set1_trai4 vs On_set1_trail4)
        if 'On_set1_trai4' in dv_dict and 'On_set1_trail4' in ldv_dict:
            if 'On_set1_trai4' not in matched_dv and 'On_set1_trail4' not in matched_ldv:
                pairs.append({
                    'name': 'On_set1_trail4',
                    'dv': dv_dict['On_set1_trai4'],
                    'ldv': ldv_dict['On_set1_trail4'],
                    'label': self._extract_label('On_set1_trail4')
                })
                matched_dv.add('On_set1_trai4')
                matched_ldv.add('On_set1_trail4')

        # Report unmatched files
        unmatched_dv = set(dv_dict.keys()) - matched_dv
        unmatched_ldv = set(ldv_dict.keys()) - matched_ldv

        if unmatched_dv:
            print(f"Warning: Unmatched DV files: {unmatched_dv}")
        if unmatched_ldv:
            print(f"Warning: Unmatched LDV files: {unmatched_ldv}")

        print(f"Found {len(pairs)} matching DV-LDV file pairs")
        return pairs

    def _extract_label(self, filename: str) -> int:
        """Extract label from filename."""
        if filename.lower().startswith('off'):
            return 0
        elif filename.lower().startswith('on'):
            return 1
        else:
            return -1

    def _split_data(self):
        """
        Split data into train/val/test sets using stratified sampling.

        The dataset has 6 experimental configurations:
        - 2 illumination conditions (Off=low, On=high)
        - 3 camera settings (set1, set2, set3)
        - 5 trials each

        We split by trials to prevent data leakage:
        - Train: trials 1, 2, 3 (60% of data)
        - Val: trial 4 (20% of data)
        - Test: trial 5 (20% of data)

        This ensures each split has all 6 experimental configurations.
        """
        # Group samples by configuration
        config_groups = {}
        for idx, pair in enumerate(self.data_pairs):
            name = pair['name']

            # Extract configuration (illumination + camera setting)
            # Format: Off_set1_trail1, On_set2_trail3, etc.
            parts = name.split('_')
            if len(parts) >= 3:
                illumination = parts[0]  # Off or On
                camera_set = parts[1]    # set1, set2, or set3
                trial = parts[2]          # trail1, trail2, etc.

                # Extract trial number (handle typo: trai4 -> trail4)
                if 'trail' in trial:
                    trial_num = int(trial.replace('trail', ''))
                elif 'trai' in trial:
                    trial_num = int(trial.replace('trai', ''))
                else:
                    trial_num = 0

                config = f"{illumination}_{camera_set}"

                if config not in config_groups:
                    config_groups[config] = {1: [], 2: [], 3: [], 4: [], 5: []}

                if trial_num in config_groups[config]:
                    config_groups[config][trial_num].append(idx)

        # Create stratified splits based on trials
        train_indices = []
        val_indices = []
        test_indices = []

        # For each configuration, assign trials to splits
        for config, trials in config_groups.items():
            # Train: trials 1, 2, 3
            for trial_num in [1, 2, 3]:
                train_indices.extend(trials.get(trial_num, []))

            # Val: trial 4
            val_indices.extend(trials.get(4, []))

            # Test: trial 5
            test_indices.extend(trials.get(5, []))

        # Convert to numpy arrays for consistency
        train_indices = np.array(train_indices)
        val_indices = np.array(val_indices)
        test_indices = np.array(test_indices)

        # Shuffle within each split (with fixed seed for reproducibility)
        np.random.seed(42)
        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)
        np.random.shuffle(test_indices)

        # Assign indices based on mode
        if self.mode == 'train':
            self.indices = train_indices
        elif self.mode == 'val':
            self.indices = val_indices
        elif self.mode == 'test':
            self.indices = test_indices
        elif self.mode == 'all':
            # Combine all indices for full dataset access
            all_indices = np.concatenate([train_indices, val_indices, test_indices])
            # Sort by original order for consistency
            self.indices = np.sort(all_indices)
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Must be 'train', 'val', 'test', or 'all'")

        # Print split statistics
        print(f"Dataset mode: {self.mode}, samples: {len(self.indices)}")

        # Print configuration distribution
        config_counts = {}
        for idx in self.indices:
            pair = self.data_pairs[idx]
            parts = pair['name'].split('_')
            if len(parts) >= 2:
                config = f"{parts[0]}_{parts[1]}"
                config_counts[config] = config_counts.get(config, 0) + 1

        if config_counts:
            print(f"Configuration distribution in {self.mode} set:")
            for config, count in sorted(config_counts.items()):
                illumination = "Low" if config.startswith("Off") else "High"
                camera_set = config.split('_')[1]
                print(f"  {illumination} illumination, {camera_set}: {count} samples")

    def _get_cache_path(self, sample_name: str, data_type: str) -> Path:
        if self.cache_dir is None:
            return None

        extra = ""
        if self.event_representation == "event_rate":
            extra += f"_{self._roi_signature()}"
        if self.event_representation == "image_sequence":
            n_frames = self.image_seq_n_frames or self.n_event_rate
            extra += f"_h{self.image_seq_height}_w{self.image_seq_width}_n{n_frames}"
        if self.align:
            extra += "_align1"

        return self.cache_dir / f"{sample_name}_{data_type}_{self.event_representation}{extra}.lz4"




    def _save_to_cache(self, data: np.ndarray, cache_path: Path):
        """Save data to cache with compression."""
        if cache_path is None:
            return
        with lz4.frame.open(str(cache_path), 'wb') as f:
            pickle.dump(data, f)

    def _load_from_cache(self, cache_path: Path) -> Optional[np.ndarray]:
        """Load data from cache."""
        if cache_path is None or not cache_path.exists():
            return None
        try:
            with lz4.frame.open(str(cache_path), 'rb') as f:
                return pickle.load(f)
        except:
            return None

    def _load_events(self, dv_path: Path, sample_name: str = None) -> np.ndarray:
        """Load and preprocess event data from .dat file with optimizations."""
        # Check disk cache first
        if sample_name and self.cache_dir:
            cache_path = self._get_cache_path(sample_name, 'events')
            cached_data = self._load_from_cache(cache_path)
            if cached_data is not None:
                return cached_data

        # Check memory cache
        if self.cache_events and str(dv_path) in self.event_cache:
            return self.event_cache[str(dv_path)]

        # Load events using PSEELoader
        video = PSEELoader(str(dv_path))
        events = video.load_delta_t(self.event_duration)

        # No fixed window here; alignment/cropping happens in __getitem__
        events_loaded = events

        if self.align:
            # return raw events; we will align + then convert representation in __getitem__
            processed = events_loaded
        else:
            # old behavior with fixed window
            mask = (events_loaded['t'] >= self.event_window[0]) & (events_loaded['t'] <= self.event_window[1])
            events_filtered = events_loaded[mask]

            if self.event_representation == 'frame':
                processed = self._events_to_frame_vectorized(events_filtered)
            else:
                processed = self._events_to_raw(events_filtered)

        # Cache to disk if enabled
        if sample_name and self.cache_dir:
            self._save_to_cache(processed, cache_path)

        # Cache in memory if enabled
        if self.cache_events:
            self.event_cache[str(dv_path)] = processed

        return processed

    def _events_to_frame_vectorized(self, events: np.ndarray) -> np.ndarray:
        """
        Vectorized conversion of events to frame representation.
        Much faster than iterating through events.
        """
        frame = np.zeros((2, *self.image_size), dtype=np.float32)

        if len(events) == 0:
            return frame

        # Filter valid coordinates
        valid_mask = (events['x'] < self.image_size[1]) & (events['y'] < self.image_size[0])
        valid_events = events[valid_mask]

        if len(valid_events) == 0:
            return frame

        # Separate positive and negative events
        pos_mask = valid_events['p'] == 1
        neg_mask = ~pos_mask

        # Use np.add.at for efficient accumulation
        np.add.at(frame[0], (valid_events['y'][pos_mask], valid_events['x'][pos_mask]), 1)
        np.add.at(frame[1], (valid_events['y'][neg_mask], valid_events['x'][neg_mask]), 1)

        # Normalize
        max_val = frame.max()
        if max_val > 0:
            frame = frame / max_val

        return frame

    def _events_to_raw(self, events: np.ndarray) -> Dict:
        """Return raw events as dictionary."""
        return {
            'x': events['x'].astype(np.float32),
            'y': events['y'].astype(np.float32),
            't': events['t'].astype(np.float32),
            'p': events['p'].astype(np.float32)
        }

    def _load_vibration_fast(self, ldv_path: Path) -> np.ndarray:
        """Optimized vibration data loading using numpy's loadtxt."""
        try:
            # Try to use numpy's fast loading
            data = np.loadtxt(ldv_path, skiprows=5, dtype=np.float32)

            if data.ndim == 1:
                data = data.reshape(-1, 2)

            # Limit samples if specified
            if self.max_vibration_samples and len(data) > self.max_vibration_samples:
                indices = np.linspace(0, len(data) - 1, self.max_vibration_samples, dtype=int)
                data = data[indices]

            return data

        except:
            # Fallback to original method if fast loading fails
            return self._load_vibration(ldv_path)

    def _load_vibration(self, ldv_path: Path) -> np.ndarray:
        """Load and preprocess vibration data from .txt file."""
        with open(ldv_path, 'r') as f:
            lines = f.readlines()

        # Find where the data starts
        data_start = 0
        for i, line in enumerate(lines):
            if line.strip() and not any(char.isalpha() or char == '[' or char == ']' for char in line):
                data_start = i
                break

        # Parse data using list comprehension (faster)
        data = []
        for line in lines[data_start:]:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    data.append([float(parts[0]), float(parts[1])])
                except ValueError:
                    continue

        vibration_signal = np.array(data, dtype=np.float32)

        # Limit samples if specified
        if self.max_vibration_samples and len(vibration_signal) > self.max_vibration_samples:
            indices = np.linspace(0, len(vibration_signal) - 1, self.max_vibration_samples, dtype=int)
            vibration_signal = vibration_signal[indices]

        return vibration_signal

    def _preprocess_all_data(self, num_workers: int = 4):
        """Preprocess all data in parallel at initialization."""
        print(f"Preprocessing {len(self.indices)} samples with {num_workers} workers...")

        def process_sample(idx):
            actual_idx = self.indices[idx]
            sample = self.data_pairs[actual_idx]

            # Load and cache events
            _ = self._load_events(sample['dv'], sample['name'])

            return idx

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            list(tqdm(executor.map(process_sample, range(len(self.indices))),
                     total=len(self.indices),
                     desc="Preprocessing"))

        print("Preprocessing complete!")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        if self.use_windowing:
            # Each original sample is split into multiple windows
            num_windows = self.n_event_rate // self.window_size
            return len(self.indices) * num_windows
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Get a sample from the dataset."""
        if self.use_windowing:
            # Map windowed idx to (original_sample_idx, window_idx)
            num_windows = self.n_event_rate // self.window_size
            original_idx = idx // num_windows
            window_idx = idx % num_windows

            # Apply shuffling if enabled (training mode only)
            if self.shuffle_windows and self.mode == 'train':
                # Shuffle window index for this sample
                np.random.seed(idx)  # Use idx as seed for reproducibility within epoch
                window_perm = np.random.permutation(num_windows)
                window_idx = window_perm[window_idx]

            actual_idx = self.indices[original_idx]
        else:
            actual_idx = self.indices[idx]
            window_idx = None

        sample = self.data_pairs[actual_idx]

        # Load raw events (or processed if align=False)
        event_data = self._load_events(sample['dv'], sample['name'])

        # Load vibration
        vibration_data = self._load_vibration_fast(sample['ldv'])

        # Get vibration signal and resample to n_vibration_samples
        t_v, y_v, fs_v = self._get_vibration_time_and_signal(vibration_data)
        t_v = (t_v - float(t_v.min())).astype(np.float32)
        t_grid = np.linspace(0.0, self.vib_duration_s, self.n_vibration_samples, dtype=np.float32)
        y_resampled = np.interp(t_grid, t_v, y_v, left=0.0, right=0.0).astype(np.float32)

        # Apply bandpass filter if configured
        if self.bandpass_filter is not None:
            low_hz, high_hz = self.bandpass_filter
            fs_resampled = self.n_vibration_samples / self.vib_duration_s
            y_resampled = self._apply_bandpass_filter(y_resampled, fs_resampled, low_hz, high_hz)

        # If align enabled, align raw events to vibration and then build representation
        if self.align:
            # event_data is raw structured array here
            events_aligned, meta = self._align_events_to_vibration(
                events=event_data,
                vib_data=vibration_data,
                sample_name=sample['name']
            )

            # Apply circular shift augmentation (training only)
            # This must happen BEFORE converting to representation so events and vibration stay aligned
            if self.circular_shift_aug and self.mode == 'train':
                events_aligned, y_resampled = self._apply_circular_shift(events_aligned, y_resampled, 2048)

            # Try disk cache for aligned processed representation (only if no augmentation)
            use_cache = self.cache_dir and not (self.circular_shift_aug and self.mode == 'train')

            if use_cache:
                cache_path = self._get_cache_path(sample['name'], f'events_aligned')
                cached = self._load_from_cache(cache_path)
            else:
                cached = None

            if cached is not None:
                processed = cached
            else:
                # Now convert aligned events to desired representation
                if self.event_representation == 'frame':
                    processed = self._events_to_frame_vectorized(events_aligned)
                elif self.event_representation == 'event_rate':
                    processed = self._events_to_eventrate_2048(
                        events_aligned,
                        n=self.n_event_rate,
                        use_polarity=self.use_polarity_event_rate,
                        rois=self.rois_resolved,
                    )
                elif self.event_representation == 'image_sequence':
                    processed = self._events_to_image_sequence(
                        events_aligned,
                        n_frames=self.image_seq_n_frames,
                        image_height=self.image_seq_height,
                        image_width=self.image_seq_width,
                    )
                else:
                    processed = self._events_to_raw(events_aligned)

                if use_cache:
                    self._save_to_cache(processed, cache_path)

            event_data = processed

        # Extract specific window if windowing is enabled
        if self.use_windowing and window_idx is not None and not isinstance(event_data, dict):
            # Extract the specific window from the full sequence
            start_idx = window_idx * self.window_size
            end_idx = start_idx + self.window_size

            # Handle different event representations
            if event_data.ndim == 2:
                # event_rate: (C, T) -> extract (C, window_size)
                event_data = event_data[:, start_idx:end_idx]
            elif event_data.ndim == 4:
                # image_sequence: (n_frames, C, H, W) -> extract (window_size, C, H, W)
                event_data = event_data[start_idx:end_idx, :, :, :]

            # Extract corresponding vibration window
            y_resampled = y_resampled[start_idx:end_idx]

        # Convert to PyTorch tensors
        if isinstance(event_data, dict):
            # Raw events
            event_tensor = {k: torch.from_numpy(v) for k, v in event_data.items()}
        else:
            event_tensor = torch.from_numpy(event_data)

        vibration_tensor = torch.from_numpy(y_resampled)

        # Apply transforms if any
        if self.transform:
            event_tensor = self.transform(event_tensor)

        if self.target_transform:
            vibration_tensor = self.target_transform(vibration_tensor)

        # if self.is_spiking:
        #     print("Spiking Data")

        return event_tensor, vibration_tensor, sample['label']

    def get_sample_info(self, idx: int) -> Dict:
        """Get information about a sample without loading the data."""
        actual_idx = self.indices[idx]
        return self.data_pairs[actual_idx]

    def get_all_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load all data at once and return as tensors.

        Returns:
            Tuple of (all_events, all_vibrations, all_labels)
            - all_events: Tensor of shape (N, C, H, W) or dict of tensors
            - all_vibrations: Tensor of shape (N, max_samples, 2)
            - all_labels: Tensor of shape (N,)

        Note: This loads all data into memory at once. Use with caution for large datasets.
        """
        print(f"Loading all {len(self)} samples into memory...")

        all_events = []
        all_vibrations = []
        all_labels = []

        for i in tqdm(range(len(self)), desc="Loading samples"):
            event_data, vibration_data, label = self[i]
            all_events.append(event_data)
            all_vibrations.append(vibration_data)
            all_labels.append(label)

        # Stack events
        if isinstance(all_events[0], dict):
            # For raw events, return dict of stacked tensors
            stacked_events = {}
            for key in all_events[0].keys():
                stacked_events[key] = torch.stack([e[key] for e in all_events])
        else:
            # For frame/voxel representations
            stacked_events = torch.stack(all_events)

        # Pad vibration signals to same length
        max_len = max(v.shape[0] for v in all_vibrations)
        padded_vibrations = []
        for v in all_vibrations:
            if len(v) < max_len:
                padding = torch.zeros(max_len - len(v), 2)
                v_padded = torch.cat([v, padding], dim=0)
            else:
                v_padded = v
            padded_vibrations.append(v_padded)

        stacked_vibrations = torch.stack(padded_vibrations)
        stacked_labels = torch.tensor(all_labels, dtype=torch.long)

        print(f"Loaded all data:")
        if isinstance(stacked_events, dict):
            print(f"  Events: dict with keys {list(stacked_events.keys())}")
        else:
            print(f"  Events shape: {stacked_events.shape}")
        print(f"  Vibrations shape: {stacked_vibrations.shape}")
        print(f"  Labels shape: {stacked_labels.shape}")

        return stacked_events, stacked_vibrations, stacked_labels

    def get_all_data_numpy(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load all data at once and return as numpy arrays.

        Returns:
            Tuple of (all_events, all_vibrations, all_labels) as numpy arrays
        """
        events, vibrations, labels = self.get_all_data()

        if isinstance(events, dict):
            # Convert dict of tensors to dict of numpy arrays
            events_np = {k: v.numpy() for k, v in events.items()}
        else:
            events_np = events.numpy()

        return events_np, vibrations.numpy(), labels.numpy()

    def get_data_by_mode(self) -> Dict:
        """
        Get all data organized by the current mode (train/val/test).

        Returns:
            Dictionary with 'events', 'vibrations', 'labels' keys
        """
        events, vibrations, labels = self.get_all_data()

        return {
            'mode': self.mode,
            'events': events,
            'vibrations': vibrations,
            'labels': labels,
            'n_samples': len(self),
            'n_normal': (labels == 0).sum().item(),
            'n_fault': (labels == 1).sum().item()
        }

    def get_split_labels(self) -> np.ndarray:
        """
        Get split labels for each sample when mode='all'.

        Returns:
            Array of split labels: 0=train, 1=val, 2=test
        """
        if self.mode != 'all':
            raise ValueError("get_split_labels() only works when mode='all'")

        # Create split label array
        split_labels = np.zeros(len(self.indices), dtype=int)

        # Reconstruct train/val/test indices to determine split for each sample
        for idx, pair in enumerate(self.data_pairs):
            name = pair['name']
            parts = name.split('_')
            if len(parts) >= 3:
                trial = parts[2]  # trail1, trail2, etc.

                if 'trail' in trial:
                    trial_num = int(trial.replace('trail', ''))
                elif 'trai' in trial:
                    trial_num = int(trial.replace('trai', ''))
                else:
                    continue

                # Determine split based on trial number
                if trial_num in [1, 2, 3]:
                    split = 0  # train
                elif trial_num == 4:
                    split = 1  # val
                elif trial_num == 5:
                    split = 2  # test
                else:
                    continue

                # Find this index in self.indices
                if idx in self.indices:
                    pos = np.where(self.indices == idx)[0][0]
                    split_labels[pos] = split

        return split_labels

    def get_all_data_with_splits(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
        """
        Load all data with split labels when mode='all'.

        Returns:
            Tuple of (events, vibrations, labels, split_labels)
            - split_labels: 0=train, 1=val, 2=test
        """
        if self.mode != 'all':
            raise ValueError("get_all_data_with_splits() only works when mode='all'")

        events, vibrations, labels = self.get_all_data()
        split_labels = self.get_split_labels()

        return events, vibrations, labels, split_labels


def create_beam_dataloaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    event_representation: str = 'frame',
    cache_dir: Optional[str] = None,
    preprocess_all: bool = False,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders for the Beam dataset.

    Args:
        data_root: Root directory containing DV and LDV folders
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        event_representation: How to represent events ('frame', 'voxel', 'raw', 'sparse')
        cache_dir: Directory for disk caching
        preprocess_all: Whether to preprocess all data at initialization
        **dataset_kwargs: Additional arguments to pass to BeamDataset

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Setup cache directory
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    # Create datasets
    train_dataset = BeamDataset(
        data_root=data_root,
        mode='train',
        event_representation=event_representation,
        cache_dir=str(cache_dir / 'train') if cache_dir else None,
        preprocess_all=preprocess_all,
        **dataset_kwargs
    )

    val_dataset = BeamDataset(
        data_root=data_root,
        mode='val',
        event_representation=event_representation,
        cache_dir=str(cache_dir / 'val') if cache_dir else None,
        preprocess_all=False,  # Don't preprocess val/test
        **dataset_kwargs
    )

    test_dataset = BeamDataset(
        data_root=data_root,
        mode='test',
        event_representation=event_representation,
        cache_dir=str(cache_dir / 'test') if cache_dir else None,
        preprocess_all=False,
        **dataset_kwargs
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0  # Keep workers alive
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Example usage
    import time

    data_root = "data/XJTU-DV-Beam"
    cache_dir = "data_loaders/cache"

    print("Testing beam dataloader...")

    # # Create dataset with optimizations
    # start_time = time.time()
    # dataset = BeamDataset(
    #     data_root=data_root,
    #     event_representation='event_rate',
    #     max_vibration_samples=None,
    #     # cache_dir=cache_dir,
    #     cache_events=True,
    #     preprocess_all=False,  # Set to True to preprocess all at once
    #     num_workers_preprocess=0,
    #     align=True,
    #     voxel_bins=50,
    # )
    # init_time = time.time() - start_time
    # print(f"Dataset initialization time: {init_time:.3f}s")

    # print(f"Dataset size: {len(dataset)}")
    # x, y, label = dataset[0]
    # print(x.shape, y.shape)

    # Full-frame (default): 1 ROI => C=2
    ds = BeamDataset(data_root="data/XJTU-DV-Beam", event_representation="event_rate", align=True)
    x, y, _ = ds[0]
    print("default:", x.shape, y.shape)  # (2,2048) (2048,)

    # Grid ROIs: 3x3 => 9 ROIs => C=18
    ds = BeamDataset(
        data_root="data/XJTU-DV-Beam",
        event_representation="event_rate",
        align=True,
        roi_grid=(3, 3),
        use_polarity_event_rate=True,
    )
    x, y, _ = ds[0]
    print("grid 3x3:", x.shape, y.shape)  # (18,2048) (2048,)

    # Image sequence for CNN3D/CNN-LSTM: (n_frames, 2, H, W) - 2 channels (pos/neg polarity)
    ds = BeamDataset(
        data_root="data/XJTU-DV-Beam",
        event_representation="image_sequence",
        align=True,
        image_seq_height=720,   # Original camera resolution
        image_seq_width=1280,
        image_seq_n_frames=64,  # 64 frames covering 8 seconds
    )
    x, y, _ = ds[0]
    print("image_sequence:", x.shape, y.shape)  # (64, 2, 720, 1280) (2048,)
    print(f"  Channel 0 (pos events) range: [{x[:, 0].min():.3f}, {x[:, 0].max():.3f}]")
    print(f"  Channel 1 (neg events) range: [{x[:, 1].min():.3f}, {x[:, 1].max():.3f}]")


    # # Test loading speed
    # if len(dataset) > 0:
    #     # First load (not cached)
    #     start_time = time.time()
    #     event_data, vibration_data, label = dataset[0]
    #     first_load_time = time.time() - start_time
    #     print(f"First load time: {first_load_time:.3f}s")
    #     print(f"Event data shape: {event_data.shape}")
    #     print(f"Vibration data shape: {vibration_data.shape}")

    #     # Second load (should be cached)
    #     start_time = time.time()
    #     event_data, vibration_data, label = dataset[0]
    #     cached_load_time = time.time() - start_time
    #     print(f"Cached load time: {cached_load_time:.3f}s")

    #     if first_load_time > 0:
    #         print(f"Speedup: {first_load_time/cached_load_time:.1f}x")