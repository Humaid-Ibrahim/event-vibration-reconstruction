"""
Optimized Beam Dataset Dataloader for XJTU-DV-Beam
Performance improvements:
- Vectorized event processing using NumPy operations
- Efficient sparse matrix representation for event frames
- Memory-mapped file access for large files
- Parallel data loading with multiprocessing
- Optimized caching with compression
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
from scipy.sparse import csr_matrix
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
        event_representation: str = 'frame',  # 'frame', 'voxel', 'raw', 'sparse'
        voxel_bins: int = 5,  # Number of time bins for voxel representation
        cache_events: bool = False,  # Whether to cache processed events in memory
        cache_dir: Optional[str] = None,  # Directory for disk caching
        use_sparse: bool = True,  # Use sparse matrices for event frames
        preprocess_all: bool = False,  # Preprocess all data at initialization
        num_workers_preprocess: int = 4  # Number of workers for preprocessing
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
            event_representation: How to represent events ('frame', 'voxel', 'raw', 'sparse')
            voxel_bins: Number of time bins for voxel representation
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
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode
        self.max_vibration_samples = max_vibration_samples
        self.event_representation = event_representation
        self.voxel_bins = voxel_bins
        self.cache_events = cache_events
        self.use_sparse = use_sparse
        self.event_cache = {} if cache_events else None

        # Setup disk caching
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None

        # Get all data file pairs
        self.data_pairs = self._get_data_pairs()

        # Split data into train/val/test
        self._split_data()

        # Preprocess all data if requested
        if preprocess_all:
            self._preprocess_all_data(num_workers_preprocess)

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
        """Get cache file path for a sample."""
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{sample_name}_{data_type}_{self.event_representation}.lz4"

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

        # Vectorized filtering (much faster than list comprehension)
        mask = (events['t'] >= self.event_window[0]) & (events['t'] <= self.event_window[1])
        events_filtered = events[mask]

        # Process events based on representation type
        if self.event_representation == 'frame':
            processed = self._events_to_frame_vectorized(events_filtered)
        elif self.event_representation == 'sparse':
            processed = self._events_to_sparse_frame(events_filtered)
        elif self.event_representation == 'voxel':
            processed = self._events_to_voxel_vectorized(events_filtered)
        else:  # raw
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

    def _events_to_sparse_frame(self, events: np.ndarray) -> Dict:
        """
        Convert events to sparse matrix representation for memory efficiency.
        """
        if len(events) == 0:
            return {
                'pos_sparse': csr_matrix(self.image_size, dtype=np.float32),
                'neg_sparse': csr_matrix(self.image_size, dtype=np.float32)
            }

        # Filter valid coordinates
        valid_mask = (events['x'] < self.image_size[1]) & (events['y'] < self.image_size[0])
        valid_events = events[valid_mask]

        if len(valid_events) == 0:
            return {
                'pos_sparse': csr_matrix(self.image_size, dtype=np.float32),
                'neg_sparse': csr_matrix(self.image_size, dtype=np.float32)
            }

        # Separate polarities
        pos_mask = valid_events['p'] == 1
        neg_mask = ~pos_mask

        # Create sparse matrices
        pos_events = valid_events[pos_mask]
        neg_events = valid_events[neg_mask]

        # Count events at each pixel using bincount
        pos_data = np.ones(len(pos_events), dtype=np.float32)
        neg_data = np.ones(len(neg_events), dtype=np.float32)

        pos_sparse = csr_matrix(
            (pos_data, (pos_events['y'], pos_events['x'])),
            shape=self.image_size,
            dtype=np.float32
        )

        neg_sparse = csr_matrix(
            (neg_data, (neg_events['y'], neg_events['x'])),
            shape=self.image_size,
            dtype=np.float32
        )

        # Normalize
        pos_max = pos_sparse.max()
        neg_max = neg_sparse.max()

        if pos_max > 0:
            pos_sparse = pos_sparse / pos_max
        if neg_max > 0:
            neg_sparse = neg_sparse / neg_max

        return {
            'pos_sparse': pos_sparse,
            'neg_sparse': neg_sparse
        }

    def _events_to_voxel_vectorized(self, events: np.ndarray) -> np.ndarray:
        """
        Vectorized conversion of events to voxel representation.
        """
        voxel = np.zeros((self.voxel_bins * 2, *self.image_size), dtype=np.float32)

        if len(events) == 0:
            return voxel

        # Filter valid coordinates
        valid_mask = (events['x'] < self.image_size[1]) & (events['y'] < self.image_size[0])
        valid_events = events[valid_mask]

        if len(valid_events) == 0:
            return voxel

        # Compute time bins
        t_min = valid_events['t'].min()
        t_max = valid_events['t'].max()
        t_range = max(t_max - t_min, 1)  # Avoid division by zero

        t_normalized = (valid_events['t'] - t_min) / t_range
        t_bins = np.clip((t_normalized * (self.voxel_bins - 1)).astype(int), 0, self.voxel_bins - 1)

        # Separate polarities
        pos_mask = valid_events['p'] == 1
        neg_mask = ~pos_mask

        # Accumulate events in voxel grid
        for t_bin in range(self.voxel_bins):
            # Positive events
            t_mask_pos = (t_bins == t_bin) & pos_mask
            if t_mask_pos.any():
                np.add.at(voxel[t_bin],
                         (valid_events['y'][t_mask_pos], valid_events['x'][t_mask_pos]),
                         1)

            # Negative events
            t_mask_neg = (t_bins == t_bin) & neg_mask
            if t_mask_neg.any():
                np.add.at(voxel[t_bin + self.voxel_bins],
                         (valid_events['y'][t_mask_neg], valid_events['x'][t_mask_neg]),
                         1)

        # Normalize
        max_val = voxel.max()
        if max_val > 0:
            voxel = voxel / max_val

        return voxel

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
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Get a sample from the dataset."""
        actual_idx = self.indices[idx]
        sample = self.data_pairs[actual_idx]

        # Load event data
        event_data = self._load_events(sample['dv'], sample['name'])

        # Load vibration data (using optimized version)
        vibration_data = self._load_vibration_fast(sample['ldv'])

        # Convert to PyTorch tensors
        if isinstance(event_data, dict):
            if 'pos_sparse' in event_data:  # Sparse representation
                # Convert sparse to dense for PyTorch
                pos_dense = event_data['pos_sparse'].toarray()
                neg_dense = event_data['neg_sparse'].toarray()
                event_tensor = torch.from_numpy(np.stack([pos_dense, neg_dense], axis=0))
            else:  # Raw events
                event_tensor = {k: torch.from_numpy(v) for k, v in event_data.items()}
        else:
            event_tensor = torch.from_numpy(event_data)

        vibration_tensor = torch.from_numpy(vibration_data)

        # Apply transforms if any
        if self.transform:
            event_tensor = self.transform(event_tensor)

        if self.target_transform:
            vibration_tensor = self.target_transform(vibration_tensor)

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

    data_root = "/home/humaid/Projects/XJTU_DV_FaultDiagnosis/data/XJTU-DV-Beam"
    cache_dir = "/home/humaid/Projects/XJTU_DV_FaultDiagnosis/data_loaders/cache"

    print("Testing beam dataloader...")

    # Create dataset with optimizations
    start_time = time.time()
    dataset = BeamDataset(
        data_root=data_root,
        event_representation='frame',
        max_vibration_samples=None,
        cache_dir=cache_dir,
        cache_events=True,
        preprocess_all=False,  # Set to True to preprocess all at once
        num_workers_preprocess=4
    )
    init_time = time.time() - start_time
    print(f"Dataset initialization time: {init_time:.3f}s")

    print(f"Dataset size: {len(dataset)}")

    # Test loading speed
    if len(dataset) > 0:
        # First load (not cached)
        start_time = time.time()
        event_data, vibration_data, label = dataset[0]
        first_load_time = time.time() - start_time
        print(f"First load time: {first_load_time:.3f}s")
        print(f"Event data shape: {event_data.shape}")
        print(f"Vibration data shape: {vibration_data.shape}")

        # Second load (should be cached)
        start_time = time.time()
        event_data, vibration_data, label = dataset[0]
        cached_load_time = time.time() - start_time
        print(f"Cached load time: {cached_load_time:.3f}s")

        if first_load_time > 0:
            print(f"Speedup: {first_load_time/cached_load_time:.1f}x")