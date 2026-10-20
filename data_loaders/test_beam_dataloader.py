"""
Test script for BeamDataset dataloader
Demonstrates various features and visualizations
Including optimized performance features like caching and vectorization
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from beam_dataloader import BeamDataset, create_beam_dataloaders
import torch
from tqdm import tqdm


def visualize_event_frame(event_frame, title="Event Frame"):
    """Visualize a 2-channel event frame."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Positive events
    axes[0].imshow(event_frame[0], cmap='Reds')
    axes[0].set_title("Positive Events")
    axes[0].axis('off')

    # Negative events
    axes[1].imshow(event_frame[1], cmap='Blues')
    axes[1].set_title("Negative Events")
    axes[1].axis('off')

    # Combined (positive - negative)
    combined = event_frame[0] - event_frame[1]
    axes[2].imshow(combined, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[2].set_title("Combined (Pos - Neg)")
    axes[2].axis('off')

    fig.suptitle(title)
    plt.tight_layout()
    return fig


def visualize_vibration_signal(vibration_data, title="Vibration Signal"):
    """Visualize a vibration signal."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Time series
    axes[0].plot(vibration_data[:, 0], vibration_data[:, 1], 'b-', linewidth=0.5)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Vibration Velocity (m/s)")
    axes[0].set_title("Vibration Time Series")
    axes[0].grid(True, alpha=0.3)

    # FFT
    if len(vibration_data) > 1:
        # Calculate FFT
        dt = np.mean(np.diff(vibration_data[:, 0]))  # Average time step
        fs = 1.0 / dt  # Sampling frequency

        # Apply window to reduce spectral leakage
        window = np.hanning(len(vibration_data))
        windowed_signal = vibration_data[:, 1] * window

        fft_vals = np.fft.fft(windowed_signal)
        fft_freq = np.fft.fftfreq(len(vibration_data), dt)

        # Only plot positive frequencies
        pos_freq_idx = fft_freq > 0

        axes[1].semilogy(fft_freq[pos_freq_idx], np.abs(fft_vals[pos_freq_idx]), 'r-', linewidth=0.5)
        axes[1].set_xlabel("Frequency (Hz)")
        axes[1].set_ylabel("Amplitude")
        axes[1].set_title("Frequency Spectrum (FFT)")
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim([0, fs/2])  # Nyquist frequency

    fig.suptitle(title)
    plt.tight_layout()
    return fig


def test_basic_loading():
    """Test basic data loading functionality."""
    print("=" * 50)
    print("Testing Basic Data Loading")
    print("=" * 50)

    data_root = "/home/humaid/Projects/XJTU_DV_FaultDiagnosis/data/XJTU-DV-Beam"

    # Test different event representations
    for representation in ['frame', 'voxel']:
        print(f"\nTesting {representation} representation...")

        dataset = BeamDataset(
            data_root=data_root,
            event_representation=representation,
            max_vibration_samples=2000,
            mode='train'
        )

        if len(dataset) > 0:
            event_data, vibration_data, label = dataset[0]
            print(f"  Event data shape: {event_data.shape}")
            print(f"  Vibration data shape: {vibration_data.shape}")
            print(f"  Label: {label} ({'Normal' if label == 0 else 'Fault' if label == 1 else 'Unknown'})")

            # Check data statistics
            print(f"  Event data range: [{event_data.min():.3f}, {event_data.max():.3f}]")
            print(f"  Vibration data range: [{vibration_data.min():.6f}, {vibration_data.max():.6f}]")


def test_dataloader():
    """Test DataLoader functionality."""
    print("\n" + "=" * 50)
    print("Testing DataLoader")
    print("=" * 50)

    data_root = "/home/humaid/Projects/XJTU_DV_FaultDiagnosis/data/XJTU-DV-Beam"

    train_loader, val_loader, test_loader = create_beam_dataloaders(
        data_root=data_root,
        batch_size=4,
        num_workers=0,  # Set to 0 for debugging
        event_representation='frame',
        max_vibration_samples=2000
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Test one batch
    if len(train_loader) > 0:
        for batch_idx, (events, vibrations, labels) in enumerate(train_loader):
            print(f"\nBatch {batch_idx}:")
            print(f"  Events batch shape: {events.shape}")
            print(f"  Vibrations batch shape: {vibrations.shape}")
            print(f"  Labels batch: {labels}")

            if batch_idx == 0:  # Only process first batch
                break


def test_visualization():
    """Test visualization of data samples."""
    print("\n" + "=" * 50)
    print("Testing Visualization")
    print("=" * 50)

    data_root = "/home/humaid/Projects/XJTU_DV_FaultDiagnosis/data/XJTU-DV-Beam"

    dataset = BeamDataset(
        data_root=data_root,
        event_representation='frame',
        max_vibration_samples=2000,
        mode='test'
    )

    if len(dataset) > 0:
        # Visualize first sample
        event_data, vibration_data, label = dataset[0]
        info = dataset.get_sample_info(0)

        print(f"Visualizing sample: {info['name']}")
        print(f"Label: {label} ({'Normal' if label == 0 else 'Fault' if label == 1 else 'Unknown'})")

        # Convert tensors to numpy for visualization
        event_np = event_data.numpy()
        vibration_np = vibration_data.numpy()

        # Visualize event frame
        fig1 = visualize_event_frame(event_np, f"Event Frame - {info['name']}")
        plt.savefig('/home/humaid/Projects/XJTU_DV_FaultDiagnosis/data_loaders/event_visualization.png', dpi=150)
        plt.show()

        # Visualize vibration signal
        fig2 = visualize_vibration_signal(vibration_np, f"Vibration Signal - {info['name']}")
        plt.savefig('/home/humaid/Projects/XJTU_DV_FaultDiagnosis/data_loaders/vibration_visualization.png', dpi=150)
        plt.show()


def test_statistics():
    """Compute and display dataset statistics."""
    print("\n" + "=" * 50)
    print("Computing Dataset Statistics")
    print("=" * 50)

    data_root = "/home/humaid/Projects/XJTU_DV_FaultDiagnosis/data/XJTU-DV-Beam"

    for mode in ['train', 'val', 'test']:
        print(f"\n{mode.upper()} Set Statistics:")

        dataset = BeamDataset(
            data_root=data_root,
            event_representation='frame',
            max_vibration_samples=2000,
            mode=mode
        )

        if len(dataset) == 0:
            print("  No samples in this split")
            continue

        # Count labels
        label_counts = {0: 0, 1: 0, -1: 0}
        vibration_lengths = []
        event_counts = []

        for i in tqdm(range(len(dataset)), desc=f"Processing {mode}", leave=False):
            event_data, vibration_data, label = dataset[i]
            label_counts[label] += 1
            vibration_lengths.append(len(vibration_data))

            # Count non-zero events
            if isinstance(event_data, torch.Tensor):
                event_counts.append((event_data > 0).sum().item())

        print(f"  Total samples: {len(dataset)}")
        print(f"  Normal samples: {label_counts[0]}")
        print(f"  Fault samples: {label_counts[1]}")
        print(f"  Unknown samples: {label_counts[-1]}")

        if vibration_lengths:
            print(f"  Avg vibration length: {np.mean(vibration_lengths):.1f} samples")
            print(f"  Avg event count: {np.mean(event_counts):.1f} events")


def test_memory_efficiency():
    """Test memory efficiency with caching."""
    print("\n" + "=" * 50)
    print("Testing Memory Efficiency")
    print("=" * 50)

    data_root = "/home/humaid/Projects/XJTU_DV_FaultDiagnosis/data/XJTU-DV-Beam"

    # Test without caching
    print("\nWithout caching:")
    dataset_no_cache = BeamDataset(
        data_root=data_root,
        event_representation='frame',
        max_vibration_samples=1000,
        mode='train',
        cache_events=False
    )

    import time
    start_time = time.time()
    for i in range(min(5, len(dataset_no_cache))):
        _ = dataset_no_cache[i]
    time_no_cache = time.time() - start_time
    print(f"  Time for 5 samples: {time_no_cache:.3f}s")

    # Test with caching
    print("\nWith caching:")
    dataset_cache = BeamDataset(
        data_root=data_root,
        event_representation='frame',
        max_vibration_samples=1000,
        mode='train',
        cache_events=True
    )

    # First pass (loading into cache)
    start_time = time.time()
    for i in range(min(5, len(dataset_cache))):
        _ = dataset_cache[i]
    time_first_pass = time.time() - start_time
    print(f"  Time for 5 samples (first pass): {time_first_pass:.3f}s")

    # Second pass (from cache)
    start_time = time.time()
    for i in range(min(5, len(dataset_cache))):
        _ = dataset_cache[i]
    time_second_pass = time.time() - start_time
    print(f"  Time for 5 samples (cached): {time_second_pass:.3f}s")
    print(f"  Speedup: {time_first_pass/time_second_pass:.1f}x")


if __name__ == "__main__":
    # Run all tests
    test_basic_loading()
    test_dataloader()
    test_statistics()
    test_memory_efficiency()

    # Visualization test (optional - requires display)
    try:
        test_visualization()
    except Exception as e:
        print(f"\nVisualization test failed (likely no display): {e}")

    print("\n" + "=" * 50)
    print("All tests completed!")
    print("=" * 50)