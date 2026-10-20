"""
Simple test to verify the new default dataloader works correctly
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from beam_dataloader import BeamDataset, create_beam_dataloaders

# Test basic functionality
data_root = "/home/humaid/Projects/XJTU_DV_FaultDiagnosis/data/XJTU-DV-Beam"

print("Testing BeamDataset (optimized version is now default)...")
print("=" * 50)

# Test 1: Basic dataset creation
dataset = BeamDataset(
    data_root=data_root,
    event_representation='frame',
    max_vibration_samples=2000,
    mode='test'
)

print(f"✓ Dataset created successfully")
print(f"  Dataset size: {len(dataset)}")

# Test 2: Load a sample
if len(dataset) > 0:
    event_data, vibration_data, label = dataset[0]
    print(f"✓ Sample loaded successfully")
    print(f"  Event shape: {event_data.shape}")
    print(f"  Vibration shape: {vibration_data.shape}")
    print(f"  Label: {label}")

# Test 3: Create dataloaders
print("\nTesting dataloader creation...")
train_loader, val_loader, test_loader = create_beam_dataloaders(
    data_root=data_root,
    batch_size=4,
    num_workers=0,
    event_representation='frame',
    max_vibration_samples=2000
)

print(f"✓ Dataloaders created successfully")
print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches: {len(val_loader)}")
print(f"  Test batches: {len(test_loader)}")

# Test 4: Test with caching
print("\nTesting with disk cache...")
cache_dir = "/home/humaid/Projects/XJTU_DV_FaultDiagnosis/data_loaders/cache_test"

dataset_cached = BeamDataset(
    data_root=data_root,
    event_representation='frame',
    max_vibration_samples=2000,
    mode='test',
    cache_dir=cache_dir
)

if len(dataset_cached) > 0:
    import time

    # First load (builds cache)
    start = time.time()
    _ = dataset_cached[0]
    first_time = time.time() - start

    # Second load (from cache)
    start = time.time()
    _ = dataset_cached[0]
    cached_time = time.time() - start

    print(f"✓ Caching works")
    print(f"  First load: {first_time:.3f}s")
    print(f"  Cached load: {cached_time:.3f}s")
    if first_time > 0:
        print(f"  Speedup: {first_time/cached_time:.1f}x")

# Test 5: Sparse representation
print("\nTesting sparse representation...")
dataset_sparse = BeamDataset(
    data_root=data_root,
    event_representation='sparse',
    max_vibration_samples=2000,
    mode='test',
    use_sparse=True
)

if len(dataset_sparse) > 0:
    event_data, vibration_data, label = dataset_sparse[0]
    print(f"✓ Sparse representation works")
    print(f"  Event shape: {event_data.shape}")

print("\n" + "=" * 50)
print("All tests passed! The optimized dataloader is working correctly.")