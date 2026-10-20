"""
Test script demonstrating how to get all data at once from the dataset
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from beam_dataloader import BeamDataset
import numpy as np
import torch

# Create dataset
data_root = "/home/humaid/Projects/XJTU_DV_FaultDiagnosis/data/XJTU-DV-Beam"

print("Creating dataset...")
dataset = BeamDataset(
    data_root=data_root,
    event_representation='frame',
    max_vibration_samples=2000,  # Limit for consistency
    mode='train',  # Can be 'train', 'val', or 'test'
    cache_events=True  # Enable memory caching for faster access
)

print(f"Dataset mode: {dataset.mode}")
print(f"Dataset size: {len(dataset)}")

# Method 1: Get all data as PyTorch tensors
print("\n" + "="*50)
print("Method 1: Get all data as PyTorch tensors")
print("="*50)

events_tensor, vibrations_tensor, labels_tensor = dataset.get_all_data()

print(f"Events shape: {events_tensor.shape}")
print(f"Vibrations shape: {vibrations_tensor.shape}")
print(f"Labels shape: {labels_tensor.shape}")
print(f"Data types: {events_tensor.dtype}, {vibrations_tensor.dtype}, {labels_tensor.dtype}")

# Method 2: Get all data as NumPy arrays
print("\n" + "="*50)
print("Method 2: Get all data as NumPy arrays")
print("="*50)

events_np, vibrations_np, labels_np = dataset.get_all_data_numpy()

print(f"Events shape: {events_np.shape}")
print(f"Vibrations shape: {vibrations_np.shape}")
print(f"Labels shape: {labels_np.shape}")
print(f"Data types: {events_np.dtype}, {vibrations_np.dtype}, {labels_np.dtype}")

# Method 3: Get data with metadata
print("\n" + "="*50)
print("Method 3: Get data organized by mode with metadata")
print("="*50)

data_dict = dataset.get_data_by_mode()

print(f"Mode: {data_dict['mode']}")
print(f"Number of samples: {data_dict['n_samples']}")
print(f"Normal samples: {data_dict['n_normal']}")
print(f"Fault samples: {data_dict['n_fault']}")
print(f"Events shape: {data_dict['events'].shape}")
print(f"Vibrations shape: {data_dict['vibrations'].shape}")
print(f"Labels shape: {data_dict['labels'].shape}")

# Example: Access specific samples from the loaded data
print("\n" + "="*50)
print("Example: Accessing specific samples")
print("="*50)

# Get first sample
first_event = events_tensor[0]
first_vibration = vibrations_tensor[0]
first_label = labels_tensor[0]

print(f"First sample:")
print(f"  Event shape: {first_event.shape}")
print(f"  Vibration shape: {first_vibration.shape}")
print(f"  Label: {first_label} ({'Normal' if first_label == 0 else 'Fault'})")

# Get all normal samples
normal_mask = labels_tensor == 0
normal_events = events_tensor[normal_mask]
normal_vibrations = vibrations_tensor[normal_mask]

print(f"\nNormal samples:")
print(f"  Count: {normal_mask.sum()}")
print(f"  Events shape: {normal_events.shape}")
print(f"  Vibrations shape: {normal_vibrations.shape}")

# Get all fault samples
fault_mask = labels_tensor == 1
fault_events = events_tensor[fault_mask]
fault_vibrations = vibrations_tensor[fault_mask]

print(f"\nFault samples:")
print(f"  Count: {fault_mask.sum()}")
print(f"  Events shape: {fault_events.shape}")
print(f"  Vibrations shape: {fault_vibrations.shape}")

# Example: Use for training without DataLoader
print("\n" + "="*50)
print("Example: Direct training without DataLoader")
print("="*50)

# Simulate a simple training step
batch_size = 4
n_samples = len(dataset)

# Random permutation for shuffling
perm = torch.randperm(n_samples)

# Get a batch
batch_indices = perm[:batch_size]
batch_events = events_tensor[batch_indices]
batch_vibrations = vibrations_tensor[batch_indices]
batch_labels = labels_tensor[batch_indices]

print(f"Batch shapes:")
print(f"  Events: {batch_events.shape}")
print(f"  Vibrations: {batch_vibrations.shape}")
print(f"  Labels: {batch_labels.shape}")

# Memory usage estimate
print("\n" + "="*50)
print("Memory Usage Estimate")
print("="*50)

def get_tensor_size_mb(tensor):
    """Calculate tensor size in MB"""
    if isinstance(tensor, dict):
        return sum(t.element_size() * t.nelement() / (1024**2) for t in tensor.values())
    else:
        return tensor.element_size() * tensor.nelement() / (1024**2)

events_size = get_tensor_size_mb(events_tensor)
vibrations_size = get_tensor_size_mb(vibrations_tensor)
labels_size = get_tensor_size_mb(labels_tensor)
total_size = events_size + vibrations_size + labels_size

print(f"Events memory: {events_size:.2f} MB")
print(f"Vibrations memory: {vibrations_size:.2f} MB")
print(f"Labels memory: {labels_size:.2f} MB")
print(f"Total memory: {total_size:.2f} MB")

print("\n" + "="*50)
print("Note: get_all_data() loads everything into memory at once.")
print("For large datasets, consider using DataLoader for batched loading.")
print("="*50)