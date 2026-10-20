"""
Test script for the 'all' mode in BeamDataset
Demonstrates loading all data splits at once for EDA
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from beam_dataloader import BeamDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def test_all_mode():
    """Test the 'all' mode functionality"""
    data_root = "/home/humaid/Projects/XJTU_DV_FaultDiagnosis/data/XJTU-DV-Beam"

    print("="*60)
    print("Testing 'all' Mode - Load Entire Dataset")
    print("="*60)

    # Create dataset with mode='all'
    dataset = BeamDataset(
        data_root=data_root,
        mode='all',
        event_representation='frame',
        max_vibration_samples=2000,
        cache_events=True
    )

    print(f"\nDataset in 'all' mode:")
    print(f"Total samples: {len(dataset)}")

    # Get all data with split labels
    events, vibrations, labels, split_labels = dataset.get_all_data_with_splits()

    print(f"\nData shapes:")
    print(f"  Events: {events.shape}")
    print(f"  Vibrations: {vibrations.shape}")
    print(f"  Labels: {labels.shape}")
    print(f"  Split labels: {split_labels.shape}")

    # Analyze split distribution
    print("\nSplit distribution:")
    split_names = ['Train', 'Val', 'Test']
    for i, name in enumerate(split_names):
        count = (split_labels == i).sum()
        percentage = count / len(split_labels) * 100
        print(f"  {name}: {count} samples ({percentage:.1f}%)")

    # Analyze labels per split
    print("\nLabel distribution per split:")
    for i, split_name in enumerate(split_names):
        split_mask = split_labels == i
        split_labels_subset = labels[split_mask]
        normal = (split_labels_subset == 0).sum().item()
        fault = (split_labels_subset == 1).sum().item()
        print(f"  {split_name}: Normal={normal}, Fault={fault}")

    return dataset, events, vibrations, labels, split_labels


def perform_eda(dataset, events, vibrations, labels, split_labels):
    """Perform exploratory data analysis on the full dataset"""
    print("\n" + "="*60)
    print("Exploratory Data Analysis")
    print("="*60)

    # Create a DataFrame for easier analysis
    data = []
    for i in range(len(dataset)):
        info = dataset.get_sample_info(i)
        name = info['name']
        parts = name.split('_')

        if len(parts) >= 3:
            illumination = 'Low' if parts[0] == 'Off' else 'High'
            camera_set = parts[1]
            trial = parts[2]

            if 'trail' in trial:
                trial_num = int(trial.replace('trail', ''))
            elif 'trai' in trial:
                trial_num = int(trial.replace('trai', ''))
            else:
                trial_num = 0

            split_name = ['Train', 'Val', 'Test'][split_labels[i]]

            data.append({
                'name': name,
                'illumination': illumination,
                'camera_set': camera_set,
                'trial': trial_num,
                'label': 'Normal' if labels[i] == 0 else 'Fault',
                'split': split_name,
                'event_mean': events[i].mean().item(),
                'event_std': events[i].std().item(),
                'vibration_mean': vibrations[i, :, 1].mean().item(),  # Vibration velocity
                'vibration_std': vibrations[i, :, 1].std().item(),
                'vibration_max': vibrations[i, :, 1].abs().max().item()
            })

    df = pd.DataFrame(data)

    # Print statistics
    print("\nDataset Statistics:")
    print(df.groupby(['split', 'illumination', 'camera_set']).size().unstack(fill_value=0))

    print("\nVibration Signal Statistics by Configuration:")
    stats = df.groupby(['illumination', 'camera_set'])[['vibration_mean', 'vibration_std', 'vibration_max']].mean()
    print(stats.round(6))

    print("\nEvent Statistics by Configuration:")
    event_stats = df.groupby(['illumination', 'camera_set'])[['event_mean', 'event_std']].mean()
    print(event_stats.round(4))

    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: Sample distribution
    ax = axes[0, 0]
    split_counts = df['split'].value_counts()
    ax.bar(split_counts.index, split_counts.values)
    ax.set_title('Samples per Split')
    ax.set_ylabel('Count')

    # Plot 2: Configuration distribution
    ax = axes[0, 1]
    config_counts = df.groupby(['illumination', 'camera_set']).size()
    ax.bar(range(len(config_counts)), config_counts.values)
    ax.set_xticks(range(len(config_counts)))
    ax.set_xticklabels([f"{k[0][:1]}-{k[1]}" for k in config_counts.index], rotation=45)
    ax.set_title('Samples per Configuration')
    ax.set_ylabel('Count')

    # Plot 3: Vibration statistics by illumination
    ax = axes[0, 2]
    vibration_by_illum = df.groupby('illumination')['vibration_max'].mean()
    ax.bar(vibration_by_illum.index, vibration_by_illum.values)
    ax.set_title('Mean Max Vibration by Illumination')
    ax.set_ylabel('Max Vibration (m/s)')

    # Plot 4: Event density by illumination
    ax = axes[1, 0]
    event_by_illum = df.groupby('illumination')['event_mean'].mean()
    ax.bar(event_by_illum.index, event_by_illum.values)
    ax.set_title('Mean Event Density by Illumination')
    ax.set_ylabel('Mean Event Value')

    # Plot 5: Trial distribution
    ax = axes[1, 1]
    trial_counts = df['trial'].value_counts().sort_index()
    ax.bar(trial_counts.index, trial_counts.values)
    ax.set_title('Samples per Trial Number')
    ax.set_xlabel('Trial')
    ax.set_ylabel('Count')

    # Plot 6: Correlation heatmap
    ax = axes[1, 2]
    corr_data = df[['event_mean', 'event_std', 'vibration_mean', 'vibration_std', 'vibration_max']].corr()
    sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Feature Correlations')

    plt.tight_layout()
    plt.savefig('/home/humaid/Projects/XJTU_DV_FaultDiagnosis/data_loaders/eda_analysis.png', dpi=150)
    plt.show()

    return df


def test_individual_splits():
    """Compare individual split loading vs all mode"""
    print("\n" + "="*60)
    print("Comparing Individual Splits vs All Mode")
    print("="*60)

    data_root = "/home/humaid/Projects/XJTU_DV_FaultDiagnosis/data/XJTU-DV-Beam"

    # Load individual splits
    train_dataset = BeamDataset(data_root=data_root, mode='train')
    val_dataset = BeamDataset(data_root=data_root, mode='val')
    test_dataset = BeamDataset(data_root=data_root, mode='test')
    all_dataset = BeamDataset(data_root=data_root, mode='all')

    print(f"\nIndividual splits:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    print(f"  Total: {len(train_dataset) + len(val_dataset) + len(test_dataset)} samples")

    print(f"\nAll mode:")
    print(f"  Total: {len(all_dataset)} samples")

    # Verify they match
    expected_total = len(train_dataset) + len(val_dataset) + len(test_dataset)
    assert len(all_dataset) == expected_total, "Mismatch in total samples!"
    print("\n✓ All mode correctly contains all samples from individual splits")


if __name__ == "__main__":
    # Test all mode functionality
    dataset, events, vibrations, labels, split_labels = test_all_mode()

    # Perform EDA
    df = perform_eda(dataset, events, vibrations, labels, split_labels)

    # Test individual splits comparison
    test_individual_splits()

    print("\n" + "="*60)
    print("All tests completed successfully!")
    print("="*60)

    # Save DataFrame for further analysis
    df.to_csv('/home/humaid/Projects/XJTU_DV_FaultDiagnosis/data_loaders/beam_dataset_analysis.csv', index=False)
    print("\nDataFrame saved to beam_dataset_analysis.csv for further analysis")