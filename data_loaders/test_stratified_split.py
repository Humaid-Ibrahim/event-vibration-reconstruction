"""
Test the stratified split implementation for the Beam dataset
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from beam_dataloader import BeamDataset
import pandas as pd

def analyze_split(dataset, mode):
    """Analyze the composition of a dataset split"""
    print(f"\n{'='*50}")
    print(f"{mode.upper()} SET ANALYSIS")
    print(f"{'='*50}")
    print(f"Total samples: {len(dataset)}")

    # Collect all sample names
    samples = []
    for i in range(len(dataset)):
        info = dataset.get_sample_info(i)
        samples.append(info['name'])

    # Create a dataframe for analysis
    data = []
    for name in samples:
        parts = name.split('_')
        if len(parts) >= 3:
            illumination = parts[0]
            camera_set = parts[1]
            trial = parts[2]

            # Handle typo
            if 'trail' in trial:
                trial_num = int(trial.replace('trail', ''))
            elif 'trai' in trial:
                trial_num = int(trial.replace('trai', ''))
            else:
                trial_num = 0

            data.append({
                'name': name,
                'illumination': 'Low' if illumination == 'Off' else 'High',
                'camera_set': camera_set,
                'trial': trial_num,
                'config': f"{illumination}_{camera_set}"
            })

    df = pd.DataFrame(data)

    # Print trial distribution
    print("\nTrials in this split:")
    trials = sorted(df['trial'].unique())
    print(f"  Trials: {trials}")

    # Print configuration distribution
    print("\nConfiguration distribution:")
    config_counts = df.groupby(['illumination', 'camera_set']).size()
    for (illum, cam_set), count in config_counts.items():
        print(f"  {illum} illumination, {cam_set}: {count} samples")

    # Print detailed breakdown
    print("\nDetailed breakdown by configuration and trial:")
    pivot = df.pivot_table(
        index=['illumination', 'camera_set'],
        columns='trial',
        aggfunc='size',
        fill_value=0
    )
    print(pivot)

    return df


def main():
    data_root = "/home/humaid/Projects/XJTU_DV_FaultDiagnosis/data/XJTU-DV-Beam"

    print("Testing Stratified Split Implementation")
    print("="*60)
    print("\nDataset Configuration:")
    print("- 2 illumination conditions: Low (Off), High (On)")
    print("- 3 camera settings: set1, set2, set3")
    print("- 5 trials per configuration")
    print("- Total: 30 samples (2 x 3 x 5)")
    print("\nSplit Strategy:")
    print("- Train: trials 1, 2, 3 (18 samples, 60%)")
    print("- Val: trial 4 (6 samples, 20%)")
    print("- Test: trial 5 (6 samples, 20%)")

    # Create datasets for each split
    train_dataset = BeamDataset(
        data_root=data_root,
        mode='train',
        event_representation='frame'
    )

    val_dataset = BeamDataset(
        data_root=data_root,
        mode='val',
        event_representation='frame'
    )

    test_dataset = BeamDataset(
        data_root=data_root,
        mode='test',
        event_representation='frame'
    )

    # Analyze each split
    train_df = analyze_split(train_dataset, 'train')
    val_df = analyze_split(val_dataset, 'val')
    test_df = analyze_split(test_dataset, 'test')

    # Verify no overlap between splits
    print("\n" + "="*50)
    print("SPLIT VERIFICATION")
    print("="*50)

    train_names = set(train_df['name'])
    val_names = set(val_df['name'])
    test_names = set(test_df['name'])

    print(f"\nTotal unique samples: {len(train_names | val_names | test_names)}")
    print(f"Train-Val overlap: {len(train_names & val_names)} samples")
    print(f"Train-Test overlap: {len(train_names & test_names)} samples")
    print(f"Val-Test overlap: {len(val_names & test_names)} samples")

    if len(train_names & val_names) == 0 and \
       len(train_names & test_names) == 0 and \
       len(val_names & test_names) == 0:
        print("\n✓ No overlap between splits - Data integrity maintained!")
    else:
        print("\n✗ WARNING: Overlap detected between splits!")

    # Verify all configurations are present in each split
    print("\n" + "="*50)
    print("CONFIGURATION COVERAGE")
    print("="*50)

    all_configs = set(['Off_set1', 'Off_set2', 'Off_set3',
                      'On_set1', 'On_set2', 'On_set3'])

    for split_name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        configs = set(df['config'].unique())
        print(f"\n{split_name} configurations: {sorted(configs)}")
        if configs == all_configs:
            print(f"✓ {split_name} has all 6 configurations")
        else:
            missing = all_configs - configs
            print(f"✗ {split_name} missing: {missing}")


if __name__ == "__main__":
    main()