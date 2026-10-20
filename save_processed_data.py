"""
Quick script to save processed and normalized train/test/val data.

This saves the exact data after:
- Event processing (event_rate representation)
- Normalization/scaling
- Ready to feed into model

Usage:
    python save_processed_data.py --model cnn2d_snn_eventrate --output data_processed.npz
"""

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from data_loaders.beam_dataloader_aligned import BeamDataset
from train import MODEL_CONFIGS


def save_processed_data(model_name, output_path, data_root="data/XJTU-DV-Beam", downsample_output=None):
    """Save processed train/val/test data to a single npz file.

    Args:
        model_name: Model name from MODEL_CONFIGS
        output_path: Output file path
        data_root: Path to dataset
        downsample_output: If specified, downsample vibration output to this size (e.g., 128)
    """

    # Get model config
    config = MODEL_CONFIGS[model_name].copy()

    print(f"Loading data for model: {model_name}")
    print(f"Config: {config}")

    # Create datasets
    common_args = {
        "data_root": data_root,
        "event_representation": config["event_representation"],
        "cache_events": True,
        "align": True,
    }

    if config.get("roi_grid"):
        common_args["roi_grid"] = config["roi_grid"]
        common_args["use_polarity_event_rate"] = config["use_polarity"]

    # Load datasets
    print("\nLoading train set...")
    train_ds = BeamDataset(mode="train", **common_args)
    print(f"Train samples: {len(train_ds)}")

    print("Loading val set...")
    val_ds = BeamDataset(mode="val", **common_args)
    print(f"Val samples: {len(val_ds)}")

    print("Loading test set...")
    test_ds = BeamDataset(mode="test", **common_args)
    print(f"Test samples: {len(test_ds)}")

    # Extract all data
    def extract_dataset(dataset):
        X, y, metadata = [], [], []
        for i in range(len(dataset)):
            x_i, y_i, meta_i = dataset[i]
            X.append(x_i.numpy())
            y.append(y_i.numpy())
            metadata.append(meta_i)
        return np.array(X), np.array(y), metadata

    print("\nExtracting train data...")
    X_train, y_train, meta_train = extract_dataset(train_ds)

    print("Extracting val data...")
    X_val, y_val, meta_val = extract_dataset(val_ds)

    print("Extracting test data...")
    X_test, y_test, meta_test = extract_dataset(test_ds)

    # Downsample outputs if requested (for Speck chip deployment)
    if downsample_output is not None:
        print(f"\nDownsampling vibration outputs from {y_train.shape[1]} -> {downsample_output}...")
        from scipy.interpolate import interp1d

        def downsample_batch(y_full, output_size):
            N, T_full = y_full.shape
            x_full = np.linspace(0, 1, T_full)
            x_down = np.linspace(0, 1, output_size)
            y_down = np.zeros((N, output_size), dtype=y_full.dtype)
            for i in range(N):
                interp = interp1d(x_full, y_full[i], kind='cubic')
                y_down[i] = interp(x_down)
            return y_down

        y_train_down = downsample_batch(y_train, downsample_output)
        y_val_down = downsample_batch(y_val, downsample_output)
        y_test_down = downsample_batch(y_test, downsample_output)

        # Use downsampled versions
        y_train_save = y_train_down
        y_val_save = y_val_down
        y_test_save = y_test_down
        print(f"  Downsampled to {downsample_output} samples")
    else:
        y_train_save = y_train
        y_val_save = y_val
        y_test_save = y_test

    # Save to npz (compressed)
    print(f"\nSaving to {output_path}...")
    np.savez_compressed(
        output_path,
        # Train
        X_train=X_train,
        y_train=y_train_save,
        # Val
        X_val=X_val,
        y_val=y_val_save,
        # Test
        X_test=X_test,
        y_test=y_test_save,
        # Metadata
        config=str(config),
        model_name=model_name,
    )

    # Print summary
    print("\n" + "="*60)
    print("SAVED DATA SUMMARY")
    print("="*60)
    print(f"Train: X={X_train.shape}, y={y_train_save.shape}")
    print(f"Val:   X={X_val.shape}, y={y_val_save.shape}")
    print(f"Test:  X={X_test.shape}, y={y_test_save.shape}")
    print(f"\nFile size: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")
    print(f"Saved to: {output_path}")
    print("="*60)

    # Show how to load
    print("\nTo load this data:")
    print(f"  data = np.load('{output_path}')")
    print("  X_train, y_train = data['X_train'], data['y_train']")
    print("  X_val, y_val = data['X_val'], data['y_val']")
    print("  X_test, y_test = data['X_test'], data['y_test']")


def main():
    parser = argparse.ArgumentParser(description="Save processed train/test/val data")
    parser.add_argument("--model", type=str, default="cnn2d_snn_eventrate",
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Model to extract data for")
    parser.add_argument("--output", type=str, default="processed_data.npz",
                        help="Output file path")
    parser.add_argument("--data_root", type=str, default="data/XJTU-DV-Beam",
                        help="Path to dataset")
    parser.add_argument("--downsample_output", type=int, default=None,
                        choices=[64, 128, 256, 512],
                        help="Downsample vibration output to this size (for Speck chip)")
    args = parser.parse_args()

    save_processed_data(args.model, args.output, args.data_root, args.downsample_output)


if __name__ == "__main__":
    main()
