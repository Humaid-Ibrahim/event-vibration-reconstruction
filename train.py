"""
Unified training script for Event-to-Vibration prediction models.

Usage:
    python train.py --model simple1dcnn
    python train.py --model cnn1d
    python train.py --model cnn1d_mlp
    python train.py --model cnn2d
    python train.py --model gnn
    python train.py --model cnn3d
    python train.py --model cnn3d_light
    python train.py --model cnn_lstm
    python train.py --model cnn_lstm_attention

    # With custom hyperparameters
    python train.py --model gnn --epochs 100 --lr 1e-4

    # Image sequence models with smaller resolution (recommended for faster training)
    python train.py --model cnn3d --image_seq_height 64 --image_seq_width 64 --image_seq_n_frames 64
    python train.py --model cnn_lstm --image_seq_height 128 --image_seq_width 128 --image_seq_n_frames 128
"""

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader


def get_git_info():
    """Get git commit info for reproducibility tracking."""
    try:
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        commit_short = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        # Check for uncommitted changes
        dirty = subprocess.call(
            ["git", "diff", "--quiet"], stderr=subprocess.DEVNULL
        ) != 0
        return {
            "commit_hash": commit_hash,
            "commit_short": commit_short,
            "branch": branch,
            "dirty": dirty,
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

from data_loaders.beam_dataloader_aligned import BeamDataset
from models import Simple1DCNN, Simple1DCNNAdaptive, CNN1D, CNN1D_MLP, CNN2D, CNN2D_SNN, CNN2D_SNN_EventRate, CNN3D, CNN3D_Light, CNN_LSTM, CNN_LSTM_Attention, GNNModel
from utils import Trainer, plot_predictions, plot_frequency_analysis, plot_training_curves


# Model configurations
MODEL_CONFIGS = {
    "simple1dcnn": {
        "class": Simple1DCNN,
        "event_representation": "event_rate",
        "roi_grid": (3, 3),
        "use_polarity": True,
        "batch_size": 16,
        "epochs": 1,
        "lr": 5e-4,
        "dropout": 0.5,
    },
    "simple1dcnn_adaptive": {
        "class": Simple1DCNNAdaptive,
        "event_representation": "event_rate",
        "roi_grid": (3, 3),
        "use_polarity": True,
        "batch_size": 256,  # Large batch for windowing
        "epochs": 50,
        "lr": 5e-4,
        "dropout": 0.3,  # Lower dropout for smaller model
    },
    "cnn1d": {
        "class": CNN1D,
        "event_representation": "event_rate",
        "roi_grid": (5, 5),  # Smaller grid - events are concentrated in center
        "use_polarity": True,  # Use polarity for richer features
        "batch_size": 16,  # Smaller batch for small dataset
        "epochs": 100,
        "lr": 5e-4,  # Lower LR for stability
        "dropout": 0.7,
    },
    "cnn1d_mlp": {
        "class": CNN1D_MLP,
        "event_representation": "event_rate",
        "roi_grid": (5, 5),
        "use_polarity": True,
        "batch_size": 16,
        "epochs": 100,
        "lr": 5e-4,
        "dropout": 0.5,
    },
    "cnn2d": {
        "class": CNN2D,
        "event_representation": "image_sequence",
        "roi_grid": None,  # Not used for image_sequence
        "use_polarity": True,
        "batch_size": 16,
        "epochs": 20,
        "lr": 1e-4,
        "dropout": 0.5,
    },
    "cnn2d_snn": {
        "class": CNN2D_SNN,
        "event_representation": "image_sequence",
        "roi_grid": None,  # Not used for image_sequence
        "use_polarity": True,
        "batch_size": 16,
        "epochs": 50,
        "lr": 1e-4,
        "dropout": 0.0,  # No dropout for SNN conversion
        # CRITICAL: Set these to 28x28 to avoid rendering huge images
        "image_seq_height": 28,
        "image_seq_width": 28,
        "image_seq_n_frames": 28,  # Fewer frames for speed
    },
    "cnn2d_snn_eventrate": {
        "class": CNN2D_SNN_EventRate,
        "event_representation": "event_rate",
        "roi_grid": (1, 1),  # 1x1 grid with polarity = 2 channels
        "use_polarity": True,
        "batch_size": 18,
        "epochs": 1,
        "lr": 5e-4,
        "dropout": 0.0,  # No dropout for SNN conversion
        "reshape_h": 32,  # Reshape 1024 -> (32, 32) SQUARE for Speck!
        "reshape_w": 32,
        "output_size": 32,  # REDUCED OUTPUT for Speck chip! (flattened_size=64)
        "n_vibration_samples": 32,  # Downsample ground truth to match model output
        "n_time_bins": 1024,  # REDUCED from 2048 for square 32x32 reshape
    },
    "gnn": {
        "class": GNNModel,
        "event_representation": "event_rate",
        "roi_grid": (5, 5),
        "use_polarity": True,
        "batch_size": 16,
        "epochs": 50,
        "lr": 5e-4,
        "dropout": 0.3,
        "embed_dim": 128,
        "gnn_hidden_dim": 128,
        "num_gnn_layers": 3,
        "neighborhood": "8",
        "use_attention": True,
    },
    # Image sequence models (CNN3D, CNN_LSTM)
    "cnn3d": {
        "class": CNN3D,
        "event_representation": "image_sequence",
        "roi_grid": None,  # Not used for image_sequence
        "use_polarity": True,  # 2 channels (pos/neg)
        "batch_size": 4,  # Smaller batch due to large input size
        "epochs": 50,
        "lr": 1e-4,
        "dropout": 0.5,
        "base_channels": 32,
        # Recommended: use smaller resolution for faster training
        # --image_seq_height 64 --image_seq_width 64 --image_seq_n_frames 64
    },
    "cnn3d_light": {
        "class": CNN3D_Light,
        "event_representation": "image_sequence",
        "roi_grid": None,
        "use_polarity": True,
        "batch_size": 8,
        "epochs": 50,
        "lr": 1e-4,
        "dropout": 0.6,
        "base_channels": 16,
    },
    "cnn_lstm": {
        "class": CNN_LSTM,
        "event_representation": "image_sequence",
        "roi_grid": None,
        "use_polarity": True,
        "batch_size": 4,
        "epochs": 50,
        "lr": 1e-4,
        "dropout": 0.5,
        "cnn_features": 128,
        "lstm_hidden": 256,
        "lstm_layers": 2,
        "bidirectional": True,
    },
    "cnn_lstm_attention": {
        "class": CNN_LSTM_Attention,
        "event_representation": "image_sequence",
        "roi_grid": None,
        "use_polarity": True,
        "batch_size": 4,
        "epochs": 50,
        "lr": 1e-4,
        "dropout": 0.5,
        "cnn_features": 128,
        "lstm_hidden": 256,
        "lstm_layers": 2,
        "bidirectional": True,
    },
}


def create_model(model_name, in_channels, config, output_size=2048):
    """Create model instance based on name and config.

    Args:
        model_name: Name of the model (key in MODEL_CONFIGS)
        in_channels: Number of input channels
        config: Model configuration dict
        output_size: Output sequence length (default 2048, or window_size if windowing enabled)
    """
    model_class = config["class"]

    if model_name == "gnn":
        return model_class(
            grid_shape=config["roi_grid"],
            channels_per_roi=2 if config["use_polarity"] else 1,
            output_size=output_size,
            embed_dim=config.get("embed_dim", 64),
            gnn_hidden_dim=config.get("gnn_hidden_dim", 128),
            num_gnn_layers=config.get("num_gnn_layers", 3),
            dropout=config["dropout"],
            neighborhood=config.get("neighborhood", "8"),
            use_attention=config.get("use_attention", True),
        )
    elif model_name == "cnn2d":
        return model_class(
            in_channels=in_channels,
            output_size=output_size,
            dropout=config["dropout"],
            roi_grid=config["roi_grid"],
            use_polarity=config["use_polarity"],
        )
    elif model_name == "cnn2d_snn":
        return model_class(
            output_size=output_size,
        )
    elif model_name == "cnn2d_snn_eventrate":
        return model_class(
            in_channels=in_channels,
            output_size=output_size,
            reshape_h=config.get("reshape_h", 32),
            reshape_w=config.get("reshape_w", 64),
        )
    elif model_name in ("cnn3d", "cnn3d_light"):
        return model_class(
            in_channels=2,  # pos/neg polarity channels
            output_size=output_size,
            base_channels=config.get("base_channels", 32),
            dropout=config["dropout"],
        )
    elif model_name in ("cnn_lstm", "cnn_lstm_attention"):
        return model_class(
            in_channels=2,  # pos/neg polarity channels
            output_size=output_size,
            cnn_features=config.get("cnn_features", 128),
            lstm_hidden=config.get("lstm_hidden", 256),
            lstm_layers=config.get("lstm_layers", 2),
            dropout=config["dropout"],
            bidirectional=config.get("bidirectional", True),
        )
    else:
        return model_class(
            in_channels=in_channels,
            output_size=output_size,
            dropout=config["dropout"],
        )


def create_dataloaders(
    config, data_root, num_workers=0, circular_shift_aug=False, bandpass_filter=None,
    image_seq_height=720, image_seq_width=1280, image_seq_n_frames=None,
    use_windowing=False, window_size=8, shuffle_windows=False
):
    """Create train/val/test dataloaders."""
    common_args = {
        "data_root": data_root,
        "event_representation": config["event_representation"],
        "cache_events": True,
        "bandpass_filter": bandpass_filter,
        "align": True,  # Always align events to vibration
        # Image sequence parameters (used when event_representation='image_sequence')
        "image_seq_height": image_seq_height,
        "image_seq_width": image_seq_width,
        "image_seq_n_frames": image_seq_n_frames,
        # Windowing parameters
        "use_windowing": use_windowing,
        "window_size": window_size,
    }

    if config.get("roi_grid"):
        common_args["roi_grid"] = config["roi_grid"]
        common_args["use_polarity_event_rate"] = config["use_polarity"]

    # Downsample vibration output if specified (for Speck chip deployment)
    if config.get("n_vibration_samples"):
        common_args["n_vibration_samples"] = config["n_vibration_samples"]

    # Set time bins if specified (for square 32x32 reshape with 1024 bins)
    if config.get("n_time_bins"):
        common_args["n_event_rate"] = config["n_time_bins"]
        print(f"Setting n_event_rate to {config['n_time_bins']} time bins")

    # Training dataset gets circular shift augmentation and window shuffling, val/test don't
    train_ds = BeamDataset(mode="train", circular_shift_aug=circular_shift_aug, shuffle_windows=shuffle_windows, **common_args)
    val_ds = BeamDataset(mode="val", circular_shift_aug=False, shuffle_windows=False, **common_args)
    test_ds = BeamDataset(mode="test", circular_shift_aug=False, shuffle_windows=False, **common_args)

    loader_args = {
        "batch_size": config["batch_size"],
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_args)

    return train_loader, val_loader, test_loader, train_ds


def main():
    parser = argparse.ArgumentParser(description="Train event-to-vibration model")
    parser.add_argument("--model", type=str, default="cnn1d_mlp",
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Model architecture to train")
    parser.add_argument("--data_root", type=str, default="data/XJTU-DV-Beam",
                        help="Path to dataset")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of epochs (overrides default)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (overrides default)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size (overrides default)")
    parser.add_argument("--dropout", type=float, default=None,
                        help="Dropout rate (overrides default)")
    parser.add_argument("--weight_decay", type=float, default=1e-3,
                        help="Weight decay")
    parser.add_argument("--mse_lambda", type=float, default=1.0,
                        help="MSE loss weight (default 1.0)")
    parser.add_argument("--smooth_lambda", type=float, default=0.0,
                        help="Smoothness regularization (penalizes high freq)")
    parser.add_argument("--spectral_lambda", type=float, default=0.0,
                        help="Spectral loss weight (helps capture frequency peaks)")
    parser.add_argument("--envelope_lambda", type=float, default=0.0,
                        help="Envelope loss weight (prevents amplitude underestimation)")
    parser.add_argument("--ssim_lambda", type=float, default=0.0,
                        help="SSIM loss weight (structural similarity)")
    parser.add_argument("--sdtw_lambda", type=float, default=0.0,
                        help="Soft-DTW loss weight (time series alignment)")
    parser.add_argument("--no_scaler", action="store_true",
                        help="Disable input/output scaling")
    parser.add_argument("--early_stopping_patience", type=int, default=15,
                        help="Number of epochs to wait before stopping (default: 15)")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers")
    parser.add_argument("--circular_shift_aug", action="store_true",
                        help="Enable circular shift augmentation (prevents learning absolute time)")
    parser.add_argument("--bandpass_low", type=float, default=None,
                        help="Bandpass filter low cutoff frequency (Hz)")
    parser.add_argument("--bandpass_high", type=float, default=None,
                        help="Bandpass filter high cutoff frequency (Hz)")
    # Image sequence representation options (for CNN/ViT models)
    parser.add_argument("--image_seq_height", type=int, default=720,
                        help="Image height for image_sequence representation (default 720, original camera resolution)")
    parser.add_argument("--image_seq_width", type=int, default=1280,
                        help="Image width for image_sequence representation (default 1280, original camera resolution)")
    parser.add_argument("--image_seq_n_frames", type=int, default=None,
                        help="Number of frames for image_sequence (default: n_event_rate=2048)")
    # Windowing options
    parser.add_argument("--use_windowing", action="store_true",
                        help="Enable windowing (split sequences into fixed-size windows)")
    parser.add_argument("--window_size", type=int, default=8,
                        help="Window size for windowing (must divide 2048 evenly, default: 8)")
    parser.add_argument("--shuffle_windows", action="store_true",
                        help="Shuffle windows during training (requires --use_windowing)")
    args = parser.parse_args()

    # Get config and apply overrides
    config = MODEL_CONFIGS[args.model].copy()
    if args.epochs:
        config["epochs"] = args.epochs
    if args.lr:
        config["lr"] = args.lr
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.dropout is not None:
        config["dropout"] = args.dropout

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Print header
    print("=" * 60)
    print(f"TRAINING: {args.model.upper()}")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Config: {json.dumps({k: v for k, v in config.items() if k != 'class'}, indent=2)}")

    # Parse bandpass filter
    bandpass_filter = None
    if args.bandpass_low is not None and args.bandpass_high is not None:
        bandpass_filter = (args.bandpass_low, args.bandpass_high)
        print(f"Bandpass filter: {args.bandpass_low}-{args.bandpass_high} Hz")
    if args.circular_shift_aug:
        print("Circular shift augmentation: ENABLED")
    if args.use_windowing:
        print(f"Windowing: ENABLED (window_size={args.window_size}, shuffle={args.shuffle_windows})")

    # Create dataloaders
    print("\n--- Loading Data ---")
    # Use config values if available, otherwise use args
    image_seq_height = config.get("image_seq_height", args.image_seq_height)
    image_seq_width = config.get("image_seq_width", args.image_seq_width)
    image_seq_n_frames = config.get("image_seq_n_frames", args.image_seq_n_frames)

    train_loader, val_loader, test_loader, train_ds = create_dataloaders(
        config, args.data_root, args.num_workers,
        circular_shift_aug=args.circular_shift_aug,
        bandpass_filter=bandpass_filter,
        image_seq_height=image_seq_height,
        image_seq_width=image_seq_width,
        image_seq_n_frames=image_seq_n_frames,
        use_windowing=args.use_windowing,
        window_size=args.window_size,
        shuffle_windows=args.shuffle_windows,
    )

    # Get input shape
    x0, y0, _ = train_ds[0]
    in_channels = x0.shape[0]
    output_size = y0.shape[0] if hasattr(y0, 'shape') else len(y0)

    # Override output_size if specified in config (for Speck chip deployment)
    if "output_size" in config:
        output_size = config["output_size"]
        print(f"NOTE: Overriding output_size to {output_size} (from config)")

    print(f"Input shape: {tuple(x0.shape)}")
    print(f"Output shape: {tuple(y0.shape) if hasattr(y0, 'shape') else y0}")
    print(f"Model output size: {output_size}")
    print(f"Train: {len(train_ds)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # Create model
    print("\n--- Creating Model ---")
    model = create_model(args.model, in_channels, config, output_size=output_size)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model}")
    print(f"Parameters: {num_params:,}")

    # Setup paths - each run gets a unique timestamped directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = Path("results") / args.model / run_id
    model_dir.mkdir(exist_ok=True, parents=True)

    checkpoint_path = model_dir / "best_model.pth"
    scaler_path = model_dir / "scalers.pkl"

    # Train
    print("\n--- Training ---")
    trainer = Trainer(
        model,
        device=device,
        use_scaler=not args.no_scaler,
        mse_lambda=args.mse_lambda,
        smooth_lambda=args.smooth_lambda,
        spectral_lambda=args.spectral_lambda,
        envelope_lambda=args.envelope_lambda,
        ssim_lambda=args.ssim_lambda,
        sdtw_lambda=args.sdtw_lambda,
    )
    best_val_loss = trainer.train(
        train_loader,
        val_loader,
        epochs=config["epochs"],
        lr=config["lr"],
        weight_decay=args.weight_decay,
        checkpoint_path=str(checkpoint_path),
        scaler_path=str(scaler_path),
        early_stopping_patience=args.early_stopping_patience,
    )

    # Load best model for evaluation
    trainer.load_checkpoint(checkpoint_path)

    # Evaluate
    fs = 2048 / 8.0  # 256 Hz sampling rate

    print("\n" + "=" * 60)
    print("VALIDATION SET EVALUATION")
    print("=" * 60)
    val_metrics, val_preds, val_targets = trainer.evaluate(val_loader)
    print(f"R2={val_metrics['r2']:.4f} MSE={val_metrics['mse']:.6f} MAE={val_metrics['mae']:.6f} DTW={val_metrics['dtw']:.4f}")

    plot_predictions(val_preds, val_targets, model_dir / "preds_val.png", fs=fs)
    plot_frequency_analysis(val_preds, val_targets, model_dir / "freq_val.png", fs=fs)

    # Save raw prediction values for debugging
    import numpy as np
    np.savez(
        model_dir / "predictions_val.npz",
        predictions=val_preds,
        targets=val_targets,
    )
    # Save as text file (predictions and ground truth side by side for each sample)
    with open(model_dir / "predictions_val.txt", "w") as f:
        f.write(f"# Validation predictions and ground truth\n")
        f.write(f"# Shape: {val_preds.shape[0]} samples x {val_preds.shape[1]} timesteps\n")
        f.write(f"# Format: sample_idx, timestep, prediction, ground_truth\n\n")
        for i in range(val_preds.shape[0]):
            f.write(f"# Sample {i}\n")
            for t in range(val_preds.shape[1]):
                f.write(f"{i},{t},{val_preds[i, t]:.6f},{val_targets[i, t]:.6f}\n")
    print(f"Saved val predictions: {val_preds.shape}, range=[{val_preds.min():.4f}, {val_preds.max():.4f}]")
    print(f"  Sample diffs: 0v1={np.abs(val_preds[0]-val_preds[1]).max():.6f}, 0v2={np.abs(val_preds[0]-val_preds[2]).max():.6f}")

    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    test_metrics, test_preds, test_targets = trainer.evaluate(test_loader)
    print(f"R2={test_metrics['r2']:.4f} MSE={test_metrics['mse']:.6f} MAE={test_metrics['mae']:.6f} DTW={test_metrics['dtw']:.4f}")

    plot_predictions(test_preds, test_targets, model_dir / "preds_test.png", fs=fs)
    plot_frequency_analysis(test_preds, test_targets, model_dir / "freq_test.png", fs=fs)

    # Save raw prediction values for debugging
    np.savez(
        model_dir / "predictions_test.npz",
        predictions=test_preds,
        targets=test_targets,
    )
    # Save as text file (predictions and ground truth side by side for each sample)
    with open(model_dir / "predictions_test.txt", "w") as f:
        f.write(f"# Test predictions and ground truth\n")
        f.write(f"# Shape: {test_preds.shape[0]} samples x {test_preds.shape[1]} timesteps\n")
        f.write(f"# Format: sample_idx, timestep, prediction, ground_truth\n\n")
        for i in range(test_preds.shape[0]):
            f.write(f"# Sample {i}\n")
            for t in range(test_preds.shape[1]):
                f.write(f"{i},{t},{test_preds[i, t]:.6f},{test_targets[i, t]:.6f}\n")
    print(f"Saved test predictions: {test_preds.shape}, range=[{test_preds.min():.4f}, {test_preds.max():.4f}]")
    print(f"  Sample diffs: 0v1={np.abs(test_preds[0]-test_preds[1]).max():.6f}, 0v2={np.abs(test_preds[0]-test_preds[2]).max():.6f}")

    # Training curves
    plot_training_curves(
        trainer.train_losses,
        trainer.val_losses,
        trainer.val_r2,
        val_metrics,
        test_metrics,
        model_dir / "training_curves.png"
    )

    # Summary
    print("\n" + "=" * 60)
    print(f"FINAL RESULTS ({args.model.upper()})")
    print("=" * 60)
    print(f"Best Validation Loss: {best_val_loss:.6f}")
    print(f"\nValidation: R²={val_metrics['r2']:.4f} MSE={val_metrics['mse']:.6f} MAE={val_metrics['mae']:.6f} DTW={val_metrics['dtw']:.4f}")
    print(f"Test:       R²={test_metrics['r2']:.4f} MSE={test_metrics['mse']:.6f} MAE={test_metrics['mae']:.6f} DTW={test_metrics['dtw']:.4f}")
    print("=" * 60)

    # Save comprehensive results JSON (MLflow-ready structure)
    results = {
        # Run identification
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "model_name": args.model,

        # Model architecture
        "model": {
            "name": args.model,
            "num_params": num_params,
            "architecture_config": {k: v for k, v in config.items() if k != "class"},
        },

        # Data configuration
        "data": {
            "data_root": args.data_root,
            "event_representation": config["event_representation"],
            "roi_grid": config.get("roi_grid"),
            "use_polarity": config.get("use_polarity", False),
            "train_samples": len(train_loader.dataset),
            "val_samples": len(val_loader.dataset),
            "test_samples": len(test_loader.dataset),
            "input_shape": list(x0.shape),
            "output_size": output_size,
            "circular_shift_aug": args.circular_shift_aug,
            "bandpass_filter": bandpass_filter,
            # Image sequence config (only relevant for image_sequence representation)
            "image_seq_height": args.image_seq_height,
            "image_seq_width": args.image_seq_width,
            "image_seq_n_frames": args.image_seq_n_frames,
            # Windowing config
            "use_windowing": args.use_windowing,
            "window_size": args.window_size,
            "shuffle_windows": args.shuffle_windows,
        },

        # Training hyperparameters
        "hyperparameters": {
            "epochs_configured": config["epochs"],
            "epochs_run": len(trainer.train_losses),
            "batch_size": config["batch_size"],
            "learning_rate": config["lr"],
            "weight_decay": args.weight_decay,
            "dropout": config.get("dropout", 0.0),
            "early_stopping_patience": args.early_stopping_patience,
            "use_scaler": not args.no_scaler,
        },

        # Loss configuration (critical for reproducibility)
        "loss_weights": {
            "mse_lambda": args.mse_lambda,
            "smooth_lambda": args.smooth_lambda,
            "spectral_lambda": args.spectral_lambda,
            "envelope_lambda": args.envelope_lambda,
            "ssim_lambda": args.ssim_lambda,
            "sdtw_lambda": args.sdtw_lambda,
        },

        # Training results
        "training": {
            "best_val_loss": float(best_val_loss),
            "best_epoch": trainer.train_losses.index(min(trainer.train_losses)) + 1 if trainer.train_losses else 0,
            "final_train_loss": float(trainer.train_losses[-1]) if trainer.train_losses else None,
            "final_val_loss": float(trainer.val_losses[-1]) if trainer.val_losses else None,
            "early_stopped": len(trainer.train_losses) < config["epochs"],
            "train_loss_history": [float(x) for x in trainer.train_losses],
            "val_loss_history": [float(x) for x in trainer.val_losses],
            "val_r2_history": [float(x) for x in trainer.val_r2],
        },

        # Evaluation metrics
        "metrics": {
            "validation": {k: float(v) for k, v in val_metrics.items()},
            "test": {k: float(v) for k, v in test_metrics.items()},
        },

        # Environment info (for reproducibility)
        "environment": {
            "device": device,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch_version": torch.__version__,
            "num_workers": args.num_workers,
            "git": get_git_info(),
        },

        # File paths (for artifact tracking)
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "scalers": str(scaler_path) if not args.no_scaler else None,
            "predictions_val": str(model_dir / "preds_val.png"),
            "predictions_test": str(model_dir / "preds_test.png"),
            "frequency_val": str(model_dir / "freq_val.png"),
            "frequency_test": str(model_dir / "freq_test.png"),
            "training_curves": str(model_dir / "training_curves.png"),
        },
    }

    with open(model_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {model_dir}/")


if __name__ == "__main__":
    main()
