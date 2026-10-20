"""
Visualization utilities for model evaluation.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


def plot_predictions(pred, targ, outpath, fs=256.0, num_samples=6):
    """
    Plot time-domain predictions vs actual signals.

    Args:
        pred: Predictions array (N, T)
        targ: Targets array (N, T)
        outpath: Output file path
        fs: Sampling frequency in Hz
        num_samples: Number of samples to plot
    """
    Path(outpath).parent.mkdir(exist_ok=True, parents=True)

    n = min(num_samples, pred.shape[0])
    t = np.arange(pred.shape[1]) / fs

    plt.figure(figsize=(12, 4 * n))
    for i in range(n):
        plt.subplot(n, 1, i + 1)
        plt.plot(t, targ[i], linewidth=0.8, alpha=0.7, label="Actual")
        plt.plot(t, pred[i], linewidth=0.8, alpha=0.7, label="Predicted")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.title(f"Sample {i} | R2={r2_score(targ[i], pred[i]):.4f}")

    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_frequency_analysis(pred, targ, outpath, fs=256.0, num_samples=6):
    """
    Plot frequency analysis comparing predicted vs actual signals.

    Creates two plots:
        1. Individual samples with time and frequency domain side by side
        2. Averaged frequency spectrum across all samples

    Args:
        pred: Predictions array (N, T)
        targ: Targets array (N, T)
        outpath: Output file path
        fs: Sampling frequency in Hz
        num_samples: Number of individual samples to plot
    """
    Path(outpath).parent.mkdir(exist_ok=True, parents=True)

    n = min(num_samples, pred.shape[0])
    signal_len = pred.shape[1]
    freqs = np.fft.rfftfreq(signal_len, d=1.0/fs)

    # Individual samples plot
    plt.figure(figsize=(14, 5 * n))

    for i in range(n):
        fft_pred = np.abs(np.fft.rfft(pred[i])) / signal_len
        fft_targ = np.abs(np.fft.rfft(targ[i])) / signal_len

        # Time domain
        plt.subplot(n, 2, 2*i + 1)
        t = np.arange(signal_len) / fs
        plt.plot(t, targ[i], linewidth=0.8, alpha=0.7, label="Actual")
        plt.plot(t, pred[i], linewidth=0.8, alpha=0.7, label="Predicted")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title(f"Sample {i} - Time Domain | R2={r2_score(targ[i], pred[i]):.4f}")
        plt.legend()
        plt.grid(alpha=0.3)

        # Frequency domain
        plt.subplot(n, 2, 2*i + 2)
        plt.plot(freqs, fft_targ, linewidth=1.0, alpha=0.7, label="Actual")
        plt.plot(freqs, fft_pred, linewidth=1.0, alpha=0.7, label="Predicted")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.title(f"Sample {i} - Frequency Domain")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.xlim([0, fs/2])

    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

    # Averaged frequency spectrum plot
    avg_fft_pred = np.mean([
        np.abs(np.fft.rfft(pred[i])) / signal_len
        for i in range(pred.shape[0])
    ], axis=0)
    avg_fft_targ = np.mean([
        np.abs(np.fft.rfft(targ[i])) / signal_len
        for i in range(targ.shape[0])
    ], axis=0)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(freqs, avg_fft_targ, linewidth=1.5, alpha=0.8, label="Actual (avg)")
    plt.plot(freqs, avg_fft_pred, linewidth=1.5, alpha=0.8, label="Predicted (avg)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("Average Frequency Spectrum")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xlim([0, fs/2])

    plt.subplot(1, 2, 2)
    plt.semilogy(freqs, avg_fft_targ, linewidth=1.5, alpha=0.8, label="Actual (avg)")
    plt.semilogy(freqs, avg_fft_pred, linewidth=1.5, alpha=0.8, label="Predicted (avg)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (log)")
    plt.title("Average Frequency Spectrum (Log Scale)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xlim([0, fs/2])

    plt.tight_layout()
    avg_outpath = str(outpath).replace(".png", "_avg.png")
    plt.savefig(avg_outpath, dpi=150)
    plt.close()


def plot_training_curves(train_losses, val_losses, val_r2, val_metrics, test_metrics, outpath):
    """
    Plot training curves and final metrics comparison.

    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        val_r2: List of validation R² scores per epoch
        val_metrics: Final validation metrics dict
        test_metrics: Final test metrics dict
        outpath: Output file path
    """
    Path(outpath).parent.mkdir(exist_ok=True, parents=True)

    plt.figure(figsize=(14, 5))

    # Loss curves
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # R² curve
    plt.subplot(1, 3, 2)
    plt.plot(val_r2, label='Val R²', linewidth=2, color='green')
    plt.axhline(y=val_metrics['r2'], color='g', linestyle='--', alpha=0.5,
                label=f'Final Val R²={val_metrics["r2"]:.4f}')
    plt.axhline(y=test_metrics['r2'], color='r', linestyle='--', alpha=0.5,
                label=f'Test R²={test_metrics["r2"]:.4f}')
    plt.xlabel('Epoch')
    plt.ylabel('R² Score')
    plt.title('Validation R² Over Training')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Metrics comparison
    plt.subplot(1, 3, 3)
    x = np.arange(2)
    width = 0.35
    plt.bar(x - width/2, [val_metrics['mse'], val_metrics['mae']], width,
            label='Validation', alpha=0.8)
    plt.bar(x + width/2, [test_metrics['mse'], test_metrics['mae']], width,
            label='Test', alpha=0.8)
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.title('Val vs Test Metrics')
    plt.xticks(x, ['MSE', 'MAE'])
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
