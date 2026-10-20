"""
Loss functions for vibration signal reconstruction.

Includes:
- spectral_loss: FFT-based frequency domain matching
- envelope_loss: Amplitude envelope matching
- ssim_loss: 1D Structural Similarity Index
"""

import torch
import torch.nn.functional as F


def spectral_loss(pred, target, weight_high_freq=2.0):
    """
    Compute loss in frequency domain to capture spectral characteristics.

    Args:
        pred: Predicted signal (B, T)
        target: Target signal (B, T)
        weight_high_freq: Extra weight for higher frequencies (>50% of Nyquist)

    Returns:
        Combined magnitude and phase-aware spectral loss, normalized to be
        comparable in magnitude to MSE loss.
    """
    # Compute FFT
    pred_fft = torch.fft.rfft(pred, dim=-1)
    target_fft = torch.fft.rfft(target, dim=-1)

    # Normalize magnitudes by sequence length (Parseval's theorem)
    # This makes the spectral loss comparable to time-domain MSE
    seq_len = pred.shape[-1]
    pred_mag = torch.abs(pred_fft) / seq_len
    target_mag = torch.abs(target_fft) / seq_len
    mag_loss = F.mse_loss(pred_mag, target_mag)

    # Weight higher frequencies more (they're typically under-represented)
    n_freqs = pred_mag.shape[-1]
    freq_weights = torch.ones(n_freqs, device=pred.device)
    # Boost frequencies in upper half of spectrum
    freq_weights[n_freqs // 2:] = weight_high_freq

    weighted_mag_loss = (freq_weights * (pred_mag - target_mag).pow(2)).mean()

    return mag_loss + weighted_mag_loss


def envelope_loss(pred, target, window_size=64):
    """
    Compute envelope (amplitude) matching loss.

    This loss encourages the model to match the amplitude envelope of the signal,
    helping prevent amplitude underestimation that MSE tends to cause.

    The envelope is computed as the local RMS over sliding windows.

    Args:
        pred: Predicted signal (B, T)
        target: Target signal (B, T)
        window_size: Size of sliding window for envelope computation

    Returns:
        MSE between predicted and target envelopes
    """
    # Ensure window_size doesn't exceed signal length
    T = pred.shape[-1]
    window_size = min(window_size, T)

    # Compute squared signals
    pred_sq = pred.pow(2)
    target_sq = target.pow(2)

    # Use avg_pool1d to compute local mean of squared values (RMS envelope)
    # Need to add channel dimension for avg_pool1d: (B, T) -> (B, 1, T)
    pred_sq = pred_sq.unsqueeze(1)
    target_sq = target_sq.unsqueeze(1)

    # Compute local RMS with stride=window_size//2 for overlapping windows
    stride = max(window_size // 2, 1)
    pred_env = F.avg_pool1d(pred_sq, kernel_size=window_size, stride=stride, padding=0).sqrt()
    target_env = F.avg_pool1d(target_sq, kernel_size=window_size, stride=stride, padding=0).sqrt()

    # Also add global energy matching (overall amplitude)
    pred_rms = pred.pow(2).mean(dim=-1).sqrt()
    target_rms = target.pow(2).mean(dim=-1).sqrt()
    energy_loss = F.mse_loss(pred_rms, target_rms)

    # Combine local envelope and global energy
    local_env_loss = F.mse_loss(pred_env, target_env)

    return local_env_loss + energy_loss


def ssim_loss(pred, target, window_size=11, sigma=1.5):
    """
    Compute 1D Structural Similarity Index (SSIM) loss for time series.

    SSIM captures structural similarity by comparing local statistics (mean, variance,
    covariance) between predicted and target signals. This helps preserve local patterns
    and structure that MSE might miss.

    Args:
        pred: Predicted signal (B, T)
        target: Target signal (B, T)
        window_size: Size of Gaussian window for local statistics
        sigma: Standard deviation of Gaussian window

    Returns:
        1 - SSIM (so lower is better, like other losses)
    """
    # Create 1D Gaussian window
    coords = torch.arange(window_size, device=pred.device, dtype=pred.dtype) - window_size // 2
    gauss = torch.exp(-coords.pow(2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    window = gauss.view(1, 1, -1)  # (1, 1, window_size)

    # Add channel dimension: (B, T) -> (B, 1, T)
    pred = pred.unsqueeze(1)
    target = target.unsqueeze(1)

    # Compute local means
    mu_pred = F.conv1d(pred, window, padding=window_size // 2)
    mu_target = F.conv1d(target, window, padding=window_size // 2)

    # Compute local variances and covariance
    mu_pred_sq = mu_pred.pow(2)
    mu_target_sq = mu_target.pow(2)
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = F.conv1d(pred.pow(2), window, padding=window_size // 2) - mu_pred_sq
    sigma_target_sq = F.conv1d(target.pow(2), window, padding=window_size // 2) - mu_target_sq
    sigma_pred_target = F.conv1d(pred * target, window, padding=window_size // 2) - mu_pred_target

    # SSIM constants for stability
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # SSIM formula
    numerator = (2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)
    denominator = (mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2)
    ssim_map = numerator / denominator

    # Return 1 - SSIM so we can minimize it
    return 1 - ssim_map.mean()
