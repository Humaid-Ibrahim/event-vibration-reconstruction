"""
Lightweight 1D CNN optimized for windowed input (small sequence lengths).

For window_size=8:
- Input: (B, 50, 8) - 50 channels (5x5 grid * 2 polarity), 8 timesteps
- Output: (B, 8) - 8 timestep predictions

Key differences from standard Simple1DCNN:
- Smaller kernel sizes (3 instead of 9) - appropriate for short sequences
- Fewer channels (32→16→8 instead of 64→32→16→8) - less over-parameterization
- Designed for window_size in range [8, 64]
"""

import torch
import torch.nn as nn


class Simple1DCNNWindowed(nn.Module):
    """
    Lightweight 1D CNN for windowed sequences.

    Optimized for window_size=8 to 64.
    Uses smaller kernels and fewer channels than the standard version.
    """

    def __init__(self, in_channels: int, output_size: int = 8, dropout: float = 0.3):
        super().__init__()

        # Adapt architecture based on output size
        if output_size <= 8:
            # Very small windows: use kernel_size=3
            kernel_size = 3
            hidden_dims = [32, 16, 8]
        elif output_size <= 32:
            # Medium windows: use kernel_size=5
            kernel_size = 5
            hidden_dims = [48, 24, 12]
        else:
            # Larger windows: use kernel_size=7
            kernel_size = 7
            hidden_dims = [64, 32, 16]

        padding = kernel_size // 2

        self.network = nn.Sequential(
            # Layer 1: in_channels -> hidden_dims[0]
            nn.Conv1d(in_channels, hidden_dims[0], kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            # Layer 2: hidden_dims[0] -> hidden_dims[1]
            nn.Conv1d(hidden_dims[0], hidden_dims[1], kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            # Layer 3: hidden_dims[1] -> hidden_dims[2]
            nn.Conv1d(hidden_dims[1], hidden_dims[2], kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.ReLU(inplace=True),

            # Final layer: hidden_dims[2] -> 1
            nn.Conv1d(hidden_dims[2], 1, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, T) input event rate where T = window_size
        Returns:
            (B, T) vibration signal
        """
        y = self.network(x)  # (B, 1, T)
        return y.squeeze(1)  # (B, T)


class Simple1DCNNAdaptive(nn.Module):
    """
    Adaptive 1D CNN that adjusts architecture based on sequence length.

    Works for both windowed (8-64) and full sequences (2048).
    """

    def __init__(self, in_channels: int, output_size: int = 2048, dropout: float = 0.3):
        super().__init__()

        self.output_size = output_size

        # Choose architecture based on sequence length
        if output_size <= 16:
            # Tiny sequences (8-16): very lightweight
            self.network = nn.Sequential(
                nn.Conv1d(in_channels, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(dropout),

                nn.Conv1d(32, 16, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(16),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(dropout),

                nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(8),
                nn.LeakyReLU(inplace=True),

                nn.Conv1d(8, 1, kernel_size=3, stride=1, padding=1),
            )
        elif output_size <= 64:
            # Small sequences (17-64): lightweight
            self.network = nn.Sequential(
                nn.Conv1d(in_channels, 48, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm1d(48),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(dropout),

                nn.Conv1d(48, 24, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm1d(24),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(dropout),

                nn.Conv1d(24, 12, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm1d(12),
                nn.LeakyReLU(inplace=True),

                nn.Conv1d(12, 1, kernel_size=5, stride=1, padding=2),
            )
        else:
            # Full sequences (2048): standard architecture
            self.network = nn.Sequential(
                nn.Conv1d(in_channels, 64, kernel_size=9, stride=1, padding=4),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(dropout),

                nn.Conv1d(64, 32, kernel_size=9, stride=1, padding=4),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(dropout),

                nn.Conv1d(32, 16, kernel_size=9, stride=1, padding=4),
                nn.BatchNorm1d(16),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(dropout),

                nn.Conv1d(16, 8, kernel_size=9, stride=1, padding=4),
                nn.BatchNorm1d(8),
                nn.LeakyReLU(inplace=True),

                nn.Conv1d(8, 1, kernel_size=9, stride=1, padding=4),
            )

    def forward(self, x):
        """
        Args:
            x: (B, C, T) input event rate
        Returns:
            (B, T) vibration signal
        """
        y = self.network(x)  # (B, 1, T)
        return y.squeeze(1)  # (B, T)


if __name__ == "__main__":
    # Test with different window sizes
    import torch

    print("Testing Simple1DCNNWindowed:")

    # Test window_size=8
    model_8 = Simple1DCNNWindowed(in_channels=50, output_size=8)
    x_8 = torch.randn(16, 50, 8)
    y_8 = model_8(x_8)
    print(f"  window_size=8:  input={x_8.shape}, output={y_8.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model_8.parameters()):,}")

    # Test window_size=32
    model_32 = Simple1DCNNWindowed(in_channels=50, output_size=32)
    x_32 = torch.randn(16, 50, 32)
    y_32 = model_32(x_32)
    print(f"  window_size=32: input={x_32.shape}, output={y_32.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model_32.parameters()):,}")

    print("\nTesting Simple1DCNNAdaptive:")

    # Test adaptive model with different sizes
    for size in [8, 16, 32, 64, 2048]:
        model = Simple1DCNNAdaptive(in_channels=50, output_size=size)
        x = torch.randn(4, 50, size)
        y = model(x)
        params = sum(p.numel() for p in model.parameters())
        print(f"  size={size:4d}: input={tuple(x.shape)}, output={tuple(y.shape)}, params={params:,}")
