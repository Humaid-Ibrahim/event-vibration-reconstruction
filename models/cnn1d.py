"""
1D CNN models for event-rate to vibration prediction.

Input: (B, C, T) where C = num_channels, T = sequence length (2048)
Output: (B, T) vibration signal
"""

import torch
import torch.nn as nn


class Simple1DCNN(nn.Module):
    def __init__(self, in_channels: int, output_size: int = 2048, dropout: float = 0.1):
        super().__init__()

        # All convolutions use stride=1 to maintain sequence length
        # Progressively reduce channels: in_channels -> 1
        self.network = nn.Sequential(
            # Layer 1: in_channels -> 64
            nn.Conv1d(in_channels, 64, kernel_size=9, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            # Layer 2: 64 -> 32
            nn.Conv1d(64, 32, kernel_size=9, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            # Layer 3: 32 -> 16
            nn.Conv1d(32, 16, kernel_size=9, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            # Layer 4: 16 -> 8
            nn.Conv1d(16, 8, kernel_size=9, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),

            # Final layer: 8 -> 1 (output)
            nn.Conv1d(8, 1, kernel_size=9, stride=1, padding=4, bias=False),
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, 2048) input event rate
        Returns:
            (B, 2048) vibration signal
        """
        y = self.network(x)  # (B, 1, 2048)
        return y.squeeze(1)  # (B, 2048)


class CNN1D(nn.Module):
    """
    1D CNN with transposed convolution decoder.
    Maps (C, 2048) -> (2048).
    """

    def __init__(self, in_channels: int, output_size: int = 2048, dropout: float = 0.1):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(inplace=True),

            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4),   # 2048 -> 1024
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),

            nn.Conv1d(64, 128, kernel_size=9, stride=2, padding=4),  # 1024 -> 512
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),

            nn.Conv1d(128, 256, kernel_size=9, stride=2, padding=4), # 512 -> 256
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),

            nn.Dropout(dropout),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),  # 256 -> 512
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),   # 512 -> 1024
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),

            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),    # 1024 -> 2048
            nn.BatchNorm1d(32),
            nn.LeakyReLU(inplace=True),

            nn.Conv1d(32, 1, kernel_size=1),
        )

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y.squeeze(1)


class CNN1D_MLP(nn.Module):
    """
    1D CNN encoder + MLP decoder.
    Lighter than transposed conv decoder.
    Maps (C, 2048) -> (2048).
    """

    def __init__(self, in_channels: int, output_size: int = 2048, dropout: float = 0.1):
        super().__init__()

        self.output_size = output_size

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(inplace=True),

            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4),   # 2048 -> 1024
            nn.BatchNorm1d(64),
            nn.LeakyReLU(inplace=True),

            nn.Conv1d(64, 128, kernel_size=9, stride=2, padding=4),  # 1024 -> 512
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),

            nn.Conv1d(128, 256, kernel_size=9, stride=2, padding=4), # 512 -> 256
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
        )

        # After encoder: (B, 256, 256) -> flatten to (B, 65536)
        bottleneck_size = 256 * 256

        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_size, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(1024, output_size),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
