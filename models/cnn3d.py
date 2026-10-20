"""
3D CNN model for event-to-vibration prediction.

Processes image sequences from event cameras using 3D convolutions.
Input: (B, T, C, H, W) where T=n_frames, C=2 (pos/neg polarity), H=height, W=width
Output: (B, output_size) vibration signal (default: 2048 samples)
"""

import torch
import torch.nn as nn


class CNN3D(nn.Module):
    """
    3D CNN for processing temporal sequences of event images.

    Architecture:
    - 3D convolutions to capture spatio-temporal patterns
    - Progressive downsampling in space and time
    - FC layers to map to vibration signal

    Input: (B, T, C, H, W) - batch of image sequences
        - T: number of frames (time bins)
        - C: 2 channels (positive and negative polarity)
        - H, W: image height and width

    Output: (B, output_size) - predicted vibration signal
    """

    def __init__(
        self,
        in_channels: int = 2,
        n_frames: int = 64,
        image_height: int = 64,
        image_width: int = 64,
        output_size: int = 2048,
        base_channels: int = 32,
        dropout: float = 0.5,
    ):
        """
        Args:
            in_channels: Number of input channels (2 for pos/neg polarity)
            n_frames: Number of input frames (time dimension)
            image_height: Height of input images
            image_width: Width of input images
            output_size: Size of output vibration signal
            base_channels: Base number of channels (doubled at each block)
            dropout: Dropout rate
        """
        super().__init__()

        self.in_channels = in_channels
        self.n_frames = n_frames
        self.image_height = image_height
        self.image_width = image_width
        self.output_size = output_size

        # 3D Conv blocks: (B, C, T, H, W) format
        # Note: PyTorch Conv3d expects (B, C, D, H, W) where D is depth (time)
        self.encoder = nn.Sequential(
            # Block 1: (2, T, H, W) -> (32, T/2, H/4, W/4)
            nn.Conv3d(in_channels, base_channels, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),

            # Block 2: (32, T/2, H/4, W/4) -> (64, T/4, H/8, W/8)
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),

            # Block 3: (64, T/4, H/8, W/8) -> (128, T/8, H/16, W/16)
            nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),

            # Block 4: (128, T/8, H/16, W/16) -> (256, T/16, H/32, W/32)
            nn.Conv3d(base_channels * 4, base_channels * 8, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(base_channels * 8),
            nn.ReLU(inplace=True),

            # Global average pooling
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )

        # Decoder: map features to vibration signal
        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base_channels * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, output_size),
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, T, C, H, W)
               - B: batch size
               - T: number of frames
               - C: number of channels (2)
               - H: image height
               - W: image width

        Returns:
            Output tensor of shape (B, output_size)
        """
        # Input: (B, T, C, H, W)
        # Conv3d expects: (B, C, D, H, W) where D is depth (time)
        # So we permute to (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)

        # Encode
        z = self.encoder(x)  # (B, base_channels*8, 1, 1, 1)

        # Decode to vibration
        return self.decoder(z)


class CNN3D_Light(nn.Module):
    """
    Lightweight 3D CNN for smaller image sequences.

    Uses fewer parameters and is suitable for:
    - Smaller image resolutions (e.g., 64x64 or 128x128)
    - Fewer frames (e.g., 32-128 frames)
    - Quick experimentation
    """

    def __init__(
        self,
        in_channels: int = 2,
        output_size: int = 2048,
        base_channels: int = 16,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv3d(in_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),

            # Block 2
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True),

            # Block 3
            nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(inplace=True),

            # Global pooling
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )

        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base_channels * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, output_size),
        )

    def forward(self, x):
        # Input: (B, T, C, H, W) -> (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        z = self.encoder(x)
        return self.decoder(z)
