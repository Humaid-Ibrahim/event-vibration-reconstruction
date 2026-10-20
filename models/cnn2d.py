"""
2D CNN model for event-to-vibration prediction.

Treats ROI-based event rate data as a 2D spatial-temporal representation.
Input: (B, C, T) where C = num_ROIs * channels_per_ROI, T = 2048
       Reshaped internally to (B, channels_per_ROI, num_ROIs, T) for 2D convolution
Output: (B, T) vibration signal
"""

import torch
import torch.nn as nn


class CNN2D(nn.Module):
    """
    2D CNN that treats ROI event rates as spatial-temporal data.

    The input (B, C, T) is reshaped to (B, polarity_channels, H*W, T)
    where H*W is the ROI grid flattened, creating a 2D representation
    with ROIs as one spatial axis and time as the other.
    """

    def __init__(
        self,
        in_channels: int,
        output_size: int = 2048,
        dropout: float = 0.5,
        roi_grid: tuple = (5, 5),
        use_polarity: bool = True,
    ):
        super().__init__()

        self.roi_grid = roi_grid
        self.num_rois = roi_grid[0] * roi_grid[1]
        self.channels_per_roi = 2 if use_polarity else 1

        # Input will be reshaped to (B, channels_per_roi, num_rois, T)
        # This creates a 2D "image" where height=num_rois, width=T

        self.encoder = nn.Sequential(
            # Block 1: (channels_per_roi, 25, 2048) -> (32, 25, 512)
            nn.Conv2d(self.channels_per_roi, 32, kernel_size=(3, 7), stride=(1, 4), padding=(1, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Block 2: (32, 25, 512) -> (64, 25, 128)
            nn.Conv2d(32, 64, kernel_size=(3, 5), stride=(1, 4), padding=(1, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Block 3: (64, 25, 128) -> (128, 25, 32)
            nn.Conv2d(64, 128, kernel_size=(3, 5), stride=(1, 4), padding=(1, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Block 4: (128, 25, 32) -> (256, 1, 1)
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, output_size),
        )

    def forward(self, x):
        B, C, T = x.shape

        # Reshape from (B, num_rois * channels_per_roi, T)
        # to (B, channels_per_roi, num_rois, T)
        x = x.view(B, self.num_rois, self.channels_per_roi, T)
        x = x.permute(0, 2, 1, 3)  # (B, channels_per_roi, num_rois, T)

        z = self.encoder(x)
        return self.decoder(z)
