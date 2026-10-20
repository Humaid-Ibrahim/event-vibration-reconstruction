"""
CNN-LSTM hybrid model for event-to-vibration prediction.

Combines CNN for spatial feature extraction with LSTM for temporal modeling.
Input: (B, T, C, H, W) where T=n_frames, C=2 (pos/neg polarity), H=height, W=width
Output: (B, output_size) vibration signal (default: 2048 samples)
"""

import torch
import torch.nn as nn


class CNN_LSTM(nn.Module):
    """
    CNN-LSTM hybrid for processing temporal sequences of event images.

    Architecture:
    - 2D CNN encoder processes each frame independently to extract spatial features
    - LSTM processes the sequence of frame features to capture temporal dependencies
    - FC decoder maps LSTM output to vibration signal

    Input: (B, T, C, H, W) - batch of image sequences
        - T: number of frames (time bins)
        - C: 2 channels (positive and negative polarity)
        - H, W: image height and width

    Output: (B, output_size) - predicted vibration signal
    """

    def __init__(
        self,
        in_channels: int = 2,
        output_size: int = 2048,
        cnn_features: int = 128,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.5,
        bidirectional: bool = True,
    ):
        """
        Args:
            in_channels: Number of input channels (2 for pos/neg polarity)
            output_size: Size of output vibration signal
            cnn_features: Number of features output by CNN encoder per frame
            lstm_hidden: Number of hidden units in LSTM
            lstm_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()

        self.in_channels = in_channels
        self.output_size = output_size
        self.cnn_features = cnn_features
        self.lstm_hidden = lstm_hidden
        self.bidirectional = bidirectional

        # 2D CNN encoder for spatial features
        # Processes each frame independently
        self.cnn_encoder = nn.Sequential(
            # Block 1: (C, H, W) -> (32, H/4, W/4)
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: (32, H/4, W/4) -> (64, H/8, W/8)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: (64, H/8, W/8) -> (128, H/16, W/16)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Global average pooling -> (128, 1, 1)
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),

            # Project to cnn_features
            nn.Linear(128, cnn_features),
            nn.ReLU(inplace=True),
        )

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=cnn_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Output size from LSTM
        lstm_out_size = lstm_hidden * (2 if bidirectional else 1)

        # Decoder: map LSTM features to vibration signal
        self.decoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out_size, 512),
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
        B, T, C, H, W = x.shape

        # Process each frame through CNN encoder
        # Reshape to (B*T, C, H, W) for batch processing
        x_flat = x.view(B * T, C, H, W)

        # CNN encoding: (B*T, C, H, W) -> (B*T, cnn_features)
        cnn_out = self.cnn_encoder(x_flat)

        # Reshape back to sequence: (B*T, cnn_features) -> (B, T, cnn_features)
        cnn_out = cnn_out.view(B, T, -1)

        # LSTM processing: (B, T, cnn_features) -> (B, T, lstm_hidden*directions)
        lstm_out, _ = self.lstm(cnn_out)

        # Use the last hidden state for prediction
        # For bidirectional: concatenate forward and backward last states
        if self.bidirectional:
            # lstm_out[:, -1, :] is already the concatenation of both directions
            final_features = lstm_out[:, -1, :]
        else:
            final_features = lstm_out[:, -1, :]

        # Decode to vibration signal
        return self.decoder(final_features)


class CNN_LSTM_Attention(nn.Module):
    """
    CNN-LSTM with attention mechanism for better temporal modeling.

    Uses attention over LSTM outputs instead of just the final hidden state.
    """

    def __init__(
        self,
        in_channels: int = 2,
        output_size: int = 2048,
        cnn_features: int = 128,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.5,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.output_size = output_size
        self.bidirectional = bidirectional

        # CNN encoder (same as CNN_LSTM)
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, cnn_features),
            nn.ReLU(inplace=True),
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        lstm_out_size = lstm_hidden * (2 if bidirectional else 1)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(lstm_out_size, lstm_out_size // 4),
            nn.Tanh(),
            nn.Linear(lstm_out_size // 4, 1),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, output_size),
        )

    def forward(self, x):
        B, T, C, H, W = x.shape

        # CNN encoding
        x_flat = x.view(B * T, C, H, W)
        cnn_out = self.cnn_encoder(x_flat)
        cnn_out = cnn_out.view(B, T, -1)

        # LSTM
        lstm_out, _ = self.lstm(cnn_out)  # (B, T, lstm_out_size)

        # Attention weights
        attn_scores = self.attention(lstm_out)  # (B, T, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (B, T, 1)

        # Weighted sum of LSTM outputs
        context = (lstm_out * attn_weights).sum(dim=1)  # (B, lstm_out_size)

        # Decode
        return self.decoder(context)
