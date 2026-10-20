"""
2D CNN model for SNN conversion (dynapcnn format).

Architecture requirements for dynapcnn:
- NO bias in Conv2d and Linear layers
- NO BatchNorm, NO Dropout
- ReLU activations only
- AvgPool2d for downsampling

Input: (B, n_frames, 2, 32, 32) - image sequence with 2 polarity channels
Output: (B, 2048) - vibration signal prediction

This model uses nn.Sequential to match dynapcnn requirements exactly.
"""

import torch
import torch.nn as nn


def create_cnn2d_snn_model(output_size: int = 2048):
    """
    Create 2D CNN model as nn.Sequential for SNN conversion.

    Architecture:
    - Conv2d(2, 20, 5, 1, bias=False) -> ReLU -> AvgPool2d(2, 2)
    - Conv2d(20, 32, 5, 1, bias=False) -> ReLU -> AvgPool2d(2, 2)
    - Conv2d(32, 128, 3, 1, bias=False) -> ReLU -> AvgPool2d(2, 2)
    - Flatten -> Linear(2048, 500, bias=False) -> ReLU -> Linear(500, output_size, bias=False)

    Input shape: (B, 2, 32, 32) per frame
    Output shape: (B, output_size)

    Dimension tracking:
    - Input: (2, 32, 32)
    - After conv1 + pool1: (20, 16, 16)
    - After conv2 + pool2: (32, 8, 8)
    - After conv3 + pool3: (128, 4, 4)
    - After flatten: 2048
    - After linear1: 500
    - After linear2: output_size

    Args:
        output_size: Output dimension (default 2048 for full sequence)

    Returns:
        nn.Sequential model
    """
    model = nn.Sequential(
        # Layer 1: (2, 32, 32) -> (20, 16, 16)
        nn.Conv2d(2, 20, kernel_size=5, stride=1, padding=2, bias=False),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),

        # Layer 2: (20, 16, 16) -> (32, 8, 8)
        nn.Conv2d(20, 32, kernel_size=5, stride=1, padding=2, bias=False),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),

        # Layer 3: (32, 8, 8) -> (128, 4, 4)
        nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1, bias=False),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),

        # Flatten: (128, 4, 4) -> 2048
        nn.Flatten(),

        # FC layers: 2048 -> 500 -> output_size
        nn.Linear(1152, 500, bias=False),
        nn.ReLU(),
        nn.Linear(500, output_size, bias=False),
    )

    return model


class CNN2D_SNN(nn.Module):
    """
    Simple wrapper that just uses the nn.Sequential model directly.
    No frame reshaping - trains like a normal 2D CNN.

    Input: (B, 2, 32, 32) - batch of 2D images
    Output: (B, output_size) - vibration prediction
    """

    def __init__(self, output_size: int = 2048):
        super().__init__()
        self.model = create_cnn2d_snn_model(output_size)
        self.output_size = output_size

    def forward(self, x):
        """
        Args:
            x: (B, n_frames, 2, 32, 32) or (B, 2, 32, 32)
        Returns:
            (B, output_size)
        """
        # Handle image sequences by processing each frame and averaging
        if x.ndim == 5:
            B, T, C, H, W = x.shape
            # Reshape to (B*T, C, H, W) to process all frames
            x = x.view(B * T, C, H, W)
            # Process through model: (B*T, output_size)
            out = self.model(x)
            # Reshape back to (B, T, output_size)
            out = out.view(B, T, self.output_size)
            # Average across frames: (B, output_size)
            return out.mean(dim=1)
        else:
            # Single frame: (B, C, H, W)
            return self.model(x)


if __name__ == "__main__":
    # Test the model
    print("Testing CNN2D_SNN model...")

    # Test with nn.Sequential directly
    model = create_cnn2d_snn_model(output_size=2048)
    x = torch.randn(4, 2, 32, 32)
    y = model(x)
    print(f"nn.Sequential: input={tuple(x.shape)}, output={tuple(y.shape)}")

    # Test with wrapper class
    model_wrapper = CNN2D_SNN(output_size=2048)
    y_wrapper = model_wrapper(x)
    print(f"CNN2D_SNN wrapper: input={tuple(x.shape)}, output={tuple(y_wrapper.shape)}")

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")
