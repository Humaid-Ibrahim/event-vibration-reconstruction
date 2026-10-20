"""
2D CNN model for SNN conversion using event_rate representation.

This model reshapes 1D event_rate data to 2D for Conv2D processing.
Keeps the exact same nn.Sequential building blocks as cnn2d_snn.py

Architecture requirements for dynapcnn:
- NO bias in Conv2d and Linear layers
- NO BatchNorm, NO Dropout
- ReLU activations only
- AvgPool2d for downsampling

Input: (B, C, 2048) - event_rate with C channels (e.g., 18 for 3x3 grid with polarity)
Output: (B, 2048) - vibration signal prediction
"""

import torch
import torch.nn as nn


def create_cnn2d_snn_eventrate_model(in_channels: int, output_size: int = 32, reshape_h: int = 32, reshape_w: int = 32):
    """
    Create ULTRA-TINY 2D CNN model for Speck chip - SQUARE INPUT for maximum compatibility.

    Reshapes (C, 1024) → (C, H, W) where H*W = 1024
    Default: (C, 32, 32) since 32*32 = 1024

    Architecture for Speck chip (SQUARE INPUT + SQUARE KERNELS):
    - Conv2d(in_channels, 8, 3, 1, bias=False) -> ReLU -> AvgPool2d(4, 4)
    - Conv2d(8, 16, 3, 1, bias=False) -> ReLU -> AvgPool2d(4, 4)
    - Flatten -> Linear(32, output_size, bias=False)

    Dimension tracking for (C, 32, 32):
    - Input: (C, 32, 32)
    - After conv1 + pool1(4x4): (8, 8, 8)
    - After conv2 + pool2(4x4): (16, 2, 2)
    - After flatten: 16*2*2 = 64
    - After linear: output_size (default 32)

    Speck chip constraints:
    - Max input dimension: 128x128 ✓ (we use 32x32)
    - SQUARE input dimensions ✓ (32x32)
    - SQUARE pooling kernels ✓ (4x4, 4x4)
    - Flattened size: 64 < 100 ✓
    - Linear: 64→32 = 2,048 params

    Key optimizations:
    - SQUARE input (32x32) for Speck compatibility
    - Square pooling kernels (4x4, 4x4)
    - Flattened size = 64 (well under 100 guideline!)
    - Linear: 64→32 (ultra-compact representation)

    Args:
        in_channels: Number of input channels (e.g., 18 for 3x3 grid with polarity)
        output_size: Output dimension (default 32)
        reshape_h: Height to reshape to (default 32)
        reshape_w: Width to reshape to (default 32)

    Returns:
        nn.Sequential model
    """
    # Calculate flattened size with SQUARE dimensions: 32 -> 8 -> 2
    # After pool1(4x4): H=8, W=8
    # After pool2(4x4): H=2, W=2
    # Flatten: 16 * 2 * 2 = 64
    flattened_size = 64

    model = nn.Sequential(
        # Layer 1: (in_channels, 32, 32) -> (8, 8, 8)
        nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1, bias=False),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=4, stride=4),  # 32->8

        # Layer 2: (8, 8, 8) -> (16, 2, 2)
        nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, bias=False),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=4, stride=4),  # 8->2

        # Flatten: (16, 2, 2) -> 64
        nn.Flatten(),

        # FC layer: 64 -> output_size (default 32)
        nn.Linear(flattened_size, output_size, bias=False),
    )

    return model


class CNN2D_SNN_EventRate(nn.Module):
    """
    2D CNN that processes event_rate data (1D) by reshaping to 2D.

    Input: (B, C, 1024) - event_rate with C channels (REDUCED from 2048 for square reshape)
    Output: (B, output_size) - vibration prediction

    Process:
    1. Reshape (B, C, 1024) -> (B, C, H, W) where H*W = 1024
    2. Process through 2D CNN
    3. Output (B, output_size)
    """

    def __init__(self, in_channels: int, output_size: int = 32, reshape_h: int = 32, reshape_w: int = 32):
        super().__init__()
        assert reshape_h * reshape_w == 1024, f"reshape_h * reshape_w must equal 1024, got {reshape_h * reshape_w}"

        self.in_channels = in_channels
        self.output_size = output_size
        self.reshape_h = reshape_h
        self.reshape_w = reshape_w

        self.model = create_cnn2d_snn_eventrate_model(in_channels, output_size, reshape_h, reshape_w)

    def forward(self, x):
        """
        Args:
            x: (B, C, 1024) - event_rate data
        Returns:
            (B, output_size)
        """
        B, C, T = x.shape
        assert T == 1024, f"Expected temporal dimension of 1024, got {T}"

        # Reshape (B, C, 1024) -> (B, C, H, W)
        x = x.view(B, C, self.reshape_h, self.reshape_w)

        # Process through 2D CNN
        return self.model(x)


if __name__ == "__main__":
    # Test the model
    print("Testing CNN2D_SNN_EventRate model...")

    # Test with event_rate data - SQUARE INPUT for Speck chip!
    in_channels = 18  # 3x3 grid with polarity
    output_size = 32  # Reduced from 2048 to fit Speck chip
    model = CNN2D_SNN_EventRate(in_channels=in_channels, output_size=output_size, reshape_h=32, reshape_w=32)

    x = torch.randn(4, 18, 1024)  # Batch=4, Channels=18, Time=1024 (SQUARE reshape!)
    print(f"Input:  {tuple(x.shape)}")

    y = model(x)
    print(f"Output: {tuple(y.shape)}")

    assert y.shape == (4, output_size), f"Expected (4, {output_size}), got {y.shape}"
    print(f"[PASS] Event rate test passed! Output size: {output_size} (64x smaller than original 2048)")

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")

    # Verify no bias
    has_bias = any('bias' in n for n, p in model.named_parameters())
    if has_bias:
        print("[FAIL] Model has bias parameters!")
    else:
        print("[PASS] No bias parameters - ready for SNN conversion!")

    print("\n" + "="*60)
    print("Model architecture:")
    print(model.model)
