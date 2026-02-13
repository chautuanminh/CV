"""
TrackNet — Ball detection via heatmap regression on consecutive frames.

Architecture: VGG16-style encoder → transposed-convolution decoder.
Input:  3 consecutive RGB frames stacked → (batch, 9, 360, 640)
Output: heatmap (batch, 1, 360, 640) with Gaussian at ball center.

Reference: https://nol.cs.nctu.edu.tw:234/open-source/TrackNet
"""

import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    """Two consecutive Conv-BN-ReLU layers."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class _TripleConvBlock(nn.Module):
    """Three consecutive Conv-BN-ReLU layers (used in deeper encoder stages)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class TrackNet(nn.Module):
    """
    TrackNet model for small-ball heatmap detection.

    Args:
        num_input_frames: Number of consecutive frames (default 3 → 9 channels).
        dropout:          Dropout probability in the decoder.
    """

    def __init__(self, num_input_frames: int = 3, dropout: float = 0.1):
        super().__init__()
        in_channels = num_input_frames * 3

        # ── Encoder (VGG16-style) ──
        self.enc1 = _ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc2 = _ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.enc3 = _TripleConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.enc4 = _TripleConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.enc5 = _TripleConvBlock(512, 512)

        # ── Decoder (transpose convolutions) ──
        self.up1 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.dec1 = _TripleConvBlock(1024, 512)
        self.drop1 = nn.Dropout2d(dropout)

        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec2 = _TripleConvBlock(768, 256)
        self.drop2 = nn.Dropout2d(dropout)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = _ConvBlock(384, 128)
        self.drop3 = nn.Dropout2d(dropout)

        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec4 = _ConvBlock(192, 64)

        self.final_conv = nn.Conv2d(64, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        e5 = self.enc5(self.pool4(e4))

        # Decoder with skip connections
        d1 = self.drop1(self.dec1(torch.cat([self.up1(e5), e4], dim=1)))
        d2 = self.drop2(self.dec2(torch.cat([self.up2(d1), e3], dim=1)))
        d3 = self.drop3(self.dec3(torch.cat([self.up3(d2), e2], dim=1)))
        d4 = self.dec4(torch.cat([self.up4(d3), e1], dim=1))

        return self.sigmoid(self.final_conv(d4))

    def detect_ball(self, heatmap: torch.Tensor, threshold: float = 0.5):
        """
        Extract ball (x, y) position from a predicted heatmap.

        Args:
            heatmap:    Model output tensor (1, 1, H, W) or (1, H, W).
            threshold:  Minimum confidence to consider a detection.

        Returns:
            (x, y) tuple or None if no ball detected.
        """
        heatmap = heatmap.squeeze().cpu().numpy()
        max_val = heatmap.max()
        if max_val < threshold:
            return None
        y, x = divmod(heatmap.argmax(), heatmap.shape[1])
        return int(x), int(y)
