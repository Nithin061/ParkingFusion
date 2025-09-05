"""UNet for Free Space Segmentation (returns torchvision-style dict)

Usage:
    from models.unet import UNetSeg
    model = UNetSeg(in_channels=2, num_classes=1, base_ch=32)
    y = model(x)["out"]  # logits

This wrapper mimics torchvision segmentation API so your train loop
with model(x)["out"] works for FCN/DeepLab and UNet alike.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "UNetSeg",
    "UNetCore",
    "DoubleConv",
    "Down",
    "Up",
    "OutConv",
]


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad if odd dims
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        if diffX != 0 or diffY != 0:
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNetCore(nn.Module):
    def __init__(self, in_channels: int = 2, num_classes: int = 1, base_ch: int = 32):
        super().__init__()
        # encoder
        self.inc = DoubleConv(in_channels, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.down3 = Down(base_ch * 4, base_ch * 8)
        # bottleneck
        self.bot = DoubleConv(base_ch * 8, base_ch * 16)
        # decoder (note: concat doubles channels)
        self.up3 = Up(base_ch * 16 + base_ch * 8, base_ch * 8)
        self.up2 = Up(base_ch * 8 + base_ch * 4, base_ch * 4)
        self.up1 = Up(base_ch * 4 + base_ch * 2, base_ch * 2)
        self.up0 = Up(base_ch * 2 + base_ch, base_ch)
        self.outc = OutConv(base_ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.inc(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        xb = self.bot(x3)
        y3 = self.up3(xb, x3)
        y2 = self.up2(y3, x2)
        y1 = self.up1(y2, x1)
        y0 = self.up0(y1, x0)
        logits = self.outc(y0)
        return logits


class UNetSeg(nn.Module):
    """Wrapper to return {"out": logits} like torchvision segmentation models."""

    def __init__(self, in_channels: int = 2, num_classes: int = 1, base_ch: int = 32):
        super().__init__()
        self.net = UNetCore(in_channels, num_classes, base_ch)

    def forward(self, x: torch.Tensor) -> dict:
        logits = self.net(x)
        return {"out": logits}
