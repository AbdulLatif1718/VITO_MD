# src/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------------
# 🔹 Basic Conv block: Conv + BN + SiLU (same as YOLO)
# -------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()  # YOLO-style

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# -------------------------------------------------------------
# 🔹 CSP Residual Block
# -------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = ConvBlock(channels, channels)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


# -------------------------------------------------------------
# 🔹 CBAM (Convolutional Block Attention Module)
# -------------------------------------------------------------
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )

        self.spatial = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        # Channel attention
        avg = self.shared_mlp(self.avg_pool(x))
        max_ = self.shared_mlp(self.max_pool(x))
        ch_attn = torch.sigmoid(avg + max_)
        x = x * ch_attn

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sp_attn = torch.sigmoid(self.spatial(torch.cat([avg_out, max_out], dim=1)))
        return x * sp_attn


# -------------------------------------------------------------
# 🔹 TransformerLite Block
# -------------------------------------------------------------
class TransformerLite(nn.Module):
    def __init__(self, dim, heads=4, ff_mult=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.ReLU(),
            nn.Linear(dim * ff_mult, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1).permute(0, 2, 1)  # (B, HW, C)
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        x = self.norm1(x_flat + attn_out)
        ff_out = self.norm2(x + self.ff(x))
        return ff_out.permute(0, 2, 1).view(B, C, H, W)


# -------------------------------------------------------------
# 🔹 Detection Head (YOLO-style)
# -------------------------------------------------------------
class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.head = nn.Sequential(
            ConvBlock(in_channels, in_channels),
            nn.Conv2d(in_channels, num_classes + 5, 1)  # [x, y, w, h, conf, cls...]
        )

    def forward(self, x):
        return self.head(x)


# -------------------------------------------------------------
# 🔹 Final Malaria Detection Model
# -------------------------------------------------------------
class MalariaNet(nn.Module):
    def __init__(self, num_classes=16):
        super().__init__()
        self.stem = ConvBlock(3, 32, stride=2)

        self.stage1 = nn.Sequential(
            ConvBlock(32, 64, stride=2),
            ResidualBlock(64),
            CBAM(64)
        )

        self.stage2 = nn.Sequential(
            ConvBlock(64, 128, stride=2),
            ResidualBlock(128),
            TransformerLite(128)
        )

        self.stage3 = nn.Sequential(
            ConvBlock(128, 256, stride=2),
            ResidualBlock(256),
            CBAM(256)
        )

        self.det_head = DetectionHead(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.det_head(x)
