# models/vitomd.py

import torch
import torch.nn as nn
import torchvision.models as models


class ConvBlock(nn.Module):
    """Simple Conv + BN + SiLU block"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )

    def forward(self, x):
        return self.conv(x)


class DetectionHead(nn.Module):
    """
    A basic YOLO-style detection head:
    - Assumes anchor-free objectness prediction
    - Outputs: (B, S, S, num_classes + 5) → [x, y, w, h, obj, class scores...]
    """
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(256, num_classes + 5, 1)
        )

    def forward(self, x):
        return self.head(x)


class ViTOMD(nn.Module):
    """
    ViTO_MD Backbone + Head
    - Feature extractor: ResNet18 pretrained backbone
    - Detection head: Custom YOLO-like head
    """
    def __init__(self, num_classes=16):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)

        # Remove classifier head
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # Output (B, 512, H/32, W/32)

        self.neck = nn.Sequential(
            ConvBlock(512, 256, 3, 1, 1),
            ConvBlock(256, 128, 3, 1, 1)
        )

        self.head = DetectionHead(128, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x


if __name__ == "__main__":
    model = ViTOMD(num_classes=16)
    dummy_input = torch.randn(1, 3, 640, 640)
    out = model(dummy_input)
    print("Output shape:", out.shape)  # Expected: [1, num_classes+5, H', W']
