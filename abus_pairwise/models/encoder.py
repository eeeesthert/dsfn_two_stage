from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50MultiScale(nn.Module):
    """ResNet50 feature pyramid compatible with warp/fusion stages."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        base = resnet50(weights=weights)

        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        f1 = self.layer1(x)   # 1/4
        f2 = self.layer2(f1)  # 1/8
        f3 = self.layer3(f2)  # 1/16
        f4 = self.layer4(f3)  # 1/32
        return [f1, f2, f3, f4]
