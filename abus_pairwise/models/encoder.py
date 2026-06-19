from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchvision.models import (
    DenseNet121_Weights,
    Inception_V3_Weights,
    ResNet50_Weights,
    densenet121,
    inception_v3,
    resnet50,
)


def _extract_state_dict(obj: Any) -> dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        for k in ("state_dict", "model", "net", "weights"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    raise ValueError("Cannot parse checkpoint format: expected a state_dict-like object.")


def _strip_prefixes(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for k, v in sd.items():
        nk = k
        for prefix in ("module.", "encoder.", "backbone.", "resnet.", "densenet.", "model."):
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        out[nk] = v
    return out


def _adapt_first_conv(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    if in_channels == conv.in_channels:
        return conv
    new_conv = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )
    with torch.no_grad():
        base_weight = conv.weight.mean(dim=1, keepdim=True)
        new_conv.weight.copy_(base_weight.repeat(1, in_channels, 1, 1))
        if conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(conv.bias)
    return new_conv


class MultiScaleEncoder(nn.Module):
    def __init__(self, name: str = "resnet50", pretrain_source: str = "imagenet", checkpoint_path: str | None = None, radimagenet_url: str | None = None, strict_load: bool = False, in_channels: int = 3):
        super().__init__()
        self.name = name.lower()
        source = pretrain_source.lower()
        if in_channels <= 0:
            raise ValueError("in_channels must be > 0")
        if self.name not in {"resnet50", "densenet121", "inceptionv3"}:
            raise ValueError(f"Unsupported encoder: {name}")
        if source not in {"imagenet", "radimagenet", "local", "none"}:
            raise ValueError(f"Unsupported pretrain source: {pretrain_source}")

        if self.name == "resnet50":
            base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if source in {"imagenet", "radimagenet"} else None)
            if source in {"radimagenet", "local"}:
                ckpt = self._load_external_ckpt(source, checkpoint_path, radimagenet_url)
                if ckpt is not None:
                    sd = _strip_prefixes(_extract_state_dict(ckpt))
                    sd = {k: v for k, v in sd.items() if not k.startswith("fc.")}
                    base.load_state_dict(sd, strict=strict_load)
            base.conv1 = _adapt_first_conv(base.conv1, in_channels)
            self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
            self.layer1, self.layer2, self.layer3, self.layer4 = base.layer1, base.layer2, base.layer3, base.layer4
            self.out_channels = [256, 512, 1024, 2048]
        elif self.name == "densenet121":
            base = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1 if source in {"imagenet", "radimagenet"} else None)
            if source in {"radimagenet", "local"}:
                ckpt = self._load_external_ckpt(source, checkpoint_path, radimagenet_url)
                if ckpt is not None:
                    sd = _strip_prefixes(_extract_state_dict(ckpt))
                    base.load_state_dict(sd, strict=strict_load)
            base.features.conv0 = _adapt_first_conv(base.features.conv0, in_channels)
            self.features = base.features
            self.out_channels = [256, 512, 1024, 1024]
        else:
            base = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1 if source in {"imagenet", "radimagenet"} else None, aux_logits=False)
            if source in {"radimagenet", "local"}:
                ckpt = self._load_external_ckpt(source, checkpoint_path, radimagenet_url)
                if ckpt is not None:
                    sd = _strip_prefixes(_extract_state_dict(ckpt))
                    sd = {k: v for k, v in sd.items() if not k.startswith("fc.") and not k.startswith("AuxLogits.")}
                    base.load_state_dict(sd, strict=strict_load)
            base.Conv2d_1a_3x3.conv = _adapt_first_conv(base.Conv2d_1a_3x3.conv, in_channels)
            self.inception = base
            self.out_channels = [192, 288, 768, 2048]

    def _load_external_ckpt(self, source: str, checkpoint_path: str | None, radimagenet_url: str | None):
        if source == "radimagenet" and checkpoint_path is None:
            url = (radimagenet_url or "").strip() or os.getenv("RADIMAGENET_URL", "").strip()
            if url:
                return torch.hub.load_state_dict_from_url(url, map_location="cpu", progress=True)
            return None
        if checkpoint_path is None:
            raise ValueError(f"{source} source requires --encoder-ckpt path.")
        return torch.load(Path(checkpoint_path), map_location="cpu")

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        if self.name == "resnet50":
            x = self.stem(x)
            f1 = self.layer1(x)
            f2 = self.layer2(f1)
            f3 = self.layer3(f2)
            f4 = self.layer4(f3)
            return [f1, f2, f3, f4]

        if self.name == "densenet121":
            x = self.features.conv0(x)
            x = self.features.norm0(x)
            x = self.features.relu0(x)
            x = self.features.pool0(x)
            x = self.features.denseblock1(x)
            x = self.features.transition1(x)
            f1 = x
            x = self.features.denseblock2(x)
            x = self.features.transition2(x)
            f2 = x
            x = self.features.denseblock3(x)
            x = self.features.transition3(x)
            f3 = x
            x = self.features.denseblock4(x)
            f4 = x
            return [f1, f2, f3, f4]

        # inceptionv3
        x = self.inception.Conv2d_1a_3x3(x)
        x = self.inception.Conv2d_2a_3x3(x)
        x = self.inception.Conv2d_2b_3x3(x)
        x = self.inception.maxpool1(x)
        x = self.inception.Conv2d_3b_1x1(x)
        x = self.inception.Conv2d_4a_3x3(x)
        x = self.inception.maxpool2(x)
        f1 = x
        x = self.inception.Mixed_5b(x)
        x = self.inception.Mixed_5c(x)
        x = self.inception.Mixed_5d(x)
        f2 = x
        x = self.inception.Mixed_6a(x)
        x = self.inception.Mixed_6b(x)
        x = self.inception.Mixed_6c(x)
        x = self.inception.Mixed_6d(x)
        x = self.inception.Mixed_6e(x)
        f3 = x
        x = self.inception.Mixed_7a(x)
        x = self.inception.Mixed_7b(x)
        x = self.inception.Mixed_7c(x)
        f4 = x
        return [f1, f2, f3, f4]
