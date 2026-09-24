from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def pad_to_multiple8(image: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    h, w = image.shape[-2:]
    dh, dw = (-h) % 8, (-w) % 8
    padding = (dw // 2, dw - dw // 2, dh // 2, dh - dh // 2)
    return F.pad(image, padding), padding


def unpad(image: torch.Tensor, padding: tuple[int, int, int, int]) -> torch.Tensor:
    left, right, top, bottom = padding
    h, w = image.shape[-2:]
    return image[..., top : h - bottom if bottom else h, left : w - right if right else w]


def load_checkpoint_validated(model: torch.nn.Module, checkpoint: str | Path, device: str | torch.device) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location=device)
    state = payload.get("state_dict", payload.get("model", payload)) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise RuntimeError("Checkpoint does not contain a state dictionary")
    normalized = OrderedDict((key.removeprefix("module."), value) for key, value in state.items())
    incompatible = model.load_state_dict(normalized, strict=False)
    loaded = len(model.state_dict()) - len(incompatible.missing_keys)
    print(f"Loaded params: {loaded}/{len(model.state_dict())}")
    print(f"Missing keys: {list(incompatible.missing_keys)}")
    print(f"Unexpected keys: {list(incompatible.unexpected_keys)}")
    critical_prefixes = ("fnet.", "cnet.", "update_block.")
    critical_missing = [key for key in incompatible.missing_keys if key.startswith(critical_prefixes)]
    if critical_missing or loaded < 0.9 * len(model.state_dict()):
        raise RuntimeError(f"Checkpoint is incompatible with RAFTStitch; critical missing keys: {critical_missing}")
    return {"loaded": loaded, "missing": list(incompatible.missing_keys), "unexpected": list(incompatible.unexpected_keys)}
