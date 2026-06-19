from __future__ import annotations

import random
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F

from .datasets import ABUSPairDataset

TemporalStackMode = Literal["stack", "mean", "max", "center"]


class ABUSTemporalPairDataset(ABUSPairDataset):
    """
    Pair dataset with a fixed neighboring-slice window for each view.

    ``left`` and ``right`` are always the center RGB slices used as the single
    stitching target/output. ``left_context`` and ``right_context`` contain the
    model input context. With ``radius=5`` and ``stack_mode='stack'``, each
    context tensor has 33 channels: 11 RGB frames concatenated by channel.
    """

    def __init__(
        self,
        root: str | Path,
        stage: str = "12",
        image_size: int | None = 512,
        radius: int = 5,
        stack_mode: TemporalStackMode = "stack",
        augment: bool = False,
        hflip_prob: float = 0.0,
        brightness_jitter: float = 0.0,
        contrast_jitter: float = 0.0,
    ):
        super().__init__(root=root, stage=stage, image_size=image_size, augment=False)
        if radius < 0:
            raise ValueError("radius must be >= 0")
        if stack_mode not in {"stack", "mean", "max", "center"}:
            raise ValueError(f"Unsupported temporal stack mode: {stack_mode}")
        self.radius = radius
        self.stack_mode: TemporalStackMode = stack_mode
        self.augment = augment
        self.hflip_prob = hflip_prob
        self.brightness_jitter = brightness_jitter
        self.contrast_jitter = contrast_jitter
        self._neighbor_cache: dict[Path, list[Path]] = {}

    @property
    def context_channels(self) -> int:
        if self.stack_mode == "stack":
            return (2 * self.radius + 1) * 3
        return 3

    def _neighbors_for(self, path: Path) -> list[Path]:
        if self.radius == 0:
            return [path]

        parent = path.parent
        if not parent.name.startswith("input"):
            return [path] * (2 * self.radius + 1)

        if parent not in self._neighbor_cache:
            self._neighbor_cache[parent] = self._collect_images(parent)
        siblings = self._neighbor_cache[parent]
        if not siblings or path not in siblings:
            return [path] * (2 * self.radius + 1)

        center_idx = siblings.index(path)
        last_idx = len(siblings) - 1
        return [
            siblings[min(max(center_idx + offset, 0), last_idx)]
            for offset in range(-self.radius, self.radius + 1)
        ]

    def _load_window(self, path: Path) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        paths = self._neighbors_for(path)
        frames = torch.stack([self._load_img(p) for p in paths], dim=0)
        center = frames[len(frames) // 2]
        if self.stack_mode == "stack":
            context = frames.flatten(0, 1)
        elif self.stack_mode == "mean":
            context = frames.mean(dim=0)
        elif self.stack_mode == "max":
            context = frames.max(dim=0).values
        else:
            context = center
        return center, context, [str(p) for p in paths]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | list[str]]:
        sample = self.samples[idx]
        nipple_x = self._read_nipple_x(sample.case_dir / "nipple_x.txt")

        left, left_context, left_window = self._load_window(sample.left_path)
        right, right_context, right_window = self._load_window(sample.right_path)

        left_idx = self._view_index(sample.left_path)
        right_idx = self._view_index(sample.right_path)
        left_x = torch.tensor([nipple_x[left_idx]], dtype=torch.float32)
        right_x = torch.tensor([nipple_x[right_idx]], dtype=torch.float32)

        if self.augment:
            left, right, left_context, right_context, left_x, right_x = self._apply_temporal_augment(
                left, right, left_context, right_context, left_x, right_x
            )

        return {
            "left": left,
            "right": right,
            "left_context": left_context,
            "right_context": right_context,
            "left_x": left_x,
            "right_x": right_x,
            "case": sample.case_dir.name,
            "left_path": str(sample.left_path),
            "right_path": str(sample.right_path),
            "left_window_paths": left_window,
            "right_window_paths": right_window,
        }

    def _apply_temporal_augment(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        left_context: torch.Tensor,
        right_context: torch.Tensor,
        left_x: torch.Tensor,
        right_x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _, _, w = left.shape
        if self.hflip_prob > 0 and random.random() < self.hflip_prob:
            left = torch.flip(left, dims=[2])
            right = torch.flip(right, dims=[2])
            left_context = torch.flip(left_context, dims=[2])
            right_context = torch.flip(right_context, dims=[2])
            left_x = (w - 1) - left_x
            right_x = (w - 1) - right_x

        if self.brightness_jitter > 0:
            bdelta = (random.random() * 2 - 1) * self.brightness_jitter
            left = (left + bdelta).clamp(0, 1)
            right = (right + bdelta).clamp(0, 1)
            left_context = (left_context + bdelta).clamp(0, 1)
            right_context = (right_context + bdelta).clamp(0, 1)
        if self.contrast_jitter > 0:
            cscale = 1.0 + (random.random() * 2 - 1) * self.contrast_jitter
            left = ((left - 0.5) * cscale + 0.5).clamp(0, 1)
            right = ((right - 0.5) * cscale + 0.5).clamp(0, 1)
            left_context = ((left_context - 0.5) * cscale + 0.5).clamp(0, 1)
            right_context = ((right_context - 0.5) * cscale + 0.5).clamp(0, 1)
        return left, right, left_context, right_context, left_x, right_x


def temporal_pair_collate_pad(batch: list[dict]) -> dict:
    max_h = max(item["left"].shape[1] for item in batch)
    max_w = max(item["left"].shape[2] for item in batch)

    def _pad_img(x: torch.Tensor) -> torch.Tensor:
        _, h, w = x.shape
        return F.pad(x, (0, max_w - w, 0, max_h - h), mode="constant", value=0.0)

    out = {
        "left": torch.stack([_pad_img(item["left"]) for item in batch], dim=0),
        "right": torch.stack([_pad_img(item["right"]) for item in batch], dim=0),
        "left_context": torch.stack([_pad_img(item["left_context"]) for item in batch], dim=0),
        "right_context": torch.stack([_pad_img(item["right_context"]) for item in batch], dim=0),
        "left_x": torch.stack([item["left_x"] for item in batch], dim=0),
        "right_x": torch.stack([item["right_x"] for item in batch], dim=0),
        "case": [item["case"] for item in batch],
        "left_path": [item["left_path"] for item in batch],
        "right_path": [item["right_path"] for item in batch],
    }
    return out
