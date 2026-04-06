from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import ResNet50MultiScale


class ReparamRegression(nn.Module):
    """Two-step regression: global shift + residual dense flow."""

    def __init__(self, c_in: int = 4096):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c_in, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1),
        )
        self.delta = nn.Sequential(
            nn.Conv2d(c_in, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 2, 3, padding=1),
        )

    def forward(self, f_left: torch.Tensor, f_right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([f_left, f_right], dim=1)
        global_x = self.fc(x)  # Bx1
        dense = self.delta(x)  # Bx2xhxw
        return global_x, dense


class WarpStage(nn.Module):
    def __init__(
        self,
        encoder_pretrain_source: str = "imagenet",
        encoder_ckpt: str | None = None,
        encoder_strict_load: bool = False,
    ):
        super().__init__()
        self.encoder = ResNet50MultiScale(
            pretrain_source=encoder_pretrain_source,
            checkpoint_path=encoder_ckpt,
            strict_load=encoder_strict_load,
        )
        self.reg = ReparamRegression(c_in=2048 * 2)

    @staticmethod
    def _base_grid(b: int, h: int, w: int, device: torch.device) -> torch.Tensor:
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, h, device=device),
            torch.linspace(-1, 1, w, device=device),
            indexing="ij",
        )
        base = torch.stack([x, y], dim=-1)
        return base.unsqueeze(0).repeat(b, 1, 1, 1)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        f_l = self.encoder(left)
        f_r = self.encoder(right)

        gdx, dense_low = self.reg(f_l[-1], f_r[-1])
        dense = F.interpolate(dense_low, size=left.shape[-2:], mode="bilinear", align_corners=False)

        b, _, h, w = left.shape
        grid = self._base_grid(b, h, w, left.device)
        grid[..., 0] = grid[..., 0] + (2.0 * gdx / w).view(-1, 1, 1)
        grid = grid + dense.permute(0, 2, 3, 1).tanh() * 0.15

        left_w = F.grid_sample(left, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        right_w = right
        overlap = ((left_w.sum(1, keepdim=True) > 0) & (right_w.sum(1, keepdim=True) > 0)).float()

        return {
            "left_warp": left_w,
            "right_warp": right_w,
            "grid": grid,
            "global_shift_x": gdx,
            "overlap": overlap,
        }
