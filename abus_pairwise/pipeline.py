from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torchvision.utils import save_image

from .losses import (
    fusion_regularization,
    nipple_prior_loss,
    warp_alignment_loss,
    x_heatmap_similarity_loss,
)
from .models.fusion import SoftSeamFusionUNet
from .models.warp import WarpStage


@dataclass
class LossWeights:
    warp_align: float = 1.0
    nipple_prior: float = 2.0
    x_heat: float = 1.0
    mask_tv: float = 0.05


class TwoStageStitcher(torch.nn.Module):
    def __init__(self, pretrained_backbone: bool = True):
        super().__init__()
        self.warp_net = WarpStage(pretrained=pretrained_backbone)
        self.fusion_net = SoftSeamFusionUNet()

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        warp_out = self.warp_net(left, right)
        fus_out = self.fusion_net(warp_out["left_warp"], warp_out["right_warp"])
        return {**warp_out, **fus_out}


def compute_total_loss(outputs: dict[str, torch.Tensor], left_x: torch.Tensor, right_x: torch.Tensor, w: LossWeights) -> dict[str, torch.Tensor]:
    l_warp = warp_alignment_loss(outputs["left_warp"], outputs["right_warp"], outputs["overlap"])
    l_nipple = nipple_prior_loss(outputs["global_shift_x"], left_x, right_x)
    l_xh = x_heatmap_similarity_loss(
        outputs["stitched"],
        outputs["left_warp"],
        outputs["right_warp"],
        left_x,
        right_x,
    )
    l_tv = fusion_regularization(outputs["mask_right"])

    total = w.warp_align * l_warp + w.nipple_prior * l_nipple + w.x_heat * l_xh + w.mask_tv * l_tv
    return {
        "total": total,
        "warp_align": l_warp,
        "nipple_prior": l_nipple,
        "x_heat": l_xh,
        "mask_tv": l_tv,
    }


def save_stage_results(outputs: dict[str, torch.Tensor], out_dir: str | Path, prefix: str) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_image(outputs["left_warp"].cpu(), out_dir / f"{prefix}_warp_left.png")
    save_image(outputs["right_warp"].cpu(), out_dir / f"{prefix}_warp_right.png")
    save_image(outputs["stitched"].cpu(), out_dir / f"{prefix}_fusion.png")
    save_image(outputs["mask_right"].cpu(), out_dir / f"{prefix}_mask_right.png")
