from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torchvision.utils import save_image

from .losses import (
    fusion_regularization,
    nipple_prior_loss,
    overlap_ncc_loss,
    warp_alignment_loss,
    x_heatmap_similarity_loss,
)
from .models.fusion import SoftSeamFusionUNet
from .models.warp import WarpStage


@dataclass
class LossWeights:
    warp_align: float = 1.0
    feature_align: float = 2.0
    nipple_prior: float = 0.5
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
    l_feat = overlap_ncc_loss(outputs["left_warp"], outputs["right_warp"], outputs["overlap"])
    l_nipple = nipple_prior_loss(outputs["global_shift_x"], left_x, right_x)
    l_xh = x_heatmap_similarity_loss(
        outputs["stitched"],
        outputs["left_warp"],
        outputs["right_warp"],
        left_x,
        right_x,
    )
    l_tv = fusion_regularization(outputs["mask_right"])

    total = (
        w.warp_align * l_warp
        + w.feature_align * l_feat
        + w.nipple_prior * l_nipple
        + w.x_heat * l_xh
        + w.mask_tv * l_tv
    )
    return {
        "total": total,
        "warp_align": l_warp,
        "feature_align": l_feat,
        "nipple_prior": l_nipple,
        "x_heat": l_xh,
        "mask_tv": l_tv,
    }


def save_stage_results(outputs: dict[str, torch.Tensor], out_dir: str | Path, prefix: str) -> None:
    save_stage_results_with_crop(outputs, out_dir, prefix, auto_crop=True)


def _bbox_from_valid(valid: torch.Tensor, min_size: int = 8) -> tuple[int, int, int, int]:
    """
    valid: (1, 1, H, W) binary mask.
    Returns y1, y2, x1, x2 (inclusive-exclusive).
    """
    ys, xs = torch.where(valid[0, 0] > 0)
    h, w = valid.shape[-2:]
    if ys.numel() == 0:
        return 0, h, 0, w
    y1, y2 = int(ys.min().item()), int(ys.max().item()) + 1
    x1, x2 = int(xs.min().item()), int(xs.max().item()) + 1
    if (y2 - y1) < min_size:
        yc = (y1 + y2) // 2
        y1 = max(0, yc - min_size // 2)
        y2 = min(h, y1 + min_size)
    if (x2 - x1) < min_size:
        xc = (x1 + x2) // 2
        x1 = max(0, xc - min_size // 2)
        x2 = min(w, x1 + min_size)
    return y1, y2, x1, x2


def save_stage_results_with_crop(
    outputs: dict[str, torch.Tensor],
    out_dir: str | Path,
    prefix: str,
    auto_crop: bool = True,
) -> None:
    out_dir = Path(out_dir)
    warp_dir = out_dir / "warp"
    fusion_dir = out_dir / "fusion"
    warp_dir.mkdir(parents=True, exist_ok=True)
    fusion_dir.mkdir(parents=True, exist_ok=True)

    left = outputs["left_warp"].detach().cpu()
    right = outputs["right_warp"].detach().cpu()
    stitched = outputs["stitched"].detach().cpu()
    mask_right = outputs["mask_right"].detach().cpu()

    if auto_crop:
        valid_left = (left.sum(1, keepdim=True) > 0).float()
        valid_right = (right.sum(1, keepdim=True) > 0).float()
        # Use union region (overlap + non-overlap), so output size follows effective stitched content.
        valid_union = torch.clamp(valid_left + valid_right, 0, 1)
        y1, y2, x1, x2 = _bbox_from_valid(valid_union)
        left = left[:, :, y1:y2, x1:x2]
        right = right[:, :, y1:y2, x1:x2]
        stitched = stitched[:, :, y1:y2, x1:x2]
        mask_right = mask_right[:, :, y1:y2, x1:x2]

    mask_left = 1.0 - mask_right
    bin_left = (mask_left > 0.5).float()
    bin_right = (mask_right > 0.5).float()

    save_image(left, warp_dir / f"{prefix}_left.png")
    save_image(right, warp_dir / f"{prefix}_right.png")
    save_image(stitched, fusion_dir / f"{prefix}_stitched.png")
    save_image(mask_left, fusion_dir / f"{prefix}_mask_left_soft.png")
    save_image(mask_right, fusion_dir / f"{prefix}_mask_right_soft.png")
    save_image(bin_left, fusion_dir / f"{prefix}_mask_left_bin.png")
    save_image(bin_right, fusion_dir / f"{prefix}_mask_right_bin.png")
