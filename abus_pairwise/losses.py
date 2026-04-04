from __future__ import annotations

import torch
import torch.nn.functional as F


def nipple_x_heatmap(x_pos: torch.Tensor, width: int, height: int, sigma: float = 9.0) -> torch.Tensor:
    """Create 2D heatmap only conditioned on x coordinate (broadcasted over y)."""
    b = x_pos.shape[0]
    xs = torch.arange(width, device=x_pos.device).float().view(1, 1, width)
    x = x_pos.view(b, 1, 1)
    line = torch.exp(-0.5 * ((xs - x) / sigma) ** 2)
    hm = line.repeat(1, height, 1).unsqueeze(1)
    return hm


def warp_alignment_loss(left_warp: torch.Tensor, right_warp: torch.Tensor, overlap: torch.Tensor) -> torch.Tensor:
    l1 = (left_warp - right_warp).abs() * overlap
    denom = overlap.sum().clamp_min(1.0)
    return l1.sum() / denom


def nipple_prior_loss(global_shift_x: torch.Tensor, left_x: torch.Tensor, right_x: torch.Tensor) -> torch.Tensor:
    """Force predicted shift to align left nipple x to right nipple x."""
    target_shift = right_x - left_x
    return F.smooth_l1_loss(global_shift_x, target_shift)


def x_heatmap_similarity_loss(
    stitched: torch.Tensor,
    left_warp: torch.Tensor,
    right_warp: torch.Tensor,
    left_x: torch.Tensor,
    right_x: torch.Tensor,
) -> torch.Tensor:
    _, _, h, w = stitched.shape
    hm_left = nipple_x_heatmap(left_x, width=w, height=h)
    hm_right = nipple_x_heatmap(right_x, width=w, height=h)
    hm = torch.maximum(hm_left, hm_right)

    # Encourage stitched image to preserve similar local texture near nipple x regions
    ref = 0.5 * (left_warp + right_warp)
    diff = (stitched - ref).abs().mean(1, keepdim=True)
    return (diff * hm).mean()


def fusion_regularization(mask_right: torch.Tensor) -> torch.Tensor:
    dx = (mask_right[:, :, :, 1:] - mask_right[:, :, :, :-1]).abs().mean()
    dy = (mask_right[:, :, 1:, :] - mask_right[:, :, :-1, :]).abs().mean()
    return dx + dy
