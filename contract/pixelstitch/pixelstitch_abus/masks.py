from __future__ import annotations

import torch
import torch.nn.functional as F


def residual_warp(image: torch.Tensor, sampling_grid_norm: torch.Tensor) -> torch.Tensor:
    return F.grid_sample(image, sampling_grid_norm, mode="bilinear", padding_mode="zeros", align_corners=False)


def warp_support_masks(
    homography_mask1: torch.Tensor,
    homography_mask2: torch.Tensor,
    sampling_grid_norm: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, torch.Tensor]:
    soft1 = residual_warp(homography_mask1, sampling_grid_norm[[0]])
    soft2 = residual_warp(homography_mask2, sampling_grid_norm[[1]])
    mask1 = soft1 > threshold
    mask2 = soft2 > threshold
    return {
        "mask_left_soft": soft1.clamp(0, 1),
        "mask_right_soft": soft2.clamp(0, 1),
        "mask_left": mask1,
        "mask_right": mask2,
        "overlap": mask1 & mask2,
        "union": mask1 | mask2,
    }


def common_union_crop(union: torch.Tensor) -> tuple[int, int, int, int]:
    locations = torch.nonzero(union[0, 0], as_tuple=False)
    h, w = union.shape[-2:]
    if locations.numel() == 0:
        return 0, 0, w, h
    y1, x1 = locations.min(dim=0).values.tolist()
    y2, x2 = (locations.max(dim=0).values + 1).tolist()
    return int(x1), int(y1), int(x2), int(y2)


def crop_tensor(value: torch.Tensor, bbox: tuple[int, int, int, int]) -> torch.Tensor:
    x1, y1, x2, y2 = bbox
    return value[..., y1:y2, x1:x2]
