from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from modules.models import utils as model_utils
from modules.models.raft.raft import RAFTStitch
from modules.models.utils import centralizeH, computeAndDecomposeH, warpHomo, warpPointsHomo

from .homography import HomographyProvider
from .masks import common_union_crop, crop_tensor, residual_warp, warp_support_masks
from .save_utils import output_paths, save_debug_panel, save_outputs
from .utils import load_checkpoint_validated, pad_to_multiple8, unpad


class ModelArgs(dict):
    """Mapping with attribute access, matching RAFTStitch's official argument contract."""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def build_model(checkpoint: str | Path, device: str, iters: int, corr_kernel_size: int) -> RAFTStitch:
    args = ModelArgs(ftdown=8, radius=corr_kernel_size // 2, corr_kernel_size=corr_kernel_size, iters=iters)
    model = RAFTStitch(args).to(device)
    load_checkpoint_validated(model, checkpoint, device)
    model.eval()
    return model


def _coarse_warp(
    image1: torch.Tensor, image2: torch.Tensor, corner_motion: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
    _, _, h, w = image1.shape
    motion = torch.as_tensor(corner_motion, dtype=torch.float32, device=image1.device).unsqueeze(0)
    H1, H2 = computeAndDecomposeH(h, w, motion)
    H1, H2, out_h, out_w = centralizeH(h, w, H1, H2, h, w)
    out_h_i, out_w_i = int(torch.round(out_h).item()), int(torch.round(out_w).item())
    if out_h_i <= 0 or out_w_i <= 0:
        raise ValueError(f"Invalid PixelStitch canvas {(out_h_i, out_w_i)}")
    warp1 = warpHomo(image1, H1, h, w, out_h_i, out_w_i)
    warp2 = warpHomo(image2, H2, h, w, out_h_i, out_w_i)
    support = torch.ones((1, 1, h, w), dtype=image1.dtype, device=image1.device)
    mask1 = warpHomo(support, H1, h, w, out_h_i, out_w_i)
    mask2 = warpHomo(support, H2, h, w, out_h_i, out_w_i)
    center1 = warpPointsHomo(torch.tensor([[[w / 2, h / 2]]], device=image1.device), H1)[:, 0]
    center2 = warpPointsHomo(torch.tensor([[[w / 2, h / 2]]], device=image1.device), H2)[:, 0]
    return warp1, warp2, mask1, mask2, center1, center2, (out_h_i, out_w_i)


def _flow_magnitude(flow_xy_px: torch.Tensor) -> torch.Tensor:
    magnitude = torch.linalg.vector_norm(flow_xy_px, dim=-1, keepdim=True).permute(0, 3, 1, 2)
    return magnitude / magnitude.amax().clamp_min(1e-6)


@torch.no_grad()
def infer_sample(
    model: RAFTStitch,
    sample: dict[str, Any],
    provider: HomographyProvider,
    out_dir: str | Path,
    device: str,
    iters: int = 4,
    auto_crop: bool = True,
) -> dict[str, Any]:
    image1 = sample["image1"].unsqueeze(0).to(device)
    image2 = sample["image2"].unsqueeze(0).to(device)
    result = provider.get_result(
        image1, image2, sample["case"], sample["slice_id"], sample["stage"],
        x1=float(sample["x1"]), x2=float(sample["x2"]),
    )
    try:
        coarse1, coarse2, coarse_mask1, coarse_mask2, c1, c2, canvas_shape = _coarse_warp(
            image1, image2, result.corner_motion
        )
    except (RuntimeError, ValueError) as exc:
        # A numerically valid corner polygon can still fail the official eigen decomposition.
        result = provider.get_result(
            image1, image2, sample["case"], sample["slice_id"], "identity",
            x1=None, x2=None,
        ) if provider.mode == "identity" else type(result)(
            np.zeros((4, 2), np.float32), result.source, False, "identity", str(exc), result.path
        )
        coarse1, coarse2, coarse_mask1, coarse_mask2, c1, c2, canvas_shape = _coarse_warp(
            image1, image2, result.corner_motion
        )

    padded1, padding = pad_to_multiple8(coarse1)
    padded2, _ = pad_to_multiple8(coarse2)
    image_pair = torch.cat([padded1, padded2], dim=0)
    flow_predictions, _ = model(image_pair, iters=iters)
    raw_flow_xy_px = unpad(flow_predictions[-1], padding).permute(0, 2, 3, 1).contiguous()
    weight1, weight2 = model_utils.getAdaptiveWeight(raw_flow_xy_px[[0]], raw_flow_xy_px[[1]], c1, c2)
    weighted_flow_xy_px = raw_flow_xy_px.clone()
    weighted_flow_xy_px[[0]] *= weight1
    weighted_flow_xy_px[[1]] *= weight2

    h, w = weighted_flow_xy_px.shape[1:3]
    flow_xy_norm = weighted_flow_xy_px.clone()
    flow_xy_norm[..., 0] /= w / 2
    flow_xy_norm[..., 1] /= h / 2
    sampling_grid_norm = model_utils.getGrid(h, w).to(device) + flow_xy_norm
    warp1 = residual_warp(coarse1, sampling_grid_norm[[0]])
    warp2 = residual_warp(coarse2, sampling_grid_norm[[1]])
    masks = warp_support_masks(coarse_mask1, coarse_mask2, sampling_grid_norm)
    denominator = masks["mask_left_soft"] + masks["mask_right_soft"]
    blend_avg = (warp1 * masks["mask_left_soft"] + warp2 * masks["mask_right_soft"]) / denominator.clamp_min(1e-6)
    image_sum = warp1 + warp2
    blend_official = warp1.square() / (image_sum + 1e-6) + warp2.square() / (image_sum + 1e-6)

    outputs = {"warp1": warp1, "warp2": warp2, "blend_avg": blend_avg, "blend_official": blend_official, **masks}
    bbox = common_union_crop(masks["union"]) if auto_crop else (0, 0, w, h)
    for key in list(outputs):
        outputs[key] = crop_tensor(outputs[key], bbox)

    paths = output_paths(out_dir, sample["stage"], sample["case"], sample["slice_id"])
    prefix = f"{sample['stage']}_{sample['slice_id']}"
    np.save(paths["homo"], result.corner_motion)
    arrays = {
        "raw_flow1": raw_flow_xy_px[0], "raw_flow2": raw_flow_xy_px[1],
        "weighted_flow1": weighted_flow_xy_px[0], "weighted_flow2": weighted_flow_xy_px[1],
        "adaptive_weight1": weight1[0], "adaptive_weight2": weight2[0],
    }
    for name, value in arrays.items():
        np.save(paths["flow_dir"] / f"{prefix}_{name}.npy", value.detach().cpu().numpy())

    x1, y1, x2, y2 = bbox
    metadata = {
        "method": "PixelStitch", "stage": sample["stage"], "case": sample["case"],
        "slice_id": sample["slice_id"], "input1": sample["path1"], "input2": sample["path2"],
        "input_shape": list(image1.shape[-2:]), "original_shape1": list(sample["original_shape1"]),
        "original_shape2": list(sample["original_shape2"]), "resize_factor1": list(sample["resize_factor1"]),
        "resize_factor2": list(sample["resize_factor2"]), "homography_source": result.source,
        "homography_valid": result.valid, "homography_fallback": result.fallback,
        "homography_rejection_reason": result.rejection_reason, "homography_path": result.path,
        "canvas_shape": list(canvas_shape), "output_shape": list(outputs["warp1"].shape[-2:]),
        "crop_x1": x1, "crop_x2": x2, "crop_y1": y1, "crop_y2": y2,
        "flow1_mean": float(torch.linalg.vector_norm(weighted_flow_xy_px[0], dim=-1).mean()),
        "flow1_max": float(torch.linalg.vector_norm(weighted_flow_xy_px[0], dim=-1).max()),
        "flow2_mean": float(torch.linalg.vector_norm(weighted_flow_xy_px[1], dim=-1).mean()),
        "flow2_max": float(torch.linalg.vector_norm(weighted_flow_xy_px[1], dim=-1).max()),
        "overlap_pixels": int(outputs["overlap"].sum()), "union_pixels": int(outputs["union"].sum()),
    }
    save_outputs(paths, outputs, metadata)
    save_debug_panel(paths, [
        ("Input 1", image1), ("Input 2", image2), ("Homography Warp 1", coarse1),
        ("Homography Warp 2", coarse2), ("PixelStitch Warp 1", outputs["warp1"]),
        ("PixelStitch Warp 2", outputs["warp2"]), ("Overlap Mask", outputs["overlap"].float()),
        ("Stitched", outputs["blend_avg"]), ("Flow 1", _flow_magnitude(raw_flow_xy_px[[0]])),
        ("Flow 2", _flow_magnitude(raw_flow_xy_px[[1]])),
        ("Adaptive Weight 1", weight1.permute(0, 3, 1, 2)),
        ("Adaptive Weight 2", weight2.permute(0, 3, 1, 2)),
    ])
    return metadata
