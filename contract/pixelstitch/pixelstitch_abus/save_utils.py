from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def output_paths(root: str | Path, stage: str, case: str, slice_id: str) -> dict[str, Path]:
    base = Path(root) / stage / case
    prefix = f"{stage}_{slice_id}"
    result = {
        "base": base,
        "left": base / "warp" / f"{prefix}_left.png",
        "right": base / "warp" / f"{prefix}_right.png",
        "stitched": base / "fusion" / f"{prefix}_stitched.png",
        "official": base / "fusion" / f"{prefix}_blend_official.png",
        "mask_left_soft": base / "fusion" / f"{prefix}_mask_left_soft.png",
        "mask_right_soft": base / "fusion" / f"{prefix}_mask_right_soft.png",
        "overlap": base / "fusion" / f"{prefix}_overlap.png",
        "union": base / "fusion" / f"{prefix}_union.png",
        "metadata": base / "fusion" / f"{prefix}_metadata.json",
        "debug": base / "fusion" / f"{prefix}_debug.png",
        "flow_dir": base / "flow",
        "homo": base / "homo" / f"{prefix}_homography.npy",
    }
    for folder in (base / "warp", base / "fusion", base / "flow", base / "homo"):
        folder.mkdir(parents=True, exist_ok=True)
    return result


def _to_image(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().float().cpu().squeeze(0).permute(1, 2, 0).numpy()
    return np.clip(np.rint(array * 255), 0, 255).astype(np.uint8)[..., ::-1]


def _to_mask(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().float().cpu().squeeze().numpy()
    return np.clip(np.rint(array * 255), 0, 255).astype(np.uint8)


def save_outputs(paths: dict[str, Path], outputs: dict[str, torch.Tensor], metadata: dict[str, Any]) -> None:
    import cv2

    cv2.imwrite(str(paths["left"]), _to_image(outputs["warp1"]))
    cv2.imwrite(str(paths["right"]), _to_image(outputs["warp2"]))
    cv2.imwrite(str(paths["stitched"]), _to_image(outputs["blend_avg"]))
    cv2.imwrite(str(paths["official"]), _to_image(outputs["blend_official"]))
    for key in ("mask_left_soft", "mask_right_soft", "overlap", "union"):
        cv2.imwrite(str(paths[key]), _to_mask(outputs[key]))
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def save_debug_panel(paths: dict[str, Path], panels: list[tuple[str, torch.Tensor]]) -> None:
    import cv2

    rendered: list[np.ndarray] = []
    target_h = 240
    for title, value in panels:
        if value.shape[1] == 1:
            img = cv2.cvtColor(_to_mask(value), cv2.COLOR_GRAY2BGR)
        else:
            img = _to_image(value)
        scale = target_h / max(1, img.shape[0])
        img = cv2.resize(img, (max(1, round(img.shape[1] * scale)), target_h))
        cv2.putText(img, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
        rendered.append(img)
    cols = 4
    rows = []
    for start in range(0, len(rendered), cols):
        row = rendered[start : start + cols]
        max_w = max(x.shape[1] for x in row)
        row += [np.zeros((target_h, max_w, 3), np.uint8)] * (cols - len(row))
        row = [cv2.resize(x, (max_w, target_h)) for x in row]
        rows.append(np.concatenate(row, axis=1))
    cv2.imwrite(str(paths["debug"]), np.concatenate(rows, axis=0))
