from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def image_corners(height: int, width: int) -> np.ndarray:
    return np.array([[0, 0], [width, 0], [0, height], [width, height]], dtype=np.float32)


def homography_matrix_to_corner_motion(H: np.ndarray, height: int, width: int) -> np.ndarray:
    matrix = np.asarray(H, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"Homography matrix must have shape (3,3), got {matrix.shape}")
    src = image_corners(height, width)
    dst = cv2.perspectiveTransform(src.reshape(1, 4, 2), matrix).reshape(4, 2)
    return (dst - src).astype(np.float32)


def corner_motion_to_points(motion: np.ndarray, height: int, width: int) -> np.ndarray:
    value = np.asarray(motion, dtype=np.float32)
    if value.shape != (4, 2):
        raise ValueError(f"Corner motion must have shape (4,2), got {value.shape}")
    return image_corners(height, width) + value


def validate_homography(
    corner_motion: np.ndarray,
    height: int,
    width: int,
    max_displacement_factor: float = 4.0,
    max_canvas_factor: float = 10.0,
    min_area_ratio: float = 0.01,
    min_overlap_ratio: float = 0.005,
) -> tuple[bool, str | None]:
    motion = np.asarray(corner_motion, dtype=np.float32)
    if motion.shape != (4, 2):
        return False, f"shape={motion.shape}, expected (4,2)"
    if not np.isfinite(motion).all():
        return False, "NaN or Inf corner displacement"
    diagonal = float(np.hypot(height, width))
    if np.linalg.norm(motion, axis=1).max() > max_displacement_factor * diagonal:
        return False, "extreme corner displacement"
    dst = corner_motion_to_points(motion, height, width)
    polygon = dst[[0, 1, 3, 2]]
    signed_area = float(cv2.contourArea(polygon, oriented=True))
    if signed_area <= min_area_ratio * height * width:
        return False, "flipped quadrilateral or almost-zero area"
    all_pts = np.concatenate([image_corners(height, width), dst])
    canvas_w = float(all_pts[:, 0].max() - all_pts[:, 0].min())
    canvas_h = float(all_pts[:, 1].max() - all_pts[:, 1].min())
    if canvas_w * canvas_h > max_canvas_factor * height * width:
        return False, "extreme output canvas"
    original_poly = image_corners(height, width)[[0, 1, 3, 2]]
    try:
        overlap, _ = cv2.intersectConvexConvex(original_poly, polygon)
    except cv2.error:
        return False, "non-convex projected quadrilateral"
    if overlap < min_overlap_ratio * height * width:
        return False, "too-small overlap"
    return True, None


@dataclass(frozen=True)
class HomographyResult:
    corner_motion: np.ndarray
    source: str
    valid: bool
    fallback: str | None
    rejection_reason: str | None = None
    path: str | None = None


class HomographyProvider:
    """Resolve coarse H independently of DSFN, with nipple/identity fallback."""

    MODES = {"precomputed", "udis2", "matrix", "nipple_translation", "identity"}

    def __init__(self, mode: str = "precomputed", root: str | Path | None = None) -> None:
        if mode not in self.MODES:
            raise ValueError(f"Unsupported homography mode {mode!r}")
        self.mode = mode
        self.root = Path(root) if root else None
        self.last_result: HomographyResult | None = None

    def _path(self, stage: str, case: str, slice_id: str) -> Path:
        if self.root is None:
            raise FileNotFoundError("homography_root was not provided")
        return self.root / stage / case / f"{slice_id}.npy"

    @staticmethod
    def _nipple_motion(height: int, width: int, x1: float | None, x2: float | None) -> np.ndarray | None:
        if x1 is None or x2 is None or not np.isfinite([x1, x2]).all():
            return None
        H = np.array([[1, 0, float(x2) - float(x1)], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        return homography_matrix_to_corner_motion(H, height, width)

    def get_result(
        self,
        image1: Any,
        image2: Any,
        case: str,
        slice_id: str,
        stage: str,
        x1: float | None = None,
        x2: float | None = None,
    ) -> HomographyResult:
        height, width = tuple(image1.shape[-2:])
        candidate: np.ndarray | None = None
        source = self.mode
        path: Path | None = None
        rejection: str | None = None
        try:
            if self.mode in {"precomputed", "udis2", "matrix"}:
                path = self._path(stage, case, slice_id)
                loaded = np.load(path)
                if self.mode == "matrix" or loaded.shape == (3, 3):
                    candidate = homography_matrix_to_corner_motion(loaded, height, width)
                else:
                    candidate = np.asarray(loaded, dtype=np.float32)
            elif self.mode == "nipple_translation":
                candidate = self._nipple_motion(height, width, x1, x2)
            else:
                candidate = np.zeros((4, 2), dtype=np.float32)
            if candidate is None:
                raise ValueError("nipple coordinates are unavailable")
            valid, rejection = validate_homography(candidate, height, width)
            if valid:
                result = HomographyResult(candidate, source, True, None, path=str(path) if path else None)
                self.last_result = result
                return result
        except (OSError, ValueError, cv2.error) as exc:
            rejection = str(exc)

        nipple = self._nipple_motion(height, width, x1, x2)
        if self.mode != "nipple_translation" and nipple is not None:
            valid, nipple_reason = validate_homography(nipple, height, width)
            if valid:
                result = HomographyResult(nipple, source, False, "nipple_translation", rejection, str(path) if path else None)
                self.last_result = result
                return result
            rejection = f"{rejection}; nipple rejected: {nipple_reason}"
        result = HomographyResult(np.zeros((4, 2), np.float32), source, False, "identity", rejection, str(path) if path else None)
        self.last_result = result
        return result

    def get(self, image1: Any, image2: Any, case: str, slice_id: str, stage: str, **kwargs: Any) -> np.ndarray:
        return self.get_result(image1, image2, case, slice_id, stage, **kwargs).corner_motion
