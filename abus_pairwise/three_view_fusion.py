from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def _read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _read_mask(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    return (m.astype(np.float32) / 255.0)[..., None]


def _resize_to(img: np.ndarray, h: int, w: int) -> np.ndarray:
    if img.shape[0] == h and img.shape[1] == w:
        return img
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


def _gaussian_pyramid(x: np.ndarray, levels: int) -> list[np.ndarray]:
    pyr = [x]
    cur = x
    for _ in range(levels - 1):
        cur = cv2.pyrDown(cur)
        pyr.append(cur)
    return pyr


def _laplacian_pyramid(x: np.ndarray, levels: int) -> list[np.ndarray]:
    g = _gaussian_pyramid(x, levels)
    l = []
    for i in range(levels - 1):
        up = cv2.pyrUp(g[i + 1], dstsize=(g[i].shape[1], g[i].shape[0]))
        l.append(g[i] - up)
    l.append(g[-1])
    return l


def _reconstruct_laplacian(lap: list[np.ndarray]) -> np.ndarray:
    out = lap[-1]
    for i in range(len(lap) - 2, -1, -1):
        out = cv2.pyrUp(out, dstsize=(lap[i].shape[1], lap[i].shape[0])) + lap[i]
    return out


def gaussian_pyramid_blend(images: Iterable[np.ndarray], weights: Iterable[np.ndarray], levels: int = 5) -> np.ndarray:
    imgs = list(images)
    ws = list(weights)
    if len(imgs) != len(ws):
        raise ValueError("images and weights length mismatch")

    h = max(i.shape[0] for i in imgs)
    w = max(i.shape[1] for i in imgs)
    imgs = [_resize_to(i, h, w) for i in imgs]
    ws = [_resize_to(wm, h, w) for wm in ws]

    # normalize weights at full resolution
    wsum = np.sum(ws, axis=0) + 1e-6
    ws = [w_i / wsum for w_i in ws]

    lap_imgs = [_laplacian_pyramid(i, levels) for i in imgs]
    gau_ws = [_gaussian_pyramid(w_i, levels) for w_i in ws]

    out_lap: list[np.ndarray] = []
    for lv in range(levels):
        wsum_lv = np.sum([gw[lv] for gw in gau_ws], axis=0) + 1e-6
        normalized_lv = [gw[lv] / wsum_lv for gw in gau_ws]
        blended = np.zeros_like(lap_imgs[0][lv], dtype=np.float32)
        for li, wi in zip(lap_imgs, normalized_lv):
            if wi.ndim == 2:
                wi = wi[..., None]
            blended += li[lv] * wi
        out_lap.append(blended)

    out = _reconstruct_laplacian(out_lap)
    return np.clip(out, 0.0, 1.0)


def fuse_case_from_pairwise(case12_dir: Path, case23_dir: Path, out_dir: Path, levels: int = 5, input2_boost: float = 2.0) -> int:
    f12 = case12_dir / "fusion"
    f23 = case23_dir / "fusion"
    if not f12.exists() or not f23.exists():
        return 0

    p12 = sorted(f12.glob("*_stitched.png"))
    p23 = sorted(f23.glob("*_stitched.png"))
    n = min(len(p12), len(p23))
    if n == 0:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for i in range(n):
        s12 = p12[i]
        s23 = p23[i]
        pre12 = s12.stem.replace("_stitched", "")
        pre23 = s23.stem.replace("_stitched", "")

        # input2 weights from stage12(right) and stage23(left)
        m12_path = f12 / f"{pre12}_mask_right_soft.png"
        m23_path = f23 / f"{pre23}_mask_left_soft.png"
        if not m12_path.exists():
            m12_path = f12 / f"{pre12}_mask_right_bin.png"
        if not m23_path.exists():
            m23_path = f23 / f"{pre23}_mask_left_bin.png"

        img12 = _read_rgb(s12)
        img23 = _read_rgb(s23)
        m12 = _read_mask(m12_path)
        m23 = _read_mask(m23_path)

        h = max(img12.shape[0], img23.shape[0])
        w = max(img12.shape[1], img23.shape[1])
        img12 = _resize_to(img12, h, w)
        img23 = _resize_to(img23, h, w)
        m12 = _resize_to(m12, h, w)
        m23 = _resize_to(m23, h, w)

        # proxy input2 image from two pairwise outputs
        img2 = (m12 * img12 + m23 * img23) / (m12 + m23 + 1e-6)

        # overlap of input2 from both stages => increase input2 weight
        overlap2 = np.minimum(m12, m23)
        w2 = np.clip(overlap2 * input2_boost, 0.0, 1.0)
        w12 = np.clip(1.0 - m23, 0.0, 1.0)
        w23 = np.clip(1.0 - m12, 0.0, 1.0)

        fused = gaussian_pyramid_blend([img12, img23, img2], [w12, w23, w2], levels=levels)
        out = (fused * 255.0).astype(np.uint8)
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

        out_name = out_dir / f"threeview_{i:03d}.png"
        cv2.imwrite(str(out_name), out)
        count += 1
    return count
