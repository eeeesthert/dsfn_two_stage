from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class MetricResult:
    psnr: float
    mse: float
    ssim: float
    ncc: float
    valid_pixels: int


def _read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def _build_band_mask(overlap_mask: np.ndarray, band_width: int) -> np.ndarray:
    overlap = (overlap_mask > 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    grad = cv2.morphologyEx(overlap, cv2.MORPH_GRADIENT, kernel)
    if band_width <= 1:
        return grad.astype(bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band_width + 1, 2 * band_width + 1))
    band = cv2.dilate(grad, k, iterations=1)
    return band.astype(bool)


def _masked_mse(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    d = (a.astype(np.float64) - b.astype(np.float64)) ** 2
    return float(np.mean(d[mask]))


def _masked_psnr(mse: float, maxv: float = 255.0) -> float:
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10((maxv * maxv) / mse))


def _masked_ssim(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    # global masked SSIM
    x = a.astype(np.float64)[mask]
    y = b.astype(np.float64)[mask]
    if x.size < 2:
        return float("nan")
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    ux, uy = np.mean(x), np.mean(y)
    vx, vy = np.var(x), np.var(y)
    cov = np.mean((x - ux) * (y - uy))
    den = (ux * ux + uy * uy + c1) * (vx + vy + c2)
    if den <= 1e-12:
        return 1.0
    return float(((2 * ux * uy + c1) * (2 * cov + c2)) / den)


def _masked_ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    x = a.astype(np.float64)[mask]
    y = b.astype(np.float64)[mask]
    if x.size < 2:
        return float("nan")
    x = x - np.mean(x)
    y = y - np.mean(y)
    den = np.linalg.norm(x) * np.linalg.norm(y)
    if den <= 1e-12:
        return 0.0
    return float(np.dot(x, y) / den)


def compute_metrics(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> MetricResult:
    valid = int(mask.sum())
    if valid == 0:
        return MetricResult(np.nan, np.nan, np.nan, np.nan, 0)
    mse = _masked_mse(pred, gt, mask)
    return MetricResult(
        psnr=_masked_psnr(mse),
        mse=mse,
        ssim=_masked_ssim(pred, gt, mask),
        ncc=_masked_ncc(pred, gt, mask),
        valid_pixels=valid,
    )


def _resize_to_match(pred: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h = min(pred.shape[0], gt.shape[0])
    w = min(pred.shape[1], gt.shape[1])
    return pred[:h, :w], gt[:h, :w]


def _iter_pairs(pred_dir: Path, gt_dir: Path, pred_suffix: str, gt_suffix: str) -> Iterable[Tuple[str, Path, Path]]:
    pred_files = sorted(pred_dir.glob(f"*{pred_suffix}"))
    for pf in pred_files:
        stem = pf.name[: -len(pred_suffix)] if pred_suffix else pf.stem
        gf = gt_dir / f"{stem}{gt_suffix}"
        if gf.exists():
            yield stem, pf, gf


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate full/union/overlap/seam-band metrics: PSNR MSE SSIM NCC")
    ap.add_argument("--pred-dir", required=True, help="predicted stitched image directory")
    ap.add_argument("--gt-dir", required=True, help="ground-truth image directory")
    ap.add_argument("--pred-suffix", default="_stitched.png")
    ap.add_argument("--gt-suffix", default=".png")
    ap.add_argument("--union-mask-dir", default="", help="optional union mask directory")
    ap.add_argument("--overlap-mask-dir", default="", help="optional overlap mask directory")
    ap.add_argument("--mask-suffix", default="_mask.png")
    ap.add_argument("--band-width", type=int, default=5)
    ap.add_argument("--out-csv", default="metrics_eval.csv")
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    union_dir = Path(args.union_mask_dir) if args.union_mask_dir else None
    overlap_dir = Path(args.overlap_mask_dir) if args.overlap_mask_dir else None

    rows: List[Dict[str, object]] = []

    for name, pf, gf in _iter_pairs(pred_dir, gt_dir, args.pred_suffix, args.gt_suffix):
        pred = _read_gray(pf)
        gt = _read_gray(gf)
        pred, gt = _resize_to_match(pred, gt)
        h, w = pred.shape

        full = np.ones((h, w), dtype=bool)

        if union_dir is not None:
            up = union_dir / f"{name}{args.mask_suffix}"
            union_mask = _read_gray(up)[:h, :w] > 0 if up.exists() else full
        else:
            union_mask = full

        if overlap_dir is not None:
            op = overlap_dir / f"{name}{args.mask_suffix}"
            overlap_mask = _read_gray(op)[:h, :w] > 0 if op.exists() else np.zeros((h, w), dtype=bool)
        else:
            overlap_mask = np.zeros((h, w), dtype=bool)

        seam_band = _build_band_mask((overlap_mask * 255).astype(np.uint8), args.band_width)
        groups = {
            "full": full,
            "union": union_mask,
            "overlap": overlap_mask,
            "seam-band": seam_band,
        }

        for region_name, mask in groups.items():
            m = compute_metrics(pred, gt, mask)
            rows.append(
                {
                    "name": name,
                    "region": region_name,
                    "psnr": m.psnr,
                    "mse": m.mse,
                    "ssim": m.ssim,
                    "ncc": m.ncc,
                    "valid_pixels": m.valid_pixels,
                }
            )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "region", "psnr", "mse", "ssim", "ncc", "valid_pixels"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {out_csv} (rows={len(rows)})")


if __name__ == "__main__":
    main()
