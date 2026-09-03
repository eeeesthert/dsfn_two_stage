from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


def _read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def _resize_to_common(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    return a[:h, :w], b[:h, :w]

def _masked_mse(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    d = (a.astype(np.float64) - b.astype(np.float64)) ** 2
    return float(np.mean(d[mask]))


def _masked_psnr(mse: float, maxv: float = 255.0) -> float:
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10((maxv * maxv) / mse))


def _masked_ssim(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
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


def _build_band_mask(overlap_mask: np.ndarray, band_width: int) -> np.ndarray:
    overlap = (overlap_mask > 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    grad = cv2.morphologyEx(overlap, cv2.MORPH_GRADIENT, kernel)
    if band_width <= 1:
        return grad.astype(bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band_width + 1, 2 * band_width + 1))
    band = cv2.dilate(grad, k, iterations=1)
    return band.astype(bool)


def _list_case_names(stage_dir: Path) -> List[str]:
    return sorted(p.name for p in stage_dir.glob("case*") if p.is_dir())


def _iter_slice_ids(case12_fusion: Path, case23_fusion: Path) -> Iterable[str]:
    ids12 = {p.name[len("12_") : -len("_stitched.png")] for p in case12_fusion.glob("12_*_stitched.png")}
    ids23 = {p.name[len("23_") : -len("_stitched.png")] for p in case23_fusion.glob("23_*_stitched.png")}
    for sid in sorted(ids12 & ids23):
        yield sid


def _safe_mask(path: Path, shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    if not path.exists():
        return np.zeros((h, w), dtype=bool)
    m = _read_gray(path)
    return (m[:h, :w] > 0)


def _evaluate_pair(img12: np.ndarray, img23: np.ndarray, m12: np.ndarray, m23: np.ndarray, band_width: int) -> Dict[str, Dict[str, float]]:
    full = np.ones_like(m12, dtype=bool)
    union = (m12 | m23)
    overlap = (m12 & m23)
    seam = _build_band_mask((overlap * 255).astype(np.uint8), band_width)

    result: Dict[str, Dict[str, float]] = {}
    for region, mask in {
        "full": full,
        "union": union,
        "overlap": overlap,
        "seam-band": seam,
    }.items():
        valid = int(mask.sum())
        if valid == 0:
            result[region] = {"psnr": np.nan, "mse": np.nan, "ssim": np.nan, "ncc": np.nan, "valid_pixels": 0}
            continue
        mse = _masked_mse(img12, img23, mask)
        result[region] = {
            "psnr": _masked_psnr(mse),
            "mse": mse,
            "ssim": _masked_ssim(img12, img23, mask),
            "ncc": _masked_ncc(img12, img23, mask),
            "valid_pixels": valid,
        }
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="No-GT evaluation between stage12 and stage23 stitched outputs.")
    ap.add_argument("--pairwise-root", default="./outputs/results", help="contains 12/<case>/fusion and 23/<case>/fusion")
    ap.add_argument("--band-width", type=int, default=5)
    ap.add_argument("--out-csv", default="./outputs/no_gt_eval_all.csv")
    args = ap.parse_args()

    root = Path(args.pairwise_root)
    stage12 = root / "12"
    stage23 = root / "23"
    if not stage12.exists() or not stage23.exists():
        raise FileNotFoundError(f"Missing stage folders under {root}: expected 12/ and 23/")

    rows: List[Dict[str, object]] = []
    case_names = sorted(set(_list_case_names(stage12)) & set(_list_case_names(stage23)))

    for case_name in case_names:
        c12 = stage12 / case_name / "fusion"
        c23 = stage23 / case_name / "fusion"
        if not c12.exists() or not c23.exists():
            continue
        for sid in _iter_slice_ids(c12, c23):
            p12 = c12 / f"12_{sid}_stitched.png"
            p23 = c23 / f"23_{sid}_stitched.png"
            i12 = _read_gray(p12)
            i23 = _read_gray(p23)
            i12, i23 = _resize_to_common(i12, i23)
            h, w = i12.shape

            m12 = _safe_mask(c12 / f"12_{sid}_mask_right_soft.png", (h, w))
            m23 = _safe_mask(c23 / f"23_{sid}_mask_left_soft.png", (h, w))
            metrics = _evaluate_pair(i12, i23, m12, m23, args.band_width)
            for region, vals in metrics.items():
                rows.append(
                    {
                        "case": case_name,
                        "slice": sid,
                        "region": region,
                        "psnr": vals["psnr"],
                        "mse": vals["mse"],
                        "ssim": vals["ssim"],
                        "ncc": vals["ncc"],
                        "valid_pixels": vals["valid_pixels"],
                    }
                )

    # summary rows
    for level_key in ["case", "ALL"]:
        group_keys = sorted({r["case"] for r in rows}) if level_key == "case" else ["ALL"]
        for g in group_keys:
            for region in ["full", "union", "overlap", "seam-band"]:
                part = [r for r in rows if r["region"] == region and (r["case"] == g if level_key == "case" else True)]
                if not part:
                    continue
                rows.append(
                    {
                        "case": g if level_key == "case" else "ALL",
                        "slice": "__summary__",
                        "region": region,
                        "psnr": float(np.nanmean([p["psnr"] for p in part])),
                        "mse": float(np.nanmean([p["mse"] for p in part])),
                        "ssim": float(np.nanmean([p["ssim"] for p in part])),
                        "ncc": float(np.nanmean([p["ncc"] for p in part])),
                        "valid_pixels": int(np.sum([p["valid_pixels"] for p in part])),
                    }
                )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["case", "slice", "region", "psnr", "mse", "ssim", "ncc", "valid_pixels"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {out_csv} (rows={len(rows)})")


if __name__ == "__main__":
    main()
