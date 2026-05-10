from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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
    return m[:h, :w] > 0


def _to_regions(m_left: np.ndarray, m_right: np.ndarray, band_width: int) -> Dict[str, np.ndarray]:
    full = np.ones_like(m_left, dtype=bool)
    union = m_left | m_right
    overlap = m_left & m_right
    seam = _build_band_mask((overlap * 255).astype(np.uint8), band_width)
    return {
        "full": full,
        "union": union,
        "overlap": overlap,
        "seam-band": seam,
    }


def _score_pair(a: np.ndarray, b: np.ndarray, regions: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for region, mask in regions.items():
        valid = int(mask.sum())
        if valid == 0:
            out[region] = {"psnr": np.nan, "mse": np.nan, "ssim": np.nan, "ncc": np.nan, "valid_pixels": 0}
            continue
        mse = _masked_mse(a, b, mask)
        out[region] = {
            "psnr": _masked_psnr(mse),
            "mse": mse,
            "ssim": _masked_ssim(a, b, mask),
            "ncc": _masked_ncc(a, b, mask),
            "valid_pixels": valid,
        }
    return out


def _append_rows(
    rows: List[Dict[str, object]],
    *,
    eval_family: str,
    stage: str,
    case: str,
    sid: str,
    ref_a: str,
    ref_b: str,
    metric_map: Dict[str, Dict[str, float]],
) -> None:
    for region, vals in metric_map.items():
        rows.append(
            {
                "eval_family": eval_family,
                "stage": stage,
                "case": case,
                "slice": sid,
                "region": region,
                "ref_a": ref_a,
                "ref_b": ref_b,
                "psnr": vals["psnr"],
                "mse": vals["mse"],
                "ssim": vals["ssim"],
                "ncc": vals["ncc"],
                "valid_pixels": vals["valid_pixels"],
            }
        )


def _summary_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out = list(rows)
    families = sorted({str(r["eval_family"]) for r in rows})
    stages = sorted({str(r["stage"]) for r in rows})
    regions = sorted({str(r["region"]) for r in rows})

    for level_key in ["case", "ALL"]:
        for fam in families:
            for stg in stages:
                part_fs = [r for r in rows if r["eval_family"] == fam and r["stage"] == stg]
                if not part_fs:
                    continue
                group_keys = sorted({str(r["case"]) for r in part_fs}) if level_key == "case" else ["ALL"]
                for g in group_keys:
                    for region in regions:
                        part = [
                            r
                            for r in part_fs
                            if r["region"] == region and (r["case"] == g if level_key == "case" else True)
                        ]
                        if not part:
                            continue

                        valid_arr = np.array([int(p["valid_pixels"]) for p in part], dtype=np.float64)
                        mse_arr = np.array([float(p["mse"]) for p in part], dtype=np.float64)
                        psnr_arr = np.array([float(p["psnr"]) for p in part], dtype=np.float64)
                        ssim_arr = np.array([float(p["ssim"]) for p in part], dtype=np.float64)
                        ncc_arr = np.array([float(p["ncc"]) for p in part], dtype=np.float64)

                        def _wmean(x: np.ndarray, w: np.ndarray) -> float:
                            ok = np.isfinite(x) & (w > 0)
                            if not np.any(ok):
                                return float("nan")
                            return float(np.sum(x[ok] * w[ok]) / np.sum(w[ok]))

                        out.append(
                            {
                                "eval_family": fam,
                                "stage": stg,
                                "case": g if level_key == "case" else "ALL",
                                "slice": "__summary__",
                                "region": region,
                                "ref_a": "__summary__",
                                "ref_b": "__summary__",
                                "psnr": float(np.nanmean(psnr_arr)),
                                "mse": float(np.nanmean(mse_arr)),
                                "ssim": float(np.nanmean(ssim_arr)),
                                "ncc": float(np.nanmean(ncc_arr)),
                                "valid_pixels": int(np.nansum(valid_arr)),
                                "summary_level": level_key,
                                "aggregation": "macro",
                                "psnr_micro": _wmean(psnr_arr, valid_arr),
                                "mse_micro": _wmean(mse_arr, valid_arr),
                                "ssim_micro": _wmean(ssim_arr, valid_arr),
                                "ncc_micro": _wmean(ncc_arr, valid_arr),
                            }
                        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Consistency + Fidelity no-GT evaluation for pairwise stitching outputs.")
    ap.add_argument("--pairwise-root", default="./outputs/results", help="contains 12/<case> and 23/<case>")
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
        c12 = stage12 / case_name
        c23 = stage23 / case_name
        f12 = c12 / "fusion"
        f23 = c23 / "fusion"
        w12 = c12 / "warp"
        w23 = c23 / "warp"
        if not f12.exists() or not f23.exists():
            continue

        for sid in _iter_slice_ids(f12, f23):
            # ---------- Consistency: 12 stitched vs 23 stitched ----------
            p12s = f12 / f"12_{sid}_stitched.png"
            p23s = f23 / f"23_{sid}_stitched.png"
            i12 = _read_gray(p12s)
            i23 = _read_gray(p23s)
            i12, i23 = _resize_to_common(i12, i23)
            h, w = i12.shape

            m12r = _safe_mask(f12 / f"12_{sid}_mask_right_soft.png", (h, w))
            m23l = _safe_mask(f23 / f"23_{sid}_mask_left_soft.png", (h, w))
            regions_c = _to_regions(m12r, m23l, band_width=args.band_width)
            scores_c = _score_pair(i12, i23, regions_c)
            _append_rows(
                rows,
                eval_family="consistency",
                stage="cross",
                case=case_name,
                sid=sid,
                ref_a=p12s.name,
                ref_b=p23s.name,
                metric_map=scores_c,
            )

            # ---------- Fidelity Stage12: stitched vs left/right warp ----------
            p12l = w12 / f"12_{sid}_left.png"
            p12r = w12 / f"12_{sid}_right.png"
            if p12l.exists() and p12r.exists():
                l12 = _read_gray(p12l)
                r12 = _read_gray(p12r)
                l12, r12 = _resize_to_common(l12, r12)
                s12 = _read_gray(p12s)
                s12, l12 = _resize_to_common(s12, l12)
                s12, r12 = _resize_to_common(s12, r12)
                hh, ww = s12.shape
                v12l = l12 > 0
                v12r = r12 > 0
                regions_12 = _to_regions(v12l, v12r, band_width=args.band_width)
                _append_rows(rows, eval_family="fidelity", stage="12-L", case=case_name, sid=sid, ref_a=p12s.name, ref_b=p12l.name, metric_map=_score_pair(s12, l12, regions_12))
                _append_rows(rows, eval_family="fidelity", stage="12-R", case=case_name, sid=sid, ref_a=p12s.name, ref_b=p12r.name, metric_map=_score_pair(s12, r12, regions_12))

            # ---------- Fidelity Stage23: stitched vs left/right warp ----------
            p23l = w23 / f"23_{sid}_left.png"
            p23r = w23 / f"23_{sid}_right.png"
            if p23l.exists() and p23r.exists():
                l23 = _read_gray(p23l)
                r23 = _read_gray(p23r)
                l23, r23 = _resize_to_common(l23, r23)
                s23 = _read_gray(p23s)
                s23, l23 = _resize_to_common(s23, l23)
                s23, r23 = _resize_to_common(s23, r23)
                v23l = l23 > 0
                v23r = r23 > 0
                regions_23 = _to_regions(v23l, v23r, band_width=args.band_width)
                _append_rows(rows, eval_family="fidelity", stage="23-L", case=case_name, sid=sid, ref_a=p23s.name, ref_b=p23l.name, metric_map=_score_pair(s23, l23, regions_23))
                _append_rows(rows, eval_family="fidelity", stage="23-R", case=case_name, sid=sid, ref_a=p23s.name, ref_b=p23r.name, metric_map=_score_pair(s23, r23, regions_23))

    rows = _summary_rows(rows)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "eval_family",
                "stage",
                "case",
                "slice",
                "region",
                "ref_a",
                "ref_b",
                "psnr",
                "mse",
                "ssim",
                "ncc",
                "valid_pixels",
                "summary_level",
                "aggregation",
                "psnr_micro",
                "mse_micro",
                "ssim_micro",
                "ncc_micro",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {out_csv} (rows={len(rows)})")


if __name__ == "__main__":
    main()
