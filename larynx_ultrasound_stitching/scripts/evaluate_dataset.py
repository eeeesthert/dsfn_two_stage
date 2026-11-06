"""No-ground-truth evaluation for completed pair12 and pair23 stitching jobs.

The evaluator reports two distinct questions instead of mixing them:

* registration: warped reference versus warped target, only where both exist
* fusion consistency: fused canvas versus each contributing warped image

These are no-reference consistency measures. They are not a substitute for landmark,
TRE, or known-homography accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from skimage.metrics import structural_similarity

COMPARISONS = ("registration", "fused_reference", "fused_target")
REGIONS = ("support", "overlap", "seam_band")


def read_gray(path: Path) -> np.ndarray:
	image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
	if image is None:
		raise FileNotFoundError(f"Failed to read image: {path}")
	return image


def read_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
	mask = read_gray(path)
	if mask.shape != shape:
		raise ValueError(
			f"Mask shape mismatch for {path}: expected {shape}, got {mask.shape}"
		)
	return mask > 0


def seam_band(overlap: np.ndarray, width: int) -> np.ndarray:
	"""Return a band around the inner and outer overlap boundary."""

	if width < 1:
		raise ValueError("band width must be at least one pixel")
	binary = overlap.astype(np.uint8)
	gradient = cv2.morphologyEx(
		binary,
		cv2.MORPH_GRADIENT,
		np.ones((3, 3), np.uint8),
	)
	if width == 1:
		return gradient.astype(bool)
	kernel = cv2.getStructuringElement(
		cv2.MORPH_ELLIPSE,
		(2 * width + 1, 2 * width + 1),
	)
	return cv2.dilate(gradient, kernel).astype(bool)


def masked_metrics(
	first: np.ndarray,
	second: np.ndarray,
	mask: np.ndarray,
) -> dict[str, float | int]:
	"""Compute MSE, PSNR, local-window SSIM, and NCC on one valid mask."""

	if first.shape != second.shape or first.shape != mask.shape:
		raise ValueError("Image and mask shapes must match")
	valid_pixels = int(mask.sum())
	if valid_pixels == 0:
		return {
			"mse": math.nan,
			"psnr": math.nan,
			"ssim": math.nan,
			"ncc": math.nan,
			"valid_pixels": 0,
		}
	first_float = first.astype(np.float64)
	second_float = second.astype(np.float64)
	difference = first_float[mask] - second_float[mask]
	mse = float(np.mean(difference**2))
	psnr = math.inf if mse <= 1e-12 else float(10 * np.log10(255.0**2 / mse))
	first_centered = first_float[mask] - np.mean(first_float[mask])
	second_centered = second_float[mask] - np.mean(second_float[mask])
	denominator = np.linalg.norm(first_centered) * np.linalg.norm(second_centered)
	ncc = 0.0 if denominator <= 1e-12 else float(first_centered @ second_centered / denominator)
	minimum_size = min(first.shape)
	window_size = min(7, minimum_size if minimum_size % 2 else minimum_size - 1)
	if window_size < 3:
		ssim = math.nan
	else:
		_, ssim_map = structural_similarity(
			first,
			second,
			data_range=255,
			win_size=window_size,
			full=True,
		)
		ssim = float(np.mean(ssim_map[mask]))
	return {
		"mse": mse,
		"psnr": psnr,
		"ssim": ssim,
		"ncc": ncc,
		"valid_pixels": valid_pixels,
	}


def completed_jobs(output_root: Path) -> Iterable[tuple[str, str, str, Path]]:
	"""Yield successful slice directories in deterministic order."""

	for status_path in sorted(output_root.glob("case*/pair*/slice_*/status.json")):
		status = json.loads(status_path.read_text(encoding="utf8"))
		if status.get("status") != "SUCCESS":
			continue
		slice_dir = status_path.parent
		pair_name = slice_dir.parent.name
		yield (
			slice_dir.parent.parent.name,
			pair_name.removeprefix("pair"),
			slice_dir.name,
			slice_dir,
		)


def evaluate_job(
	case: str,
	pair: str,
	slice_name: str,
	slice_dir: Path,
	band_width: int,
) -> list[dict[str, object]]:
	warp_dir = slice_dir / "09_warp"
	fusion_dir = slice_dir / "10_fusion"
	reference = read_gray(warp_dir / "prealigned_reference.png")
	target = read_gray(warp_dir / "prealigned_target.png")
	fused = read_gray(fusion_dir / "stitched_raw.png")
	if reference.shape != target.shape or reference.shape != fused.shape:
		raise ValueError(f"Canvas shape mismatch in {slice_dir}")
	reference_mask = read_mask(fusion_dir / "reference_mask.png", reference.shape)
	target_mask = read_mask(fusion_dir / "target_mask.png", reference.shape)
	overlap = reference_mask & target_mask
	seam = seam_band(overlap, band_width) & (reference_mask | target_mask)
	comparisons = {
		"registration": (reference, target, overlap),
		"fused_reference": (fused, reference, reference_mask),
		"fused_target": (fused, target, target_mask),
	}
	rows = []
	for comparison, (first, second, support) in comparisons.items():
		region_masks = {
			"support": support,
			"overlap": overlap & support,
			"seam_band": seam & support,
		}
		for region, mask in region_masks.items():
			rows.append(
				{
					"case": case,
					"pair": pair,
					"slice": slice_name,
					"comparison": comparison,
					"region": region,
					**masked_metrics(first, second, mask),
				}
			)
	return rows


def append_summaries(rows: list[dict[str, object]]) -> None:
	"""Append per-case and all-case summaries without mixing comparison sides."""

	base_rows = list(rows)
	groups: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
	for row in base_rows:
		groups[(str(row["pair"]), str(row["case"]), str(row["comparison"]), str(row["region"]))].append(row)
		groups[(str(row["pair"]), "ALL", str(row["comparison"]), str(row["region"]))].append(row)
	for (pair, case, comparison, region), parts in sorted(groups.items()):
		summary = {
			"case": case,
			"pair": pair,
			"slice": "__summary__",
			"comparison": comparison,
			"region": region,
			"valid_pixels": int(sum(int(part["valid_pixels"]) for part in parts)),
		}
		for metric in ("mse", "psnr", "ssim", "ncc"):
			values = np.asarray([float(part[metric]) for part in parts], dtype=float)
			summary[metric] = float(np.nanmean(values)) if not np.isnan(values).all() else math.nan
		rows.append(summary)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", newline="", encoding="utf8") as handle:
		writer = csv.DictWriter(
			handle,
			fieldnames=[
				"case",
				"pair",
				"slice",
				"comparison",
				"region",
				"mse",
				"psnr",
				"ssim",
				"ncc",
				"valid_pixels",
			],
		)
		writer.writeheader()
		writer.writerows(rows)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="No-GT registration and fusion-consistency evaluation for completed jobs."
	)
	parser.add_argument("--output-root", required=True, type=Path)
	parser.add_argument("--out-csv", type=Path)
	parser.add_argument("--band-width", type=int, default=5)
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	out_csv = args.out_csv or args.output_root / "evaluation.csv"
	rows = []
	failures = []
	for case, pair, slice_name, slice_dir in completed_jobs(args.output_root):
		try:
			rows.extend(evaluate_job(case, pair, slice_name, slice_dir, args.band_width))
		except (FileNotFoundError, ValueError) as error:
			failures.append({"output": str(slice_dir), "error": str(error)})
	append_summaries(rows)
	write_csv(out_csv, rows)
	failure_path = out_csv.with_name(f"{out_csv.stem}_failures.json")
	failure_path.write_text(json.dumps(failures, indent=2), encoding="utf8")
	print(f"Saved {out_csv} with {len(rows)} rows")
	print(f"Evaluation failures: {len(failures)} ({failure_path})")
	return 0 if not failures else 2


if __name__ == "__main__":
	raise SystemExit(main())
