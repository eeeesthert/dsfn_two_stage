"""Evaluate ABUS UDIS alignment without requiring a ground-truth panorama."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


def read(path: Path, grayscale: bool = False) -> np.ndarray:
	mode = "L" if grayscale else "RGB"
	return np.asarray(Image.open(path).convert(mode), dtype=np.float64)


def metrics(
	image1: np.ndarray, image2: np.ndarray, valid: np.ndarray
) -> dict[str, float | int]:
	"""Return overlap MSE, PSNR and normalized cross correlation."""
	pixels1 = image1[valid]
	pixels2 = image2[valid]
	if pixels1.size == 0:
		return {"mse": np.nan, "psnr": np.nan, "ncc": np.nan, "valid_pixels": 0}
	difference = pixels1 - pixels2
	mse = float(np.mean(difference**2))
	psnr = 99.0 if mse <= 1e-12 else float(10 * np.log10(255**2 / mse))
	centered1 = pixels1 - pixels1.mean()
	centered2 = pixels2 - pixels2.mean()
	denominator = np.linalg.norm(centered1) * np.linalg.norm(centered2)
	ncc = (
		0.0
		if denominator <= 1e-12
		else float(np.dot(centered1, centered2) / denominator)
	)
	return {"mse": mse, "psnr": psnr, "ncc": ncc, "valid_pixels": int(valid.sum())}


def evaluate(root: Path) -> list[dict[str, object]]:
	rows = []
	for warp1_path in sorted(root.glob("*/*/fusion/*_warp1.png")):
		prefix = warp1_path.name.removesuffix("_warp1.png")
		directory = warp1_path.parent
		warp2_path = directory / f"{prefix}_warp2.png"
		mask1_path = directory / f"{prefix}_mask_left_soft.png"
		mask2_path = directory / f"{prefix}_mask_right_soft.png"
		if not all(path.exists() for path in (warp2_path, mask1_path, mask2_path)):
			continue
		image1 = read(warp1_path, grayscale=True)
		image2 = read(warp2_path, grayscale=True)
		valid = (read(mask1_path, grayscale=True) > 127) & (
			read(mask2_path, grayscale=True) > 127
		)
		stage, slice_id = prefix.split("_", 1)
		rows.append(
			{
				"case": directory.parent.name,
				"slice": slice_id,
				"stage": stage,
				**metrics(image1, image2, valid),
			}
		)
	return rows


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--result-root", type=Path, default=Path("outputs/udis_abus"))
	parser.add_argument("--out-csv", type=Path, default=None)
	args = parser.parse_args()
	rows = evaluate(args.result_root)
	out_csv = args.out_csv or args.result_root / "alignment_metrics.csv"
	out_csv.parent.mkdir(parents=True, exist_ok=True)
	with out_csv.open("w", newline="", encoding="utf8") as stream:
		fieldnames = ("case", "slice", "stage", "mse", "psnr", "ncc", "valid_pixels")
		writer = csv.DictWriter(stream, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)
	print(f"saved {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
	main()
