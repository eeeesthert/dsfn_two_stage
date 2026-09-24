"""Run the two-stage UDIS baseline on the repository's native ABUS layout."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from UDIS_PyTorch.datasets import ImagePair, scan_abus_pairs
from UDIS_PyTorch.models.alignment import AlignmentPipeline
from UDIS_PyTorch.models.reconstruction import ReconstructionNet
from UDIS_PyTorch.utils.image import load_rgb, save_image


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--dataset-root", type=Path, required=True)
	parser.add_argument("--alignment-ckpt", type=Path, required=True)
	parser.add_argument("--reconstruction-ckpt", type=Path, required=True)
	parser.add_argument("--out-dir", type=Path, default=Path("outputs/udis_abus"))
	parser.add_argument(
		"--stages", nargs="+", choices=("12", "23"), default=("12", "23")
	)
	parser.add_argument("--cpu", action="store_true")
	return parser.parse_args()


def load_models(args: argparse.Namespace, device: torch.device):
	alignment = AlignmentPipeline().to(device)
	reconstruction = ReconstructionNet().to(device)
	alignment_state = torch.load(args.alignment_ckpt, map_location=device)
	reconstruction_state = torch.load(args.reconstruction_ckpt, map_location=device)
	alignment.net.load_state_dict(alignment_state["model"])
	reconstruction.load_state_dict(reconstruction_state["model"])
	alignment.eval()
	reconstruction.eval()
	return alignment, reconstruction


def output_directory(root: Path, pair: ImagePair) -> Path:
	"""Use the same stage/case convention as the repository evaluation tools."""
	path = root / pair.stage / pair.case / "fusion"
	path.mkdir(parents=True, exist_ok=True)
	return path


def save_result(root: Path, pair: ImagePair, aligned, reconstructed) -> dict[str, str]:
	prefix = f"{pair.stage}_{pair.slice_id}"
	directory = output_directory(root, pair)
	paths = {
		"stitched": directory / f"{prefix}_stitched.png",
		"warp1": directory / f"{prefix}_warp1.png",
		"warp2": directory / f"{prefix}_warp2.png",
		"mask_left": directory / f"{prefix}_mask_left_soft.png",
		"mask_right": directory / f"{prefix}_mask_right_soft.png",
	}
	save_image(reconstructed["hr"][0], paths["stitched"])
	save_image(aligned["warp1"][0], paths["warp1"])
	save_image(aligned["warp2"][0], paths["warp2"])
	save_image(aligned["mask1"][0], paths["mask_left"], mask=True)
	save_image(aligned["mask2"][0], paths["mask_right"], mask=True)
	return {key: str(value) for key, value in paths.items()}


def run_pair(
	pair: ImagePair, alignment, reconstruction, device: torch.device, root: Path
):
	image1 = load_rgb(pair.image1)[None].to(device)
	image2 = load_rgb(pair.image2)[None].to(device)
	if image1.shape != image2.shape:
		raise ValueError(
			"ABUS pair dimensions differ and UDIS does not crop or resize original input: "
			f"{pair.image1}={tuple(image1.shape[-2:])}, {pair.image2}={tuple(image2.shape[-2:])}"
		)
	with torch.no_grad():
		aligned = alignment(image1, image2)
		reconstructed = reconstruction(aligned["warp1"], aligned["warp2"])
	return save_result(root, pair, aligned, reconstructed)


def main() -> None:
	args = parse_args()
	device = torch.device(
		"cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
	)
	alignment, reconstruction = load_models(args, device)
	rows = []
	for stage in args.stages:
		for pair in scan_abus_pairs(args.dataset_root, stage):
			paths = run_pair(pair, alignment, reconstruction, device, args.out_dir)
			rows.append(
				{"case": pair.case, "slice": pair.slice_id, "stage": stage, **paths}
			)
			print(f"stage={stage} case={pair.case} slice={pair.slice_id}")

	args.out_dir.mkdir(parents=True, exist_ok=True)
	with (args.out_dir / "outputs.csv").open(
		"w", newline="", encoding="utf8"
	) as stream:
		writer = csv.DictWriter(
			stream, fieldnames=rows[0].keys() if rows else ("case", "slice", "stage")
		)
		writer.writeheader()
		writer.writerows(rows)
	print(f"saved {len(rows)} ABUS results to {args.out_dir}")


if __name__ == "__main__":
	main()
