"""Freeze Stage-1 warps and masks for independent reconstruction training."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from UDIS_PyTorch.datasets import ImagePair, scan_abus_pairs
from UDIS_PyTorch.models.alignment import AlignmentPipeline
from UDIS_PyTorch.utils.image import load_rgb, save_image


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--checkpoint", type=Path, required=True)
	source = parser.add_mutually_exclusive_group(required=True)
	source.add_argument("--input", type=Path, help="Optional image1,image2 CSV")
	source.add_argument("--dataset-root", type=Path, help="Native ABUS case directory")
	parser.add_argument("--stages", nargs="+", choices=("12", "23"), default=("12", "23"))
	parser.add_argument("--output", type=Path, required=True)
	parser.add_argument("--split", default="training")
	parser.add_argument("--cpu", action="store_true")
	return parser.parse_args()


def read_manifest(path: Path) -> list[ImagePair]:
	with path.open(newline="", encoding="utf8") as stream:
		rows = [row for row in csv.reader(stream) if row and row[0] != "image1"]
	return [
		ImagePair(Path(row[0]), Path(row[1]), slice_id=f"{index:06d}")
		for index, row in enumerate(rows)
	]


def collect_pairs(args: argparse.Namespace) -> list[ImagePair]:
	if args.input:
		return read_manifest(args.input)
	return [
		pair
		for stage in args.stages
		for pair in scan_abus_pairs(args.dataset_root, stage)
	]


def output_name(pair: ImagePair) -> str:
	if pair.case:
		return f"{pair.stage}_{pair.case}_{pair.slice_id}.png"
	return f"{pair.slice_id}.png"


def main() -> None:
	args = parse_args()
	device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
	pipeline = AlignmentPipeline().to(device)
	state = torch.load(args.checkpoint, map_location=device)
	pipeline.net.load_state_dict(state["model"])
	pipeline.eval()

	output_root = args.output / args.split
	for directory in ("warp1", "warp2", "mask1", "mask2"):
		(output_root / directory).mkdir(parents=True, exist_ok=True)

	pairs = collect_pairs(args)
	if not pairs:
		raise ValueError("no input pairs were found")
	for pair in pairs:
		image1 = load_rgb(pair.image1)[None].to(device)
		image2 = load_rgb(pair.image2)[None].to(device)
		if image1.shape != image2.shape:
			raise ValueError(f"paired image sizes differ: {pair.image1}, {pair.image2}")
		with torch.no_grad():
			output = pipeline(image1, image2)
		name = output_name(pair)
		for key in ("warp1", "warp2", "mask1", "mask2"):
			save_image(output[key][0], output_root / key / name, mask=key.startswith("mask"))
		print(f"saved {name}")


if __name__ == "__main__":
	main()
