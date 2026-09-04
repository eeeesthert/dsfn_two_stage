"""Freeze Stage-1 results as the four-directory Stage-2 dataset."""

import sys
from pathlib import Path as _BootstrapPath

sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parent.parent))
import argparse, csv
from pathlib import Path
import torch
from UDIS_PyTorch.models.alignment import AlignmentPipeline
from UDIS_PyTorch.utils.utilities import load_rgb, save_image


def main():
	p = argparse.ArgumentParser()
	p.add_argument("--checkpoint", required=True)
	p.add_argument("--input", required=True, help="pair manifest CSV")
	p.add_argument("--output", required=True)
	p.add_argument("--split", default="training")
	a = p.parse_args()
	dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	pipe = AlignmentPipeline().to(dev)
	pipe.net.load_state_dict(torch.load(a.checkpoint, map_location=dev)["model"])
	pipe.eval()
	root = Path(a.output) / a.split
	for d in ("warp1", "warp2", "mask1", "mask2"):
		(root / d).mkdir(parents=True, exist_ok=True)
	with open(a.input, newline="") as f:
		for i, row in enumerate(csv.reader(f)):
			if not row or row[0] == "image1":
				continue
			x, y = load_rgb(row[0])[None].to(dev), load_rgb(row[1])[None].to(dev)
			with torch.no_grad():
				o = pipe(x, y)
			name = f"{i:06d}.png"
			for k in ("warp1", "warp2", "mask1", "mask2"):
				save_image(o[k][0], root / k / name, k.startswith("mask"))


if __name__ == "__main__":
	main()
