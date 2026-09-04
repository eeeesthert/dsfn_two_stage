"""Complete UDIS inference: alignment, union warp, then reconstruction."""

import sys
from pathlib import Path as _BootstrapPath

sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parent.parent))
import argparse
from pathlib import Path
import torch
from UDIS_PyTorch.models.alignment import AlignmentPipeline
from UDIS_PyTorch.models.reconstruction_net import ReconstructionNet
from UDIS_PyTorch.utils.utilities import load_rgb, save_image


def main():
	p = argparse.ArgumentParser()
	p.add_argument("--image1", required=True)
	p.add_argument("--image2", required=True)
	p.add_argument("--alignment_ckpt", required=True)
	p.add_argument("--reconstruction_ckpt", required=True)
	p.add_argument("--output", required=True)
	p.add_argument("--save_intermediates", action="store_true")
	a = p.parse_args()
	dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	align = AlignmentPipeline().to(dev)
	rec = ReconstructionNet().to(dev)
	align.net.load_state_dict(torch.load(a.alignment_ckpt, map_location=dev)["model"])
	rec.load_state_dict(torch.load(a.reconstruction_ckpt, map_location=dev)["model"])
	align.eval()
	rec.eval()
	x, y = load_rgb(a.image1)[None].to(dev), load_rgb(a.image2)[None].to(dev)
	with torch.no_grad():
		o = align(x, y)
		r = rec(o["warp1"], o["warp2"])
	save_image(r["hr"][0], a.output)
	if a.save_intermediates:
		root = Path(a.output).parent
		for name, t, mask in [
			("01_image1.jpg", x[0], False),
			("02_image2.jpg", y[0], False),
			("03_warp1.jpg", o["warp1"][0], False),
			("04_warp2.jpg", o["warp2"][0], False),
			("05_mask1.png", o["mask1"][0], True),
			("06_mask2.png", o["mask2"][0], True),
			("07_lr_result.jpg", r["lr"][0], False),
			("08_final_stitched.jpg", r["hr"][0], False),
		]:
			save_image(t, root / name, mask)


if __name__ == "__main__":
	main()
