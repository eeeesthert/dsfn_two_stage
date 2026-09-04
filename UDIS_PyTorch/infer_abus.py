"""Run the unchanged two-stage UDIS method over an ABUS case hierarchy."""

import sys
from pathlib import Path as _BootstrapPath

sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parent.parent))

import argparse
from pathlib import Path

import torch

from UDIS_PyTorch.datasets import ABUSPairDataset
from UDIS_PyTorch.models.alignment import AlignmentPipeline
from UDIS_PyTorch.models.reconstruction_net import ReconstructionNet
from UDIS_PyTorch.utils.utilities import save_image


def load_models(arguments, device):
	"""Load independently trained UDIS alignment and reconstruction stages."""
	align = AlignmentPipeline().to(device)
	reconstruct = ReconstructionNet().to(device)
	align_state = torch.load(arguments.alignment_ckpt, map_location=device)
	reconstruction_state = torch.load(arguments.reconstruction_ckpt, map_location=device)
	align.net.load_state_dict(align_state.get("model", align_state))
	reconstruct.load_state_dict(reconstruction_state.get("model", reconstruction_state))
	align.eval()
	reconstruct.eval()
	return align, reconstruct


def save_eval_pairwise_result(output, root, stage, case_name, slice_name):
	"""Write the names and hierarchy consumed by ``eval_pairwise_no_gt.py``."""
	fusion_dir = root / stage / case_name / "fusion"
	warp_dir = root / stage / case_name / "warp"
	fusion_dir.mkdir(parents=True, exist_ok=True)
	warp_dir.mkdir(parents=True, exist_ok=True)
	prefix = f"{stage}_{slice_name}"
	save_image(output["stitched"][0], fusion_dir / f"{prefix}_stitched.png")
	save_image(output["mask1"][0], fusion_dir / f"{prefix}_mask_left_soft.png", mask=True)
	save_image(output["mask2"][0], fusion_dir / f"{prefix}_mask_right_soft.png", mask=True)
	save_image(output["warp1"][0], warp_dir / f"{prefix}_warp_left.png")
	save_image(output["warp2"][0], warp_dir / f"{prefix}_warp_right.png")


def run_stage(align, reconstruct, dataset_root, output_root, stage, device):
	dataset = ABUSPairDataset(dataset_root, stage)
	for sample in dataset:
		image1 = sample["image1"][None].to(device)
		image2 = sample["image2"][None].to(device)
		aligned = align(image1, image2)
		reconstructed = reconstruct(aligned["warp1"], aligned["warp2"])
		result = {
			**aligned,
			"stitched": reconstructed["hr"],
		}
		save_eval_pairwise_result(
			result,
			output_root,
			stage,
			sample["case"],
			sample["slice"],
		)


def main():
	parser = argparse.ArgumentParser(description="UDIS inference for three-view ABUS slices")
	parser.add_argument("--dataset-root", default="./dataset")
	parser.add_argument("--alignment-ckpt", required=True)
	parser.add_argument("--reconstruction-ckpt", required=True)
	parser.add_argument("--out-dir", default="./udis_outputs/results")
	parser.add_argument("--cpu", action="store_true")
	arguments = parser.parse_args()
	device = torch.device("cpu" if arguments.cpu or not torch.cuda.is_available() else "cuda")
	align, reconstruct = load_models(arguments, device)
	with torch.no_grad():
		for stage in ("12", "23"):
			run_stage(
				align,
				reconstruct,
				arguments.dataset_root,
				Path(arguments.out_dir),
				stage,
				device,
			)


if __name__ == "__main__":
	main()
