"""Run the two-stage DunHuangStitch baseline over an ABUS dataset."""

import argparse
from pathlib import Path

import numpy as np
import torch

from datasets.abus_pair_dataset import ABUSPairDataset
from models.alignment.alignment_net import AlignmentModel
from models.fusion.fusion_net import FusionModel
from utils.canvas import compute_union_canvas
from utils.checkpoint import load_checkpoint
from utils.image import save_tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--alignment-checkpoint", required=True)
    parser.add_argument("--fusion-checkpoint", required=True)
    parser.add_argument("--output-root", default="outputs/dunhuangstitch_abus")
    parser.add_argument("--stages", nargs="+", choices=("12", "23"), default=("12", "23"))
    parser.add_argument("--image-size", type=int, default=0, help="0 preserves native ABUS resolution")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    alignment_state = torch.load(args.alignment_checkpoint, map_location="cpu", weights_only=False)
    fusion_state = torch.load(args.fusion_checkpoint, map_location="cpu", weights_only=False)
    alignment = AlignmentModel(alignment_state["config"]).to(device).eval()
    fusion = FusionModel(fusion_state["config"]).to(device).eval()
    load_checkpoint(args.alignment_checkpoint, alignment, map_location=device)
    load_checkpoint(args.fusion_checkpoint, fusion, map_location=device)
    size = None if args.image_size <= 0 else (args.image_size, args.image_size)
    dataset = ABUSPairDataset(args.dataset_root, args.stages, size)

    with torch.inference_mode():
        for sample in dataset:
            reference = sample["reference"][None].to(device)
            target = sample["target"][None].to(device)
            homography = alignment(reference, target)["H3"]
            wr, wt, mr, mt, translation = compute_union_canvas(reference, target, homography)
            output = fusion(wr, wt, mr, mt)
            prefix = f"{sample['stage']}_{sample['slice_id']}"
            root = Path(args.output_root) / sample["stage"] / sample["case"]
            warp_dir, fusion_dir = root / "warp", root / "fusion"
            save_tensor(wr[0], warp_dir / f"{prefix}_reference.png")
            save_tensor(wt[0], warp_dir / f"{prefix}_target.png")
            save_tensor(mr[0], warp_dir / f"{prefix}_mask_reference.png")
            save_tensor(mt[0], warp_dir / f"{prefix}_mask_target.png")
            save_tensor(output["stitched"][0], fusion_dir / f"{prefix}_stitched.png")
            save_tensor(output["seam_mask_r"][0], fusion_dir / f"{prefix}_mask_left_soft.png")
            save_tensor(output["seam_mask_t"][0], fusion_dir / f"{prefix}_mask_right_soft.png")
            np.savez(root / f"{prefix}_transform.npz", homography=homography[0].cpu().numpy(), translation=translation[0].cpu().numpy())
            print(f"saved {sample['stage']}/{sample['case']}/{prefix}")


if __name__ == "__main__":
    main()
