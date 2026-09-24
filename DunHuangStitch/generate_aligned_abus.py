"""Freeze alignment and generate the ABUS dataset used to train fusion."""

import argparse
from pathlib import Path

import numpy as np
import torch

from datasets.abus_pair_dataset import ABUSPairDataset
from models.alignment.alignment_net import AlignmentModel
from utils.canvas import compute_union_canvas
from utils.checkpoint import load_checkpoint
from utils.image import save_tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", default="aligned_abus")
    parser.add_argument("--stages", nargs="+", choices=("12", "23"), default=("12", "23"))
    parser.add_argument("--image-size", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = AlignmentModel(state["config"]).to(device).eval()
    load_checkpoint(args.checkpoint, model, map_location=device)
    size = None if args.image_size <= 0 else (args.image_size, args.image_size)
    dataset = ABUSPairDataset(args.dataset_root, args.stages, size)
    with torch.inference_mode():
        for sample in dataset:
            reference = sample["reference"][None].to(device)
            target = sample["target"][None].to(device)
            homography = model(reference, target)["H3"]
            wr, wt, mr, mt, translation = compute_union_canvas(reference, target, homography)
            directory = Path(args.output_root) / sample["stage"] / sample["case"] / f"slice_{sample['slice_id']}"
            save_tensor(wr[0], directory / "reference.png")
            save_tensor(wt[0], directory / "target.png")
            save_tensor(mr[0], directory / "mask_reference.png")
            save_tensor(mt[0], directory / "mask_target.png")
            np.savez(directory / "transform.npz", homography=homography[0].cpu().numpy(), translation=translation[0].cpu().numpy())
            print(f"saved {directory}")


if __name__ == "__main__":
    main()
