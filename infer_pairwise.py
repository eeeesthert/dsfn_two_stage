from __future__ import annotations

import argparse
from pathlib import Path

import torch

from abus_pairwise.datasets import ABUSPairDataset
from abus_pairwise.pipeline import TwoStageStitcher, save_stage_results


def run_stage(model: TwoStageStitcher, dataset_root: str, stage: str, out_dir: str, image_size: int, device: torch.device) -> None:
    img_size = None if image_size <= 0 else image_size
    ds = ABUSPairDataset(dataset_root, stage=stage, image_size=img_size)
    model.eval()

    with torch.no_grad():
        for i in range(len(ds)):
            batch = ds[i]
            left = batch["left"].unsqueeze(0).to(device)
            right = batch["right"].unsqueeze(0).to(device)
            out = model(left, right)
            save_stage_results(out, Path(out_dir) / stage / batch["case"], prefix=f"{stage}_{i:03d}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="./infer")
    ap.add_argument("--stage", default="12", help="23")
    ap.add_argument("--checkpoint", default = r"./outputs")
    ap.add_argument("--out-dir", default="./infer_outputs")
    ap.add_argument("--image-size", type=int, default=0, help="set <=0 to keep original slice size")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    
    ckpt = Path(args.checkpoint) / f"stage_{args.stage}.pt"
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model = TwoStageStitcher(pretrained_backbone=False).to(device)
#     model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=True)
    model.load_state_dict(torch.load(ckpt, map_location=device), strict=True)

    run_stage(model, args.dataset_root, args.stage, out_dir=args.out_dir, image_size=args.image_size, device=device)

if __name__ == "__main__":
    main()
