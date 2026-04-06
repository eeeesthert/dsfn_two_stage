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
            left_x = batch["left_x"].unsqueeze(0).to(device)
            right_x = batch["right_x"].unsqueeze(0).to(device)
            out = model(left, right, left_x=left_x, right_x=right_x)
            save_stage_results(out, Path(out_dir) / stage / batch["case"], prefix=f"{stage}_{i:03d}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="./dataset")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", default="./infer_outputs")
    ap.add_argument("--image-size", type=int, default=512, help="set <=0 to keep original slice size")
    ap.add_argument("--encoder-pretrain-source", choices=["imagenet", "radimagenet", "local", "none"], default="imagenet")
    ap.add_argument("--encoder-ckpt", default=None, help="required for radimagenet/local source")
    ap.add_argument("--radimagenet-url", "--net-url", dest="radimagenet_url", default=None, help="optional URL for auto-downloading RadImageNet weights")
    ap.add_argument("--encoder-strict-load", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model = TwoStageStitcher(
        encoder_pretrain_source=args.encoder_pretrain_source,
        encoder_ckpt=args.encoder_ckpt,
        encoder_radimagenet_url=args.radimagenet_url,
        encoder_strict_load=args.encoder_strict_load,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=True)

    run_stage(model, args.dataset_root, stage="12", out_dir=args.out_dir, image_size=args.image_size, device=device)
    run_stage(model, args.dataset_root, stage="23", out_dir=args.out_dir, image_size=args.image_size, device=device)


if __name__ == "__main__":
    main()
