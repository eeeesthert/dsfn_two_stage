from __future__ import annotations

import argparse
from pathlib import Path

import torch

from abus_pairwise.pipeline import TwoStageStitcher, save_stage_results
from abus_pairwise.temporal import ABUSTemporalPairDataset, TemporalStackMode


def run_stage(
    model: TwoStageStitcher,
    dataset_root: str,
    stage: str,
    out_dir: str,
    image_size: int,
    radius: int,
    stack_mode: TemporalStackMode,
    device: torch.device,
) -> None:
    img_size = None if image_size <= 0 else image_size
    ds = ABUSTemporalPairDataset(
        dataset_root,
        stage=stage,
        image_size=img_size,
        radius=radius,
        stack_mode=stack_mode,
    )
    model.eval()

    with torch.no_grad():
        for i in range(len(ds)):
            batch = ds[i]
            left = batch["left"].unsqueeze(0).to(device)
            right = batch["right"].unsqueeze(0).to(device)
            left_context = batch["left_context"].unsqueeze(0).to(device)
            right_context = batch["right_context"].unsqueeze(0).to(device)
            left_x = batch["left_x"].unsqueeze(0).to(device)
            right_x = batch["right_x"].unsqueeze(0).to(device)
            out = model(
                left,
                right,
                left_context=left_context,
                right_context=right_context,
                left_x=left_x,
                right_x=right_x,
            )
            save_stage_results(
                out,
                Path(out_dir) / stage / batch["case"],
                prefix=f"{stage}_{i:03d}",
            )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Pairwise inference using target slice plus adjacent temporal/depth frames."
    )
    ap.add_argument("--dataset-root", default="./dataset")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", default="./infer_outputs_temporal11")
    ap.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="set <=0 to keep original slice size",
    )
    ap.add_argument(
        "--temporal-radius",
        type=int,
        default=5,
        help="number of adjacent slices on each side; 5 means 11 frames total",
    )
    ap.add_argument(
        "--temporal-stack-mode",
        choices=["stack", "mean", "max", "center"],
        default="stack",
        help="how to build model context; stack keeps all 11 RGB frames as 33 channels",
    )
    ap.add_argument(
        "--encoder-pretrain-source",
        choices=["imagenet", "radimagenet", "local", "none"],
        default="imagenet",
    )
    ap.add_argument(
        "--encoder-ckpt",
        default=None,
        help="required for radimagenet/local source",
    )
    ap.add_argument(
        "--radimagenet-url",
        "--net-url",
        dest="radimagenet_url",
        default=None,
        help="optional URL for auto-downloading RadImageNet weights",
    )
    ap.add_argument("--encoder-strict-load", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    input_channels = (2 * args.temporal_radius + 1) * 3 if args.temporal_stack_mode == "stack" else 3
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model = TwoStageStitcher(
        encoder_pretrain_source=args.encoder_pretrain_source,
        encoder_ckpt=args.encoder_ckpt,
        encoder_radimagenet_url=args.radimagenet_url,
        encoder_strict_load=args.encoder_strict_load,
        input_channels=input_channels,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=True)

    run_stage(
        model,
        args.dataset_root,
        stage="12",
        out_dir=args.out_dir,
        image_size=args.image_size,
        radius=args.temporal_radius,
        stack_mode=args.temporal_stack_mode,
        device=device,
    )
    run_stage(
        model,
        args.dataset_root,
        stage="23",
        out_dir=args.out_dir,
        image_size=args.image_size,
        radius=args.temporal_radius,
        stack_mode=args.temporal_stack_mode,
        device=device,
    )


if __name__ == "__main__":
    main()
