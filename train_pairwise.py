from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from abus_pairwise.datasets import ABUSPairDataset
from abus_pairwise.pipeline import TwoStageStitcher, compute_total_loss, save_stage_results


def train_stage(args: argparse.Namespace, stage: str, model: TwoStageStitcher, device: torch.device) -> None:
    img_size = None if args.image_size <= 0 else args.image_size
    ds = ABUSPairDataset(args.dataset_root, stage=stage, image_size=img_size)
    if len(ds) == 0:
        raise RuntimeError(
            f"No training samples found for stage={stage} under {args.dataset_root}. "
            "Supported layouts: case/inputX.jpg or case/inputX/slice_xxx.jpg."
        )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.98)

    for epoch in range(args.epochs):
        model.train()
        for batch in dl:
            left = batch["left"].to(device)
            right = batch["right"].to(device)
            left_x = batch["left_x"].to(device)
            right_x = batch["right_x"].to(device)

            optim.zero_grad()
            out = model(left, right)
            losses = compute_total_loss(out, left_x, right_x)
            losses["total"].backward()
            optim.step()
        scheduler.step()

        print(
            f"[{stage}] epoch={epoch+1}/{args.epochs} "
            f"total={losses['total'].item():.4f} warp_l1={losses['warp_l1'].item():.4f} "
            f"edge={losses['grid_edge'].item():.4f} angle={losses['grid_angle'].item():.4f} "
            f"seam={losses['seam_cost'].item():.4f}"
        )

    ckpt = Path(args.out_dir) / f"stage_{stage}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt)


def export_samples(args: argparse.Namespace, stage: str, model: TwoStageStitcher, device: torch.device) -> None:
    model.eval()
    img_size = None if args.image_size <= 0 else args.image_size
    ds = ABUSPairDataset(args.dataset_root, stage=stage, image_size=img_size)
    if len(ds) == 0:
        return
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    with torch.no_grad():
        for i, batch in enumerate(dl):
            left = batch["left"].to(device)
            right = batch["right"].to(device)
            out = model(left, right)
            case = batch["case"][0]
            save_stage_results(out, Path(args.out_dir) / "results" / stage / case, prefix=f"{stage}_{i:03d}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="./dataset")
    ap.add_argument("--out-dir", default="./outputs")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--image-size", type=int, default=512, help="set <=0 to keep original slice size")
    ap.add_argument("--lr", type=float, default=1e-4)
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

    # Step-1: input1 + input2
    train_stage(args, stage="12", model=model, device=device)
    export_samples(args, stage="12", model=model, device=device)

    # Step-2: input2 + input3
    train_stage(args, stage="23", model=model, device=device)
    export_samples(args, stage="23", model=model, device=device)


if __name__ == "__main__":
    main()
