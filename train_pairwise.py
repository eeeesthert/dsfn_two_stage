from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from abus_pairwise.datasets import ABUSPairDataset
from abus_pairwise.pipeline import LossWeights, TwoStageStitcher, compute_total_loss, save_stage_results


def train_stage(args: argparse.Namespace, stage: str, model: TwoStageStitcher, device: torch.device) -> None:
    ds = ABUSPairDataset(args.dataset_root, stage=stage, image_size=args.image_size)
    if len(ds) == 0:
        raise RuntimeError(
            f"No training samples found for stage={stage} under {args.dataset_root}. "
            "Supported layouts: case/inputX.jpg or case/inputX/slice_xxx.jpg."
        )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    lw = LossWeights()

    for epoch in range(args.epochs):
        model.train()
        for batch in dl:
            left = batch["left"].to(device)
            right = batch["right"].to(device)
            left_x = batch["left_x"].to(device)
            right_x = batch["right_x"].to(device)

            optim.zero_grad()
            out = model(left, right)
            losses = compute_total_loss(out, left_x, right_x, lw)
            losses["total"].backward()
            optim.step()

        print(
            f"[{stage}] epoch={epoch+1}/{args.epochs} "
            f"total={losses['total'].item():.4f} warp={losses['warp_align'].item():.4f} "
            f"nipple={losses['nipple_prior'].item():.4f} xheat={losses['x_heat'].item():.4f}"
        )

    ckpt = Path(args.out_dir) / f"stage_{stage}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt)


def export_samples(args: argparse.Namespace, stage: str, model: TwoStageStitcher, device: torch.device) -> None:
    model.eval()
    ds = ABUSPairDataset(args.dataset_root, stage=stage, image_size=args.image_size)
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
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model = TwoStageStitcher(pretrained_backbone=True).to(device)

    # Step-1: input1 + input2
    train_stage(args, stage="12", model=model, device=device)
    export_samples(args, stage="12", model=model, device=device)

    # Step-2: input2 + input3
    train_stage(args, stage="23", model=model, device=device)
    export_samples(args, stage="23", model=model, device=device)


if __name__ == "__main__":
    main()
