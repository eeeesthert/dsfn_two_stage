"""Train one shared DunHuangStitch alignment model on both ABUS view pairs."""

import argparse

import torch
from torch.utils.data import DataLoader

from datasets.abus_pair_dataset import ABUSPairDataset
from losses.alignment_loss import AlignmentLoss
from models.alignment.alignment_net import AlignmentModel
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.config import config
from utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--config", default="configs/alignment.yaml")
    parser.add_argument("--output", default="checkpoints/abus_alignment")
    parser.add_argument("--stages", nargs="+", choices=("12", "23"), default=("12", "23"))
    parser.add_argument("--image-size", type=int, default=0, help="0 preserves native ABUS resolution")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--resume")
    args = parser.parse_args()

    cfg = config(args.config)
    cfg["data"].update(root=args.dataset_root, image_size=None if args.image_size <= 0 else [args.image_size] * 2)
    cfg["training"]["batch_size"] = args.batch_size
    if args.workers is not None:
        cfg["training"]["workers"] = args.workers
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    model = AlignmentModel(cfg).to(device)
    train = cfg["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=train["lr"], betas=tuple(train["betas"]), weight_decay=train["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, train["scheduler_gamma"])
    start, best = 0, float("inf")
    if args.resume:
        state = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        start, best = state["epoch"] + 1, state["best_metric"]
    dataset = ABUSPairDataset(args.dataset_root, args.stages, cfg["data"]["image_size"])
    loader = DataLoader(dataset, args.batch_size, shuffle=True, num_workers=train["workers"])
    criterion = AlignmentLoss(cfg["loss"]["stage_weights"])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"] and device.type == "cuda")
    for epoch in range(start, train["epochs"]):
        total = 0.0
        for batch in loader:
            reference = batch["reference"].to(device)
            target = batch["target"].to(device)
            optimizer.zero_grad()
            with torch.autocast(device.type, enabled=scaler.is_enabled()):
                losses = criterion(reference, model(reference, target))
            scaler.scale(losses["loss"]).backward()
            scaler.step(optimizer)
            scaler.update()
            total += losses["loss"].item()
        if epoch >= train["warmup_epochs"]:
            scheduler.step()
        average = total / len(loader)
        print(f"epoch={epoch} lr={optimizer.param_groups[0]['lr']:.6g} loss={average:.6f}")
        save_checkpoint(f"{args.output}/last.pt", model, optimizer, scheduler, epoch, best, cfg)
        if average < best:
            best = average
            save_checkpoint(f"{args.output}/best.pt", model, optimizer, scheduler, epoch, best, cfg)


if __name__ == "__main__":
    main()
