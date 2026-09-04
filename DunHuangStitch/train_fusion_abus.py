"""Train DunHuangStitch fusion on generated ABUS aligned pairs."""

import argparse

import torch
from torch.utils.data import DataLoader

from datasets.abus_pair_dataset import ABUSAlignedPairDataset
from losses.seam_loss import SeamLoss
from models.fusion.fusion_net import FusionModel
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.config import config
from utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned-root", required=True)
    parser.add_argument("--config", default="configs/fusion.yaml")
    parser.add_argument("--output", default="checkpoints/abus_fusion")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--resume")
    args = parser.parse_args()
    cfg = config(args.config)
    cfg["data"].update(root=args.aligned_root, image_size=None)
    cfg["training"]["batch_size"] = args.batch_size
    if args.workers is not None:
        cfg["training"]["workers"] = args.workers
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    model = FusionModel(cfg).to(device)
    train = cfg["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=train["lr"], betas=tuple(train["betas"]), weight_decay=train["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, train["scheduler_gamma"])
    start, best = 0, float("inf")
    if args.resume:
        state = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        start, best = state["epoch"] + 1, state["best_metric"]
    loader = DataLoader(ABUSAlignedPairDataset(args.aligned_root), args.batch_size, shuffle=True, num_workers=train["workers"])
    criterion = SeamLoss(**cfg["fusion_loss"])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"] and device.type == "cuda")
    for epoch in range(start, train["epochs"]):
        total = 0.0
        for batch in loader:
            inputs = [batch[key].to(device) for key in ("I_wr", "I_wt", "M_wr", "M_wt")]
            optimizer.zero_grad()
            with torch.autocast(device.type, enabled=scaler.is_enabled()):
                losses = criterion(*inputs, model(*inputs))
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
