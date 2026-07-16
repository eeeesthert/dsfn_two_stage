from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

UDIS2_WARP_DIR = Path(__file__).resolve().parent / "contract" / "udis2" / "Warp" / "Codes"
sys.path.insert(0, str(UDIS2_WARP_DIR))

from loss import cal_lp_loss, inter_grid_loss, intra_grid_loss  # noqa: E402
from network import Network, build_model  # noqa: E402


@dataclass(frozen=True)
class ABUSPair:
    case: str
    left_path: Path
    right_path: Path


class UDIS2ABUSDataset(Dataset):
    """ABUS adapter for UDIS2 Warp training without resizing or cropping."""

    def __init__(self, root: str | Path, stage: str = "12", exchange: bool = False) -> None:
        self.root = Path(root)
        self.stage = stage
        self.exchange = exchange
        self.samples = self._scan_cases()
        if not self.samples:
            raise ValueError(f"No ABUS pairs found under {self.root} for stage {stage}")

    @staticmethod
    def _slice_stem_to_id(stem: str) -> str:
        return stem.split("slice_", 1)[1] if stem.startswith("slice_") else stem

    @staticmethod
    def _collect_images(path: Path) -> List[Path]:
        if path.is_file():
            return [path]
        if not path.exists():
            return []
        files: List[Path] = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            files.extend(sorted(path.glob(ext)))
        return files

    def _scan_cases(self) -> List[ABUSPair]:
        left_name, right_name = {"12": ("input1", "input2"), "23": ("input2", "input3")}[self.stage]
        samples: List[ABUSPair] = []
        for case_dir in sorted(self.root.glob("case*")):
            left = self._collect_images(case_dir / f"{left_name}.jpg") or self._collect_images(case_dir / left_name)
            right = self._collect_images(case_dir / f"{right_name}.jpg") or self._collect_images(case_dir / right_name)
            if not left or not right:
                continue
            left_map = {self._slice_stem_to_id(p.stem): p for p in left}
            right_map = {self._slice_stem_to_id(p.stem): p for p in right}
            common = sorted(set(left_map).intersection(right_map))
            if common:
                samples.extend(ABUSPair(case_dir.name, left_map[sid], right_map[sid]) for sid in common)
            else:
                samples.extend(ABUSPair(case_dir.name, lp, rp) for lp, rp in zip(sorted(left), sorted(right)))
        return samples

    @staticmethod
    def _load_image(path: Path) -> torch.Tensor:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        img = img.astype(np.float32)
        img = (img / 127.5) - 1.0
        return torch.from_numpy(np.transpose(img, [2, 0, 1])).float()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str, str, str]:
        sample = self.samples[index]
        left = self._load_image(sample.left_path)
        right = self._load_image(sample.right_path)
        if left.shape[1:] != right.shape[1:]:
            raise ValueError(
                "UDIS2 no-resize/no-crop training requires paired inputs with the same original size: "
                f"{sample.left_path} has {left.shape[2]}x{left.shape[1]}, "
                f"{sample.right_path} has {right.shape[2]}x{right.shape[1]}"
            )
        if self.exchange and torch.randint(0, 2, ()).item() == 1:
            left, right = right, left
        return left, right, sample.case, str(sample.left_path), str(sample.right_path)


def train_stage(args: argparse.Namespace, stage: str, device: torch.device) -> Path:
    dataset = UDIS2ABUSDataset(args.dataset_root, stage=stage, exchange=args.exchange)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=args.batch_size > 1, pin_memory=device.type == "cuda")
    net = Network(pretrained=args.imagenet_pretrained).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    out_dir = Path(args.out_dir) / f"stage{stage}"
    out_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(args.epochs):
        net.train()
        totals: list[float] = []
        for input1, input2, _case, _lp, _rp in loader:
            input1 = input1.to(device, non_blocking=True)
            input2 = input2.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = build_model(net, input1, input2, is_training=not args.no_aug)
            overlap_loss = cal_lp_loss(input1, input2, out["output_H"], out["output_H_inv"], out["warp_mesh"], out["warp_mesh_mask"])
            _, _, img_h, img_w = input1.shape
            nonoverlap_loss = args.nonoverlap_weight * (
                inter_grid_loss(out["overlap"], out["mesh2"]) + intra_grid_loss(out["mesh2"], img_h=img_h, img_w=img_w)
            )
            total_loss = overlap_loss + nonoverlap_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=args.grad_clip, norm_type=2)
            optimizer.step()
            totals.append(float(total_loss.detach().cpu()))
            global_step += 1
            if args.log_every > 0 and global_step % args.log_every == 0:
                print(f"stage={stage} epoch={epoch+1}/{args.epochs} step={global_step} total={np.mean(totals[-args.log_every:]):.6f} overlap={overlap_loss.item():.6f} nonoverlap={nonoverlap_loss.item():.6f}")
        scheduler.step()
        ckpt = out_dir / "last.pth"
        torch.save({"model": net.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch + 1, "args": vars(args), "stage": stage}, ckpt)
        print(f"stage={stage} epoch={epoch+1} mean_loss={np.mean(totals):.6f} saved={ckpt}")
    return out_dir / "last.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train UDIS2 Warp baseline on ABUS pairs with no resize and no crop.")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/udis2_abus"))
    parser.add_argument("--stages", nargs="+", choices=["12", "23"], default=["12", "23"])
    parser.add_argument("--batch-size", type=int, default=1, help="Keep 1 for variable original sizes on 24GB GPUs.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-gamma", type=float, default=0.97)
    parser.add_argument("--nonoverlap-weight", type=float, default=10.0)
    parser.add_argument("--grad-clip", type=float, default=3.0)
    parser.add_argument("--exchange", action="store_true", help="Randomly swap pair order, matching original UDIS2 augmentation.")
    parser.add_argument("--no-aug", action="store_true", help="Disable UDIS2 brightness/color augmentation.")
    parser.add_argument("--imagenet-pretrained", action="store_true", default=True, help="Initialize the UDIS2 ResNet-50 feature extractor with ImageNet weights.")
    parser.add_argument("--no-imagenet-pretrained", action="store_false", dest="imagenet_pretrained")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Using device={device}; torch={torch.__version__}; no_resize=True; no_crop=True; batch_size={args.batch_size}")
    for stage in args.stages:
        ckpt = train_stage(args, stage, device)
        print(f"saved {ckpt}")


if __name__ == "__main__":
    main()
