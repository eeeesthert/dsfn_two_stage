from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

DEEPMODULE_DIR = Path(__file__).resolve().parent / "contract" / "deephomography" / "Oneline-DLTv1"
sys.path.insert(0, str(DEEPMODULE_DIR))



@dataclass(frozen=True)
class ABUSPair:
    case_dir: Path
    left_path: Path
    right_path: Path


class DeepHomographyABUSDataset(Dataset):
    """ABUS pair adapter for contract/deephomography/Oneline-DLTv1.

    The original DeepHomography dataloader expects text-file image pairs under
    ``Data/Train``. This adapter scans the repository ABUS layout directly and
    returns the four tensors consumed by the Oneline-DLT model:
    ``org_img``, ``input_tensor``, ``patch_indices`` and ``h4p``.
    """

    mean_i = np.reshape(np.array([118.93, 113.97, 102.60], dtype=np.float32), (1, 1, 3))
    std_i = np.reshape(np.array([69.85, 68.81, 72.45], dtype=np.float32), (1, 1, 3))

    def __init__(
        self,
        root: str | Path,
        stage: str = "12",
        img_w: int = 0,
        img_h: int = 0,
        patch_w: int = 0,
        patch_h: int = 0,
        rho: int = 0,
    ) -> None:
        self.root = Path(root)
        self.stage = stage
        self.img_w = img_w
        self.img_h = img_h
        self.patch_w = patch_w
        self.patch_h = patch_h
        self.rho = rho
        if (img_w <= 0) != (img_h <= 0):
            raise ValueError("img_w and img_h must both be positive for resize mode, or both be <= 0 for original-size mode")
        if (patch_w <= 0) != (patch_h <= 0):
            raise ValueError("patch_w and patch_h must both be positive, or both be <= 0 to use the full image")
        if img_w > 0 and patch_w > 0 and (patch_w + 2 * rho > img_w or patch_h + 2 * rho > img_h):
            raise ValueError("patch size plus 2*rho must fit inside img_w/img_h")
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
                samples.extend(ABUSPair(case_dir, left_map[sid], right_map[sid]) for sid in common)
            else:
                samples.extend(ABUSPair(case_dir, lp, rp) for lp, rp in zip(sorted(left), sorted(right)))
        return samples

    @staticmethod
    def _make_mesh(patch_w: int, patch_h: int) -> tuple[np.ndarray, np.ndarray]:
        x_flat = np.arange(0, patch_w)[np.newaxis, :]
        y_one = np.ones(patch_h)[:, np.newaxis]
        x_mesh = np.matmul(y_one, x_flat)
        y_flat = np.arange(0, patch_h)[:, np.newaxis]
        x_one = np.ones(patch_w)[np.newaxis, :]
        y_mesh = np.matmul(y_flat, x_one)
        return x_mesh, y_mesh

    def _load_gray_norm(self, path: Path) -> np.ndarray:
        import cv2

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        if self.img_w > 0 and self.img_h > 0:
            img = cv2.resize(img, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32)
        img = (img - self.mean_i) / self.std_i
        img = np.mean(img, axis=2, keepdims=True)
        return np.transpose(img, [2, 0, 1]).astype(np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        img_1 = self._load_gray_norm(sample.left_path)
        img_2 = self._load_gray_norm(sample.right_path)
        if img_1.shape[1:] != img_2.shape[1:]:
            raise ValueError(
                "DeepHomography full-image mode requires paired inputs to have the same original size: "
                f"{sample.left_path} has {img_1.shape[2]}x{img_1.shape[1]}, "
                f"{sample.right_path} has {img_2.shape[2]}x{img_2.shape[1]}"
            )
        org_img = np.concatenate([img_1, img_2], axis=0)
        _, img_h, img_w = org_img.shape
        patch_w = self.patch_w if self.patch_w > 0 else img_w
        patch_h = self.patch_h if self.patch_h > 0 else img_h
        if patch_w > img_w or patch_h > img_h:
            raise ValueError(f"patch size {patch_w}x{patch_h} cannot exceed image size {img_w}x{img_h}")
        if (patch_w < img_w or patch_h < img_h) and (patch_w + 2 * self.rho > img_w or patch_h + 2 * self.rho > img_h):
            raise ValueError("patch size plus 2*rho must fit inside the current image")

        # Default comparison protocol: no resize and no crop.  The full original
        # pair is used as both the full image and the DeepHomography patch.
        x = 0 if patch_w == img_w else np.random.randint(self.rho, img_w - self.rho - patch_w + 1)
        y = 0 if patch_h == img_h else np.random.randint(self.rho, img_h - self.rho - patch_h + 1)
        x_mesh, y_mesh = self._make_mesh(patch_w, patch_h)

        input_tensor = org_img[:, y : y + patch_h, x : x + patch_w]
        patch_indices = ((y_mesh.reshape(-1) + y) * img_w + (x_mesh.reshape(-1) + x)).astype(np.int64)
        h4p = np.array(
            [(x, y), (x, y + patch_h), (x + patch_w, y + patch_h), (x + patch_w, y)],
            dtype=np.float32,
        ).reshape(-1)
        return torch.from_numpy(org_img), torch.from_numpy(input_tensor), torch.from_numpy(patch_indices), torch.from_numpy(h4p)


def zero_init_homography_head(net: torch.nn.Module) -> None:
    fc = getattr(net, "fc", None)
    if isinstance(fc, torch.nn.Linear) and fc.out_features == 8:
        torch.nn.init.zeros_(fc.weight)
        torch.nn.init.zeros_(fc.bias)


def train_one_stage(args: argparse.Namespace, stage: str, device: torch.device) -> Path:
    dataset = DeepHomographyABUSDataset(
        args.dataset_root, stage=stage, img_w=args.img_w, img_h=args.img_h,
        patch_w=args.patch_size_w, patch_h=args.patch_size_h, rho=args.rho,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=args.batch_size > 1, pin_memory=device.type == "cuda")
    from torch_homography_model import build_model

    net = build_model(args.model_name, pretrained=args.pretrained).to(device)
    if args.zero_init_head:
        zero_init_homography_head(net)
    if torch.cuda.device_count() > 1 and args.data_parallel:
        net = torch.nn.DataParallel(net)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, amsgrad=True, weight_decay=args.weight_decay)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    out_dir = Path(args.out_dir) / f"stage{stage}"
    out_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(args.epochs):
        net.train()
        losses: List[float] = []
        for org_img, input_tensor, patch_indices, h4p in loader:
            org_img = org_img.float().to(device, non_blocking=True)
            input_tensor = input_tensor.float().to(device, non_blocking=True)
            patch_indices = patch_indices.float().to(device, non_blocking=True)
            h4p = h4p.float().to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                out: Dict[str, torch.Tensor] = net(org_img, input_tensor, h4p, patch_indices)
                loss = out["feature_loss"].mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            global_step += 1
            if args.log_every > 0 and global_step % args.log_every == 0:
                mask_mean = float(out.get("mask_ap_mean", torch.tensor(float("nan"))).detach().cpu())
                unmasked_loss = float(out.get("feature_loss_unmasked", torch.tensor(float("nan"))).detach().cpu())
                triplet_loss = float(out.get("triplet_loss_unmasked", torch.tensor(float("nan"))).detach().cpu())
                photometric_loss = float(out.get("photometric_loss_unmasked", torch.tensor(float("nan"))).detach().cpu())
                print(
                    f"stage={stage} epoch={epoch + 1}/{args.epochs} step={global_step} "
                    f"loss={np.mean(losses[-args.log_every:]):.6f} "
                    f"unmasked_loss={unmasked_loss:.6f} triplet={triplet_loss:.6f} "
                    f"photo={photometric_loss:.6f} mask_mean={mask_mean:.6f}"
                )
        torch.save({"model": net.state_dict(), "args": vars(args), "stage": stage, "epoch": epoch + 1}, out_dir / "last.pt")
        print(f"stage={stage} epoch={epoch + 1} mean_loss={np.mean(losses):.6f}")
    return out_dir / "last.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DeepHomography baseline on ABUS pair inputs.")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/deephomography_abus"))
    parser.add_argument("--stages", nargs="+", choices=["12", "23"], default=["12", "23"])
    parser.add_argument("--img-w", type=int, default=0, help="Resize width. Use 0 to keep original width (default).")
    parser.add_argument("--img-h", type=int, default=0, help="Resize height. Use 0 to keep original height (default).")
    parser.add_argument("--patch-size-w", type=int, default=0, help="Patch width. Use 0 to use the full image without cropping (default).")
    parser.add_argument("--patch-size-h", type=int, default=0, help="Patch height. Use 0 to use the full image without cropping (default).")
    parser.add_argument("--rho", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1, help="Default 1 preserves variable original sizes without padding, resizing, or cropping.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--model-name", choices=["resnet34", "resnet50", "resnet101", "resnet152"], default="resnet34")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--zero-init-head", action="store_true", default=True, help="Initialize the 8-offset homography head to zero for stable identity-start training.")
    parser.add_argument("--no-zero-init-head", action="store_false", dest="zero_init_head")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device={device}; original_size={args.img_w <= 0 and args.img_h <= 0}; full_image_patch={args.patch_size_w <= 0 and args.patch_size_h <= 0}; batch_size={args.batch_size}; amp={args.amp}")
    for stage in args.stages:
        ckpt = train_one_stage(args, stage, device)
        print(f"saved {ckpt}")


if __name__ == "__main__":
    main()
