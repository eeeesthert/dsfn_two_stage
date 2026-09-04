from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import torch
from torch.utils.data import Dataset


PAIR_MAP = {"12": ("input1", "input2"), "23": ("input2", "input3")}
IMAGE_PATTERNS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")


def slice_stem_to_id(stem: str) -> str:
    return stem[len("slice_") :] if stem.startswith("slice_") else stem


def _images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    result: list[Path] = []
    for pattern in IMAGE_PATTERNS:
        result.extend(path.glob(pattern))
    return sorted(result)


@dataclass(frozen=True)
class ABUSSample:
    case: str
    stage: str
    slice_id: str
    path1: Path
    path2: Path
    fallback_pairing: bool = False


class PixelStitchABUSDataset(Dataset):
    """Read existing ABUS slice pairs without augmentation, cropping, or reordering."""

    def __init__(
        self,
        root: str | Path,
        stage: str,
        image_size: int = 0,
        case: str | None = None,
        slice_id: str | None = None,
    ) -> None:
        if stage not in PAIR_MAP:
            raise ValueError(f"stage must be one of {tuple(PAIR_MAP)}, got {stage!r}")
        if image_size < 0:
            raise ValueError("image_size must be zero (native size) or a positive square size")
        self.root = Path(root)
        self.stage = stage
        self.image_size = image_size
        self.case_filter = case
        self.slice_filter = slice_id
        self.samples = self._scan()

    def _scan(self) -> list[ABUSSample]:
        view1, view2 = PAIR_MAP[self.stage]
        cases = [self.root / self.case_filter] if self.case_filter else sorted(self.root.glob("case*"))
        samples: list[ABUSSample] = []
        for case_dir in cases:
            if not case_dir.is_dir() or not (case_dir / "nipple_x.txt").is_file():
                continue
            left = _images(case_dir / view1)
            right = _images(case_dir / view2)
            if not left:
                left = _images(case_dir / f"{view1}.jpg")
            if not right:
                right = _images(case_dir / f"{view2}.jpg")
            left_map = {slice_stem_to_id(p.stem): p for p in left}
            right_map = {slice_stem_to_id(p.stem): p for p in right}
            common = sorted(set(left_map) & set(right_map))
            if common:
                pairs = [(sid, left_map[sid], right_map[sid], False) for sid in common]
            else:
                pairs = []
                if left and right:
                    warnings.warn(
                        f"{case_dir.name} stage{self.stage}: no matching slice stems; "
                        "falling back to zip(sorted(left), sorted(right))",
                        RuntimeWarning,
                    )
                    pairs = [
                        (slice_stem_to_id(lp.stem), lp, rp, True)
                        for lp, rp in zip(sorted(left), sorted(right))
                    ]
            for sid, path1, path2, fallback in pairs:
                if self.slice_filter is None or sid == self.slice_filter:
                    samples.append(ABUSSample(case_dir.name, self.stage, sid, path1, path2, fallback))
        return samples

    @staticmethod
    def _read_nipples(path: Path) -> list[float]:
        values = [float(v.strip()) for v in path.read_text(encoding="utf-8").strip().strip("[]").split(",") if v.strip()]
        if len(values) != 3:
            raise ValueError(f"Expected [x1,x2,x3] in {path}, got {values}")
        return values

    def _read_image(self, path: Path) -> tuple[torch.Tensor, tuple[int, int], tuple[float, float]]:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Unable to read image: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h0, w0 = rgb.shape[:2]
        if self.image_size > 0:
            rgb = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        h, w = rgb.shape[:2]
        tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.0
        return tensor, (h0, w0), (h / h0, w / w0)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image1, original_shape1, resize_factor1 = self._read_image(sample.path1)
        image2, original_shape2, resize_factor2 = self._read_image(sample.path2)
        if image1.shape[-2:] != image2.shape[-2:]:
            raise ValueError(f"PixelStitch requires equal pair sizes: {sample.path1} vs {sample.path2}")
        nipples = self._read_nipples(sample.path1.parents[1] / "nipple_x.txt")
        view1, view2 = PAIR_MAP[self.stage]
        idx1, idx2 = int(view1[-1]) - 1, int(view2[-1]) - 1
        x1 = nipples[idx1] * resize_factor1[1]
        x2 = nipples[idx2] * resize_factor2[1]
        return {
            "image1": image1,
            "image2": image2,
            "case": sample.case,
            "slice_id": sample.slice_id,
            "stage": sample.stage,
            "x1": torch.tensor(x1, dtype=torch.float32),
            "x2": torch.tensor(x2, dtype=torch.float32),
            "path1": str(sample.path1),
            "path2": str(sample.path2),
            "original_shape1": original_shape1,
            "original_shape2": original_shape2,
            "resize_factor1": resize_factor1,
            "resize_factor2": resize_factor2,
            "fallback_pairing": sample.fallback_pairing,
        }

    def __len__(self) -> int:
        return len(self.samples)
