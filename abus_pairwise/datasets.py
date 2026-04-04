from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class PairSample:
    case_dir: Path
    left_name: str
    right_name: str


class ABUSPairDataset(Dataset):
    """
    Expects layout:
      dataset/case001/input1.jpg
      dataset/case001/input2.jpg
      dataset/case001/input3.jpg
      dataset/case001/nipple_x.txt   # [x1,x2,x3]
    """

    def __init__(self, root: str | Path, stage: str = "12", image_size: int = 512):
        self.root = Path(root)
        self.stage = stage
        self.image_size = image_size
        self.samples = self._scan_cases()

    def _scan_cases(self) -> List[PairSample]:
        pairs = {"12": ("input1.jpg", "input2.jpg"), "23": ("input2.jpg", "input3.jpg")}
        left_name, right_name = pairs[self.stage]
        samples: List[PairSample] = []
        for case_dir in sorted(self.root.glob("case*")):
            if (case_dir / left_name).exists() and (case_dir / right_name).exists() and (case_dir / "nipple_x.txt").exists():
                samples.append(PairSample(case_dir, left_name, right_name))
        return samples

    @staticmethod
    def _read_nipple_x(path: Path) -> List[float]:
        raw = path.read_text(encoding="utf-8").strip().strip("[]")
        vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
        if len(vals) != 3:
            raise ValueError(f"nipple_x.txt must have 3 values: {path}")
        return vals

    def _load_img(self, p: Path) -> torch.Tensor:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        return tensor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        sample = self.samples[idx]
        nipple_x = self._read_nipple_x(sample.case_dir / "nipple_x.txt")
        left = self._load_img(sample.case_dir / sample.left_name)
        right = self._load_img(sample.case_dir / sample.right_name)

        # input1/2/3 -> index 0/1/2
        left_idx = int(sample.left_name.replace("input", "").replace(".jpg", "")) - 1
        right_idx = int(sample.right_name.replace("input", "").replace(".jpg", "")) - 1

        return {
            "left": left,
            "right": right,
            "left_x": torch.tensor([nipple_x[left_idx]], dtype=torch.float32),
            "right_x": torch.tensor([nipple_x[right_idx]], dtype=torch.float32),
            "case": sample.case_dir.name,
        }
