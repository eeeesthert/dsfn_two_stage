"""ABUS pair datasets used by the DunHuangStitch comparison baseline."""

from pathlib import Path

from torch.utils.data import Dataset

from .image_pair_dataset import load_image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _images(path):
    if path.is_file():
        return [path]
    if not path.is_dir():
        return []
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def _view_images(case_dir, view):
    # Support both case/input1.jpg and case/input1/slice_XXXX.jpg.
    for suffix in IMAGE_EXTENSIONS:
        found = _images(case_dir / f"{view}{suffix}")
        if found:
            return found
    return _images(case_dir / view)


def _slice_id(path):
    return path.stem.removeprefix("slice_")


def scan_abus_pairs(root, stages=("12", "23")):
    """Return matched ABUS slice pairs as ``(stage, case, id, left, right)``."""
    pairs = []
    root = Path(root)
    for case in sorted(p for p in root.iterdir() if p.is_dir()):
        for stage in stages:
            if stage not in {"12", "23"}:
                raise ValueError(f"stage must be '12' or '23', got {stage!r}")
            left, right = (_view_images(case, f"input{i}") for i in stage)
            left_by_id = {_slice_id(p): p for p in left}
            right_by_id = {_slice_id(p): p for p in right}
            common = sorted(left_by_id.keys() & right_by_id.keys())
            if common:
                matched = ((sid, left_by_id[sid], right_by_id[sid]) for sid in common)
            else:
                matched = ((_slice_id(a), a, b) for a, b in zip(left, right))
            pairs.extend((stage, case.name, sid, a, b) for sid, a, b in matched)
    return pairs


class ABUSPairDataset(Dataset):
    """Read input1-input2/input2-input3 ABUS pairs without requiring labels."""

    def __init__(self, root, stages=("12", "23"), size=None):
        self.items = scan_abus_pairs(root, tuple(stages))
        self.size = size
        if not self.items:
            raise RuntimeError(f"No matched ABUS image pairs found under {root}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        stage, case, slice_id, reference, target = self.items[index]
        return {
            "reference": load_image(reference, self.size),
            "target": load_image(target, self.size),
            "stage": stage,
            "case": case,
            "slice_id": slice_id,
            "reference_path": str(reference),
            "target_path": str(target),
        }


class ABUSAlignedPairDataset(Dataset):
    """Read aligned pairs produced by ``generate_aligned_abus.py``."""

    def __init__(self, root):
        self.items = sorted(Path(root).glob("*/**/slice_*"))
        self.items = [p for p in self.items if p.is_dir() and (p / "reference.png").is_file()]
        if not self.items:
            raise RuntimeError(f"No generated ABUS aligned pairs found under {root}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        directory = self.items[index]
        return {
            "I_wr": load_image(directory / "reference.png"),
            "I_wt": load_image(directory / "target.png"),
            "M_wr": load_image(directory / "mask_reference.png")[:1],
            "M_wt": load_image(directory / "mask_target.png")[:1],
            "name": str(directory.relative_to(directory.parents[2])),
        }
