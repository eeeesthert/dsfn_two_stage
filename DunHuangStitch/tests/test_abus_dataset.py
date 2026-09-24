from pathlib import Path

import numpy as np
from PIL import Image

from datasets.abus_pair_dataset import ABUSAlignedPairDataset, ABUSPairDataset


def _write(path: Path, value=64):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((24, 32, 3), value, dtype=np.uint8)).save(path)


def test_abus_dataset_matches_slices_and_both_stages(tmp_path):
    case = tmp_path / "case001"
    for view in ("input1", "input2", "input3"):
        _write(case / view / "slice_0002.png")
        _write(case / view / "slice_0001.png")
    dataset = ABUSPairDataset(tmp_path, stages=("12", "23"), size=None)
    assert len(dataset) == 4
    assert [(dataset[i]["stage"], dataset[i]["slice_id"]) for i in range(4)] == [
        ("12", "0001"), ("12", "0002"), ("23", "0001"), ("23", "0002")
    ]
    assert dataset[0]["reference"].shape == (3, 24, 32)


def test_abus_dataset_supports_single_image_layout(tmp_path):
    case = tmp_path / "patient_a"
    _write(case / "input1.jpg")
    _write(case / "input2.jpg")
    dataset = ABUSPairDataset(tmp_path, stages=("12",), size=(16, 20))
    assert len(dataset) == 1
    assert dataset[0]["reference"].shape == (3, 16, 20)


def test_aligned_abus_dataset_reads_generated_tree(tmp_path):
    directory = tmp_path / "12" / "case001" / "slice_0001"
    for name in ("reference.png", "target.png", "mask_reference.png", "mask_target.png"):
        _write(directory / name)
    sample = ABUSAlignedPairDataset(tmp_path)[0]
    assert sample["I_wr"].shape == (3, 24, 32)
    assert sample["M_wr"].shape == (1, 24, 32)
