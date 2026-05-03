from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from abus_pairwise.datasets import ABUSPairDataset


def _write_img(path: Path, value: int) -> None:
    arr = np.full((20, 30, 3), value, dtype=np.uint8)
    cv2.imwrite(str(path), arr)


def test_dataset_reads_case_pairs(tmp_path: Path):
    case = tmp_path / "case001"
    case.mkdir(parents=True)
    _write_img(case / "input1.jpg", 64)
    _write_img(case / "input2.jpg", 128)
    _write_img(case / "input3.jpg", 192)
    (case / "nipple_x.txt").write_text("[11,22,33]", encoding="utf-8")

    ds12 = ABUSPairDataset(tmp_path, stage="12", image_size=32)
    item12 = ds12[0]
    assert item12["left"].shape == (3, 32, 32)
    assert float(item12["left_x"][0]) == 11
    assert float(item12["right_x"][0]) == 22

    ds23 = ABUSPairDataset(tmp_path, stage="23", image_size=32)
    item23 = ds23[0]
    assert float(item23["left_x"][0]) == 22
    assert float(item23["right_x"][0]) == 33


def test_dataset_reads_slice_directory_layout(tmp_path: Path):
    case = tmp_path / "case001"
    (case / "input1").mkdir(parents=True)
    (case / "input2").mkdir(parents=True)
    (case / "input3").mkdir(parents=True)
    _write_img(case / "input1" / "slice_0001.jpg", 64)
    _write_img(case / "input2" / "slice_0001.jpg", 128)
    _write_img(case / "input3" / "slice_0001.jpg", 192)
    (case / "nipple_x.txt").write_text("[11,22,33]", encoding="utf-8")

    ds12 = ABUSPairDataset(tmp_path, stage="12", image_size=32)
    assert len(ds12) == 1
    item12 = ds12[0]
    assert float(item12["left_x"][0]) == 11
    assert float(item12["right_x"][0]) == 22


def test_dataset_keeps_original_size_when_image_size_none(tmp_path: Path):
    case = tmp_path / "case001"
    case.mkdir(parents=True)
    _write_img(case / "input1.jpg", 64)
    _write_img(case / "input2.jpg", 128)
    _write_img(case / "input3.jpg", 192)
    (case / "nipple_x.txt").write_text("[11,22,33]", encoding="utf-8")

    ds12 = ABUSPairDataset(tmp_path, stage="12", image_size=None)
    item12 = ds12[0]
    assert item12["left"].shape == (3, 20, 30)
