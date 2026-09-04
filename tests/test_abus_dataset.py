from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "contract" / "pixelstitch"))
from pixelstitch_abus.dataset import PixelStitchABUSDataset  # noqa: E402


def _write_case(root: Path) -> None:
    case = root / "case001"
    image = np.full((10, 20, 3), 127, np.uint8)
    for view in ("input1", "input2", "input3"):
        (case / view).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(case / view / "slice_0053.jpg"), image)
    (case / "nipple_x.txt").write_text("[2,4,8]", encoding="utf-8")


def test_stage_pair_direction_and_real_slice_id(tmp_path: Path) -> None:
    _write_case(tmp_path)
    stage12 = PixelStitchABUSDataset(tmp_path, "12")[0]
    stage23 = PixelStitchABUSDataset(tmp_path, "23")[0]
    assert stage12["slice_id"] == stage23["slice_id"] == "0053"
    assert "/input1/" in stage12["path1"] and "/input2/" in stage12["path2"]
    assert "/input2/" in stage23["path1"] and "/input3/" in stage23["path2"]
    assert tuple(stage12["image1"].shape) == (3, 10, 20)


def test_resize_scales_nipple_x(tmp_path: Path) -> None:
    _write_case(tmp_path)
    item = PixelStitchABUSDataset(tmp_path, "12", image_size=40)[0]
    assert float(item["x1"]) == 4.0
    assert float(item["x2"]) == 8.0
