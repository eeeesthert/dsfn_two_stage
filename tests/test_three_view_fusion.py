from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from abus_pairwise.three_view_fusion import fuse_case_from_pairwise, gaussian_pyramid_blend


def _write_rgb(path: Path, value: int) -> None:
    arr = np.full((32, 48, 3), value, dtype=np.uint8)
    cv2.imwrite(str(path), arr)


def _write_mask(path: Path, value: int) -> None:
    arr = np.full((32, 48), value, dtype=np.uint8)
    cv2.imwrite(str(path), arr)


def test_gaussian_pyramid_blend_shape():
    i1 = np.ones((32, 48, 3), dtype=np.float32)
    i2 = np.zeros((32, 48, 3), dtype=np.float32)
    w1 = np.ones((32, 48, 1), dtype=np.float32)
    w2 = np.ones((32, 48, 1), dtype=np.float32)
    out = gaussian_pyramid_blend([i1, i2], [w1, w2], levels=3)
    assert out.shape == (32, 48, 3)


def test_fuse_case_from_pairwise(tmp_path: Path):
    c12 = tmp_path / "12" / "case001" / "fusion"
    c23 = tmp_path / "23" / "case001" / "fusion"
    c12.mkdir(parents=True)
    c23.mkdir(parents=True)

    _write_rgb(c12 / "12_000_stitched.png", 100)
    _write_rgb(c23 / "23_000_stitched.png", 140)
    _write_mask(c12 / "12_000_mask_right_soft.png", 255)
    _write_mask(c23 / "23_000_mask_left_soft.png", 255)

    out_dir = tmp_path / "out"
    n = fuse_case_from_pairwise(tmp_path / "12" / "case001", tmp_path / "23" / "case001", out_dir)
    assert n == 1
    assert (out_dir / "threeview_000.png").exists()
    assert (out_dir / "metrics.csv").exists()


def test_fuse_case_no_hole_when_only_one_stage_valid(tmp_path: Path):
    c12 = tmp_path / "12" / "case001" / "fusion"
    c23 = tmp_path / "23" / "case001" / "fusion"
    c12.mkdir(parents=True)
    c23.mkdir(parents=True)

    _write_rgb(c12 / "12_000_stitched.png", 180)
    _write_rgb(c23 / "23_000_stitched.png", 0)
    _write_mask(c12 / "12_000_mask_left_soft.png", 255)
    _write_mask(c12 / "12_000_mask_right_soft.png", 0)
    _write_mask(c23 / "23_000_mask_left_soft.png", 0)
    _write_mask(c23 / "23_000_mask_right_soft.png", 0)

    out_dir = tmp_path / "out"
    n = fuse_case_from_pairwise(tmp_path / "12" / "case001", tmp_path / "23" / "case001", out_dir)
    assert n == 1
    im = cv2.imread(str(out_dir / "threeview_000.png"), cv2.IMREAD_GRAYSCALE)
    assert im is not None
    assert float(im.mean()) > 100.0
