from pathlib import Path


def test_entrypoints_exist():
    assert Path("train_pairwise.py").exists()
    assert Path("infer_pairwise.py").exists()


def test_package_layout_exists():
    assert Path("abus_pairwise/models/warp.py").exists()
    assert Path("abus_pairwise/models/fusion.py").exists()
