from pathlib import Path

import numpy as np
from PIL import Image

from UDIS_PyTorch.datasets import AlignmentDataset, scan_abus_pairs
from UDIS_PyTorch.evaluate_abus import evaluate
from UDIS_PyTorch.generate_aligned_dataset import output_name


def write_image(path: Path, value: int = 64) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	Image.fromarray(np.full((8, 12, 3), value, dtype=np.uint8)).save(path)


def test_abus_scanner_and_dataset_keep_original_size(tmp_path):
	for view in ("input1", "input2", "input3"):
		write_image(tmp_path / "case001" / view / "slice_0007.png")

	pairs = scan_abus_pairs(tmp_path, "12")
	assert len(pairs) == 1
	assert pairs[0].case == "case001"
	assert pairs[0].slice_id == "0007"

	sample = AlignmentDataset(abus_root=tmp_path, stages=("23",), augment=False)[0]
	assert sample["image1"].shape == (3, 8, 12)
	assert sample["image2"].shape == (3, 8, 12)


def test_abus_evaluator_reads_inference_layout(tmp_path):
	directory = tmp_path / "12" / "case001" / "fusion"
	write_image(directory / "12_0007_warp1.png", 80)
	write_image(directory / "12_0007_warp2.png", 80)
	write_image(directory / "12_0007_mask_left_soft.png", 255)
	write_image(directory / "12_0007_mask_right_soft.png", 255)

	rows = evaluate(tmp_path)
	assert len(rows) == 1
	assert rows[0]["mse"] == 0
	assert rows[0]["psnr"] == 99


def test_aligned_dataset_names_are_unique_across_stages(tmp_path):
	for view in ("input1", "input2", "input3"):
		write_image(tmp_path / "case001" / view / "slice_0007.png")
	pair12 = scan_abus_pairs(tmp_path, "12")[0]
	pair23 = scan_abus_pairs(tmp_path, "23")[0]
	assert output_name(pair12) == "12_case001_0007.png"
	assert output_name(pair23) == "23_case001_0007.png"
