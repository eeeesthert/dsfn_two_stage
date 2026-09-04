import os

import pytest
import torch
from PIL import Image

from UDIS_PyTorch.datasets import ABUSPairDataset
from UDIS_PyTorch.models.cost_volume import CostVolume
from UDIS_PyTorch.models.dlt import DifferentiableDLT, transform_points
from UDIS_PyTorch.models.feature_pyramid import FeaturePyramid
from UDIS_PyTorch.models.homography_net import HomographyNet
from UDIS_PyTorch.models.homography_warp import homography_warp
from UDIS_PyTorch.models.reconstruction_net import ReconstructionNet
from UDIS_PyTorch.models.stitching_transform import StitchingDomainTransformer


@pytest.mark.parametrize("search_range,channels", [(16, 1089), (8, 289), (4, 81)])
def test_cost_volume_shapes_and_backward(search_range, channels):
	feature1 = torch.randn(1, 2, 3, 4, requires_grad=True)
	feature2 = torch.randn_like(feature1, requires_grad=True)
	output = CostVolume(search_range)(feature1, feature2)
	assert output.shape == (1, channels, 3, 4)
	output.sum().backward()
	assert feature1.grad is not None and feature2.grad is not None


def test_dlt_identity():
	assert torch.allclose(
		DifferentiableDLT()(torch.zeros(1, 8), 128),
		torch.eye(3)[None],
		atol=1e-4,
	)


def test_dlt_translation():
	offsets = torch.tensor([[5.0, -3.0] * 4])
	homography = DifferentiableDLT()(offsets, 128)
	points = torch.tensor([[[0.0, 0.0], [128.0, 128.0]]])
	assert torch.allclose(
		transform_points(homography, points),
		points + torch.tensor([5.0, -3.0]),
		atol=1e-4,
	)


def test_dlt_gradient():
	offsets = torch.zeros(1, 8, requires_grad=True)
	DifferentiableDLT()(offsets, 128).sum().backward()
	assert offsets.grad is not None and torch.isfinite(offsets.grad).all()


def test_feature_pyramid_shapes():
	output = FeaturePyramid()(torch.randn(1, 1, 128, 128))
	assert [tensor.shape for tensor in output] == [
		(1, 64, 128, 128),
		(1, 64, 64, 64),
		(1, 128, 32, 32),
		(1, 128, 16, 16),
	]


@pytest.mark.skipif(
	os.environ.get("UDIS_FULL_MODEL_TEST") != "1",
	reason="exact FC layers require about 1 GB RAM, set UDIS_FULL_MODEL_TEST=1",
)
def test_homography_net_outputs():
	model = HomographyNet().eval()
	output = model(torch.randn(1, 3, 128, 128), torch.randn(1, 3, 128, 128))
	assert all(output[key].shape == (1, 8) for key in output)
	assert torch.allclose(
		output["delta_final"],
		output["delta1"] + output["delta2"] + output["delta3"],
	)


def test_homography_warp_forward_translation_direction():
	image = torch.zeros(1, 1, 7, 7)
	image[0, 0, 2, 2] = 1
	homography = torch.tensor([[[1.0, 0, 2], [0, 1.0, 1], [0, 0, 1.0]]])
	output = homography_warp(image, homography)
	assert output[0, 0, 3, 4] > 0.99


def test_reconstruction_shapes_range_backward():
	model = ReconstructionNet(lr_size=32, num_res_blocks=1)
	image = torch.randn(1, 3, 16, 24, requires_grad=True)
	output = model(image, image)
	assert output["lr"].shape == (1, 3, 32, 32)
	assert output["hr"].shape == (1, 3, 16, 24)
	assert output["lr"].abs().max() <= 1
	assert output["hr"].abs().max() <= 1
	(output["lr"].mean() + output["hr"].mean()).backward()
	assert image.grad is not None


@pytest.mark.parametrize("translation", [(0, 0), (12, 4), (-12, -4)])
def test_stitching_union(translation):
	tx, ty = translation
	image = torch.ones(1, 3, 24, 32)
	homography = torch.tensor([[[1.0, 0, tx], [0, 1.0, ty], [0, 0, 1.0]]])
	output = StitchingDomainTransformer()(image, image, homography)
	height, width = output["warp1"].shape[-2:]
	assert height % 8 == 0 and width % 8 == 0 and width >= 32 + abs(tx)
	assert 0 <= output["mask1"].min() <= output["mask1"].max() <= 1


def test_abus_dataset_matches_slice_names_and_loads_nipples(tmp_path):
	case = tmp_path / "case001"
	for view in ("input1", "input2", "input3"):
		(case / view).mkdir(parents=True)
		Image.new("RGB", (16, 12)).save(case / view / "slice_0001.jpg")
	(case / "nipple_x.txt").write_text("[1, 2, 3]", encoding="utf8")
	dataset = ABUSPairDataset(tmp_path, "23")
	sample = dataset[0]
	assert len(dataset) == 1
	assert sample["case"] == "case001"
	assert sample["slice"] == "slice_0001"
	assert sample["nipple_x"] == [1.0, 2.0, 3.0]
	assert sample["image1"].shape == sample["image2"].shape == (3, 12, 16)
