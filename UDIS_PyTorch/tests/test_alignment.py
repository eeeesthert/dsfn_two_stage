"""Focused tests for the complete Stage-1 alignment module."""

import pytest
import torch

from UDIS_PyTorch.models.alignment import (
	CostVolume,
	DifferentiableDLT,
	FeaturePyramid,
	StitchingDomainTransformer,
	homography_warp,
	transform_points,
)


@pytest.mark.parametrize("search_range,channels", [(16, 1089), (8, 289), (4, 81)])
def test_cost_volume_shape_and_gradient(search_range, channels):
	feature1 = torch.randn(1, 2, 3, 4, requires_grad=True)
	feature2 = torch.randn_like(feature1, requires_grad=True)
	volume = CostVolume(search_range)(feature1, feature2)
	assert volume.shape == (1, channels, 3, 4)
	volume.sum().backward()
	assert feature1.grad is not None
	assert feature2.grad is not None


def test_dlt_identity():
	homography = DifferentiableDLT()(torch.zeros(1, 8), 128)
	assert torch.allclose(homography, torch.eye(3)[None], atol=1e-4)


def test_dlt_translation():
	offsets = torch.tensor([[5.0, -3.0] * 4])
	homography = DifferentiableDLT()(offsets, 128)
	points = torch.tensor([[[0.0, 0.0], [128.0, 128.0]]])
	expected = points + torch.tensor([5.0, -3.0])
	assert torch.allclose(transform_points(homography, points), expected, atol=1e-4)


def test_dlt_gradient():
	offsets = torch.zeros(1, 8, requires_grad=True)
	DifferentiableDLT()(offsets, 128).sum().backward()
	assert offsets.grad is not None
	assert torch.isfinite(offsets.grad).all()


def test_feature_pyramid_shapes():
	features = FeaturePyramid()(torch.randn(1, 1, 128, 128))
	assert [feature.shape for feature in features] == [
		(1, 64, 128, 128),
		(1, 64, 64, 64),
		(1, 128, 32, 32),
		(1, 128, 16, 16),
	]


def test_homography_warp_translation_direction():
	image = torch.zeros(1, 1, 7, 7)
	image[0, 0, 2, 2] = 1
	homography = torch.tensor([[[1.0, 0, 2], [0, 1.0, 1], [0, 0, 1.0]]])
	warped = homography_warp(image, homography)
	assert warped[0, 0, 3, 4] > 0.99


@pytest.mark.parametrize("tx,ty", [(0, 0), (12, 4), (-12, -4)])
def test_stitching_canvas_union(tx, ty):
	image = torch.ones(1, 3, 24, 32)
	homography = torch.tensor([[[1.0, 0, tx], [0, 1.0, ty], [0, 0, 1.0]]])
	output = StitchingDomainTransformer()(image, image, homography)
	height, width = output["warp1"].shape[-2:]
	assert height % 8 == 0
	assert width % 8 == 0
	assert width >= 32 + abs(tx)
	assert 0 <= output["mask1"].min() <= output["mask1"].max() <= 1
