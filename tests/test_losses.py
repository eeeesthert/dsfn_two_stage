import pytest

torch = pytest.importorskip("torch")

from abus_pairwise.losses import (
    fusion_smoothness_loss,
    grid_angle_loss,
    grid_edge_length_loss,
    nipple_heatmap_alignment_loss,
    nipple_x_heatmap,
    overlap_l1_warp_loss,
    seam_cost_loss,
    seam_overlap_boundary_loss,
)


def test_nipple_x_heatmap_shape():
    x = torch.tensor([[10.0], [20.0]])
    hm = nipple_x_heatmap(x, width=32, height=16, sigma=3.0)
    assert hm.shape == (2, 1, 16, 32)


def test_overlap_l1_warp_loss_zero_for_identical():
    a = torch.ones((1, 3, 8, 8))
    b = torch.ones((1, 3, 8, 8))
    m = torch.ones((1, 1, 8, 8))
    assert torch.isclose(overlap_l1_warp_loss(a, b, m), torch.tensor(0.0))


def test_grid_regularization_nonnegative():
    disp = torch.zeros((1, 2, 5, 5))
    assert grid_edge_length_loss(disp) >= 0
    assert grid_angle_loss(disp) >= 0


def test_seam_losses_nonnegative():
    l = torch.rand((1, 3, 8, 8))
    r = torch.rand((1, 3, 8, 8))
    seam = torch.sigmoid(torch.randn((1, 1, 8, 8)))
    ov = torch.ones((1, 1, 8, 8))
    assert seam_overlap_boundary_loss(seam, ov) >= 0
    assert seam_cost_loss(l, r, seam) >= 0
    assert fusion_smoothness_loss(l) >= 0


def test_nipple_heatmap_alignment_nonnegative():
    l = torch.rand((1, 3, 8, 8))
    r = torch.rand((1, 3, 8, 8))
    lx = torch.tensor([[3.0]])
    rx = torch.tensor([[4.0]])
    assert nipple_heatmap_alignment_loss(l, r, lx, rx) >= 0
