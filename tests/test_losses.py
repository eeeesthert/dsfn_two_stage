import pytest

torch = pytest.importorskip("torch")

from abus_pairwise.losses import nipple_prior_loss, nipple_x_heatmap, x_heatmap_similarity_loss


def test_nipple_x_heatmap_shape():
    x = torch.tensor([[10.0], [20.0]])
    hm = nipple_x_heatmap(x, width=32, height=16, sigma=3.0)
    assert hm.shape == (2, 1, 16, 32)
    assert torch.all(hm >= 0)


def test_nipple_prior_loss_zero_when_aligned():
    global_shift = torch.tensor([[4.0], [2.0]])
    left_x = torch.tensor([[10.0], [20.0]])
    right_x = torch.tensor([[14.0], [22.0]])
    loss = nipple_prior_loss(global_shift, left_x, right_x)
    assert torch.isclose(loss, torch.tensor(0.0))


def test_x_heatmap_similarity_loss_prefers_similar_blend():
    b, c, h, w = 1, 3, 16, 16
    left = torch.ones((b, c, h, w))
    right = torch.ones((b, c, h, w))
    stitched_good = torch.ones((b, c, h, w))
    stitched_bad = torch.zeros((b, c, h, w))

    left_x = torch.tensor([[8.0]])
    right_x = torch.tensor([[8.0]])

    good = x_heatmap_similarity_loss(stitched_good, left, right, left_x, right_x)
    bad = x_heatmap_similarity_loss(stitched_bad, left, right, left_x, right_x)

    assert good < bad
