import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from abus_pairwise.pipeline import LossWeights, TwoStageStitcher, compute_total_loss


def test_two_stage_forward_and_loss_smoke():
    model = TwoStageStitcher(pretrained_backbone=False)
    model.eval()

    left = torch.rand(1, 3, 64, 64)
    right = torch.rand(1, 3, 64, 64)
    left_x = torch.tensor([[30.0]])
    right_x = torch.tensor([[34.0]])

    with torch.no_grad():
        out = model(left, right)

    assert out["left_warp"].shape == (1, 3, 64, 64)
    assert out["stitched"].shape == (1, 3, 64, 64)
    assert out["mask_right"].shape == (1, 1, 64, 64)

    losses = compute_total_loss(out, left_x, right_x, LossWeights())
    assert "total" in losses
    assert torch.isfinite(losses["total"])
