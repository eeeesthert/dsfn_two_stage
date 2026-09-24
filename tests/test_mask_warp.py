from pathlib import Path
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parents[1] / "contract" / "pixelstitch"))
from pixelstitch_abus.masks import warp_support_masks  # noqa: E402


def test_masks_are_warped_geometric_support_not_intensity() -> None:
    support = torch.ones(1, 1, 8, 8)
    grid = F.affine_grid(torch.eye(2, 3).repeat(2, 1, 1), (2, 1, 8, 8), align_corners=False)
    masks = warp_support_masks(support, support, grid)
    assert masks["overlap"].all()
    assert masks["union"].all()
