from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "contract" / "pixelstitch"))
from pixelstitch_abus.utils import load_checkpoint_validated  # noqa: E402


class TinyRAFT(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fnet = torch.nn.Linear(2, 2)
        self.cnet = torch.nn.Linear(2, 2)
        self.update_block = torch.nn.Linear(2, 2)


def test_checkpoint_accepts_module_prefix(tmp_path: Path) -> None:
    model = TinyRAFT()
    path = tmp_path / "ok.pt"
    torch.save({f"module.{k}": v for k, v in model.state_dict().items()}, path)
    info = load_checkpoint_validated(model, path, "cpu")
    assert not info["missing"] and not info["unexpected"]


def test_checkpoint_rejects_missing_critical_weights(tmp_path: Path) -> None:
    path = tmp_path / "bad.pt"
    torch.save({}, path)
    with pytest.raises(RuntimeError, match="incompatible"):
        load_checkpoint_validated(TinyRAFT(), path, "cpu")
