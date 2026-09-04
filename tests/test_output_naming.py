from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "contract" / "pixelstitch"))
from pixelstitch_abus.save_utils import output_paths  # noqa: E402


def test_benchmark_output_names_preserve_slice_id(tmp_path: Path) -> None:
    paths = output_paths(tmp_path, "12", "case001", "0053")
    assert paths["left"].name == "12_0053_left.png"
    assert paths["right"].name == "12_0053_right.png"
    assert paths["stitched"].name == "12_0053_stitched.png"
    assert paths["mask_left_soft"].name == "12_0053_mask_left_soft.png"
