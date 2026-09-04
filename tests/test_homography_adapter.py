from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "contract" / "pixelstitch"))
from pixelstitch_abus.homography import homography_matrix_to_corner_motion, image_corners  # noqa: E402


def test_identity_and_translations() -> None:
    assert np.allclose(homography_matrix_to_corner_motion(np.eye(3), 100, 200), 0)
    for dx, dy in ((50, 0), (50, -20)):
        H = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], np.float32)
        assert np.allclose(homography_matrix_to_corner_motion(H, 100, 200), [dx, dy])


def test_rotation_matches_opencv() -> None:
    H = cv2.getRotationMatrix2D((100, 50), 15, 1.0)
    H = np.vstack([H, [0, 0, 1]])
    src = image_corners(100, 200)
    expected = cv2.perspectiveTransform(src[None], H)[0] - src
    assert np.allclose(homography_matrix_to_corner_motion(H, 100, 200), expected)
