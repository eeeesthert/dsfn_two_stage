"""Scale-invariant geometry checks shared by REW and its diagnostics."""
import numpy as np


def projected_corners(h, shape, *, inverse=False):
    h = np.asarray(h, dtype=np.float64)
    scale = np.max(np.abs(h))
    if h.shape != (3, 3) or not np.isfinite(h).all() or scale == 0:
        raise ValueError('Invalid homography matrix')
    h = h / scale
    if inverse:
        h = np.linalg.inv(h)
        h = h / np.max(np.abs(h))
    height, width = shape[:2]
    corners = np.array([[0, width-1, width-1, 0],
                        [0, 0, height-1, height-1], [1, 1, 1, 1]], dtype=float)
    projected = h @ corners
    w = projected[2]
    wscale = np.max(np.abs(w))
    relative_w = w / wscale if wscale > 0 else np.zeros(4)
    direction = 'right_to_left' if inverse else 'left_to_right'
    if not np.isfinite(projected).all() or np.any(np.abs(relative_w) <= 1e-10):
        raise ValueError(f'Unbounded projected canvas: {direction} corner near horizon; relative_w={relative_w.tolist()}')
    if np.min(relative_w) < 0 < np.max(relative_w):
        raise ValueError(f'Unbounded projected canvas: {direction} horizon crosses image; relative_w={relative_w.tolist()}')
    return projected / w
