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


def inlier_coverage(points, shape):
    """Convex hull coverage, measured in the original image coordinates."""
    import cv2 as cv
    height, width = shape[:2]
    area = cv.contourArea(cv.convexHull(np.asarray(points, dtype=np.float32)))
    return float(area / max(1, (height-1)*(width-1)))


def validate_local_geometry(h, shape, *, max_local_scale=10.0, allow_reflection=False):
    """Sample local stretch at a 3x3 grid; not a guarantee between samples."""
    projected_corners(h, shape)
    height, width = shape[:2]
    h = np.asarray(h, dtype=float) / np.max(np.abs(h))
    scales, determinants = [], []
    for y in np.linspace(0, height-1, 3):
        for x in np.linspace(0, width-1, 3):
            p = h @ np.array([x, y, 1.])
            uv = p[:2] / p[2]
            jacobian = (h[:2, :2] - uv[:, None]*h[2, :2]) / p[2]
            scales.extend(np.linalg.svd(jacobian, compute_uv=False).tolist())
            determinants.append(float(np.linalg.det(jacobian)))
    stats = dict(local_scale_min=float(min(scales)), local_scale_max=float(max(scales)),
                 reflected=bool(any(d < 0 for d in determinants)))
    if not allow_reflection and stats['reflected']:
        raise ValueError(f'Homography reflection rejected: {stats}')
    if stats['local_scale_min'] < 1/max_local_scale or stats['local_scale_max'] > max_local_scale:
        raise ValueError(f'Extreme local homography compression/stretch: {stats}')
    return stats
