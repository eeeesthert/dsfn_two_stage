"""SIFT matching with an explicit legacy option and auditable filtering."""
import math
import cv2 as cv
import numpy as np


def feature_mask(gray, *, mode='content', black_threshold=0, exclude_top=0,
                 exclude_border=0, external=None):
    """Limit feature centres only; never crop or modify the source image."""
    height, width = gray.shape
    if mode not in {'content', 'none'} or not 0 <= black_threshold <= 255:
        raise ValueError('Invalid feature mask mode or black threshold')
    if exclude_top < 0 or exclude_top >= height or exclude_border < 0 or 2*exclude_border >= min(height, width):
        raise ValueError('Feature exclusions leave no usable image area')
    mask = (gray > black_threshold) if mode == 'content' else np.ones_like(gray, dtype=bool)
    if external is not None:
        if external.shape != gray.shape:
            raise ValueError(f'Feature mask dimensions {external.shape} do not match image {gray.shape}')
        mask &= external > 0
    mask[:exclude_top] = False
    if exclude_border:
        b = exclude_border
        mask[:b] = mask[-b:] = False
        mask[:, :b] = mask[:, -b:] = False
    return mask.astype(np.uint8) * 255


def ratio_matches(neighbors, ratio):
    return [pair[0] for pair in neighbors if len(pair) == 2
            and pair[0].distance < ratio * pair[1].distance]


def mutual_matches(forward, reverse):
    reverse_pairs = {(m.trainIdx, m.queryIdx) for m in reverse}
    return [m for m in forward if (m.queryIdx, m.trainIdx) in reverse_pairs]


def unique_correspondences(source, target, distances, radius=1.0):
    """Keep the best descriptor match once per spatial position on BOTH sides."""
    if not np.isfinite(radius) or radius < 0:
        raise ValueError('Coordinate deduplication radius must be finite and non-negative')
    grids = [{}, {}]
    kept = []
    def key(point):
        return tuple(point) if radius == 0 else tuple(np.floor(point / radius).astype(int))
    def nearby(grid, point):
        k = key(point)
        if radius == 0:
            return k in grid
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for old in grid.get((k[0]+dx, k[1]+dy), []):
                    if np.linalg.norm(old-point) <= radius:
                        return True
        return False
    for index in np.argsort(distances, kind='stable'):
        points = (source[index], target[index])
        if any(nearby(grid, point) for grid, point in zip(grids, points)):
            continue
        kept.append(int(index))
        for grid, point in zip(grids, points):
            grid.setdefault(key(point), []).append(point)
    return kept


def load_data(im1, im2, *, matching_mode='robust', ratio=0.7, dedup_radius=1.0,
              mask_mode='content', black_threshold=0, exclude_top=0,
              exclude_border=0, mask1=None, mask2=None, diagnostics=None):
    if matching_mode not in {'robust', 'legacy'} or not math.isfinite(ratio) or not 0 < ratio < 1:
        raise ValueError('Invalid matching mode or SIFT ratio')
    if not np.isfinite(dedup_radius) or dedup_radius < 0:
        raise ValueError('Coordinate deduplication radius must be finite and non-negative')
    info = diagnostics if diagnostics is not None else {}
    info.update(matching_mode=matching_mode, sift_ratio=ratio, dedup_radius_px=dedup_radius)
    gray = [cv.cvtColor(im, cv.COLOR_BGR2GRAY) if im.ndim == 3 else im for im in (im1, im2)]
    masks = [feature_mask(g, mode=mask_mode, black_threshold=black_threshold,
                          exclude_top=exclude_top, exclude_border=exclude_border, external=m)
             for g, m in zip(gray, (mask1, mask2))]
    info['feature_mask_pixels'] = [int(np.count_nonzero(m)) for m in masks]
    sift = cv.SIFT_create()
    (kp1, ds1), (kp2, ds2) = [sift.detectAndCompute(g, m) for g, m in zip(gray, masks)]
    info.update(keypoints_left=len(kp1), keypoints_right=len(kp2))
    if ds1 is None or ds2 is None or len(ds1) < 2 or len(ds2) < 2:
        raise ValueError('Insufficient SIFT descriptors in feature regions')
    if matching_mode == 'legacy':
        matcher = cv.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    else:
        matcher = cv.BFMatcher(cv.NORM_L2)
    forward = ratio_matches(matcher.knnMatch(ds1, ds2, k=2), ratio)
    info['forward_ratio_matches'] = len(forward)
    good = forward
    if matching_mode == 'robust':
        reverse = ratio_matches(matcher.knnMatch(ds2, ds1, k=2), ratio)
        good = mutual_matches(forward, reverse)
        info.update(reverse_ratio_matches=len(reverse), mutual_matches=len(good))
    source = np.asarray([kp1[m.queryIdx].pt for m in good], dtype=float).reshape(-1, 2)
    target = np.asarray([kp2[m.trainIdx].pt for m in good], dtype=float).reshape(-1, 2)
    if matching_mode == 'robust':
        selected = unique_correspondences(source, target, [m.distance for m in good], dedup_radius)
        source, target = source[selected], target[selected]
    info.update(matches=len(source), match_left_points=source[:200].tolist(),
                match_right_points=target[:200].tolist())
    if len(source) < 4:
        raise ValueError(f'Fewer than four SIFT matches after {matching_mode} filtering: {len(source)}; forward_ratio={len(forward)}')
    return (np.vstack([source.T, np.ones(len(source))]),
            np.vstack([target.T, np.ones(len(target))]))
