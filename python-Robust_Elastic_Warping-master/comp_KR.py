import numpy as np
from HM_ransac import HM_ransac
from utils import cv_draw_matches
import cv2 as cv


def comp_KR(im1, im2, X1, X2, display=True, *, estimator='opencv',
            reproj_threshold=3.0, max_iters=10000, min_inliers=6, diagnostics=None):
    """Estimate left-to-right H. Threshold is in original-image pixels."""
    if estimator not in {'opencv', 'legacy'}:
        raise ValueError(f'Unknown homography estimator: {estimator}')
    if reproj_threshold <= 0 or max_iters < 1 or min_inliers < 4:
        raise ValueError('Invalid homography estimation parameters')
    info = diagnostics if diagnostics is not None else {}
    info.update(homography_estimator=estimator, matches=int(X1.shape[1]))
    if X1.shape[1] < 4:
        raise ValueError('Fewer than four correspondences')
    src = np.asarray((X1[:2] / X1[2]).T, dtype=np.float64)
    dst = np.asarray((X2[:2] / X2[2]).T, dtype=np.float64)
    if not np.isfinite(src).all() or not np.isfinite(dst).all():
        raise ValueError('Non-finite correspondences')
    if estimator == 'legacy':
        # Preserve the original implementation for a separate baseline setting.
        H, ok, _ = HM_ransac(X1, X2, 500, 0.1)
    else:
        H, mask = cv.findHomography(src, dst, method=cv.RANSAC,
                                   ransacReprojThreshold=reproj_threshold,
                                   maxIters=max_iters, confidence=0.999)
        if H is None or mask is None:
            raise ValueError('RANSAC could not estimate a homography')
        ok = mask.ravel().astype(bool)
    if not np.isfinite(H).all() or np.max(np.abs(H)) == 0:
        raise ValueError('Non-finite or zero homography')
    H = H / np.max(np.abs(H))
    info['H_left_to_right'] = H.tolist()
    projected = H @ X1
    with np.errstate(divide='ignore', invalid='ignore'):
        error = np.linalg.norm((projected[:2] / projected[2]).T - dst, axis=1)
    if estimator == 'opencv':
        # Re-check returned consensus against its final refined matrix.
        ok = ok & np.isfinite(error) & (error <= reproj_threshold)
    score = int(np.count_nonzero(ok))
    info.update(inliers=score, inlier_ratio=score / len(src),
                inlier_left_points=src[ok][:200].tolist(),
                inlier_right_points=dst[ok][:200].tolist())
    finite_errors = error[ok & np.isfinite(error)]
    if finite_errors.size:
        info.update(reprojection_median_px=float(np.median(finite_errors)),
                    reprojection_p95_px=float(np.percentile(finite_errors, 95)))
    if score < min_inliers:
        raise ValueError(f'Insufficient RANSAC inliers: {score} < {min_inliers}')
    for points in (src[ok], dst[ok]):
        if np.linalg.matrix_rank(np.column_stack([points, np.ones(score)])) < 3:
            raise ValueError('Degenerate collinear RANSAC inliers')
    print('Number of inliers:', score)
    if display:
        cv.imshow("Ransac's results", cv_draw_matches(im1, im2, X1, X2, ok))
        cv.waitKey(1)
    return H, X1[:, ok], X2[:, ok]
