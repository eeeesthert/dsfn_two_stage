import numpy as np
from HM_ransac import HM_ransac
from utils import cv_draw_matches
import cv2 as cv
from homography_geometry import inlier_coverage, projected_corners, validate_local_geometry


def comp_KR(im1, im2, X1, X2, display=True, *, estimator='opencv',
            reproj_threshold=3.0, max_iters=10000, min_inliers=6, diagnostics=None,
            geometric_validation='robust', min_inlier_coverage=0.01,
            max_local_scale=10.0, allow_reflection=False):
    """Estimate left-to-right H. Threshold is in original-image pixels."""
    if estimator not in {'opencv', 'legacy'}:
        raise ValueError(f'Unknown homography estimator: {estimator}')
    if reproj_threshold <= 0 or max_iters < 1 or min_inliers < 4:
        raise ValueError('Invalid homography estimation parameters')
    if geometric_validation not in {'robust', 'legacy'}:
        raise ValueError('Invalid geometric validation mode')
    if not np.isfinite(min_inlier_coverage) or not 0 <= min_inlier_coverage <= 1 or not np.isfinite(max_local_scale) or max_local_scale <= 1:
        raise ValueError('Invalid inlier coverage or local-scale limit')
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
    info['geometric_validation'] = geometric_validation
    projected = H @ X1
    with np.errstate(divide='ignore', invalid='ignore'):
        error = np.linalg.norm((projected[:2] / projected[2]).T - dst, axis=1)
    if estimator == 'opencv':
        # Re-check returned consensus against its final refined matrix.
        ok = ok & np.isfinite(error) & (error <= reproj_threshold)
    info['forward_only_inliers'] = int(np.count_nonzero(ok))
    backward_error = None
    if geometric_validation == 'robust':
        inverse = np.linalg.inv(H)
        reverse_projected = inverse @ X2
        with np.errstate(divide='ignore', invalid='ignore'):
            backward_error = np.linalg.norm((reverse_projected[:2] / reverse_projected[2]).T - src, axis=1)
        ok = ok & np.isfinite(error) & np.isfinite(backward_error)
        ok = ok & (error <= reproj_threshold) & (backward_error <= reproj_threshold)
    score = int(np.count_nonzero(ok))
    info.update(inliers=score, inlier_ratio=score / len(src),
                inlier_left_points=src[ok][:200].tolist(),
                inlier_right_points=dst[ok][:200].tolist())
    finite_errors = error[ok & np.isfinite(error)]
    if finite_errors.size:
        info.update(reprojection_median_px=float(np.median(finite_errors)),
                    reprojection_p95_px=float(np.percentile(finite_errors, 95)))
    if backward_error is not None and score:
        info.update(backward_reprojection_median_px=float(np.median(backward_error[ok])),
                    backward_reprojection_p95_px=float(np.percentile(backward_error[ok], 95)))
    if score < min_inliers:
        raise ValueError(f'Insufficient RANSAC inliers: {score} < {min_inliers}')
    for points in (src[ok], dst[ok]):
        if np.linalg.matrix_rank(np.column_stack([points, np.ones(score)])) < 3:
            raise ValueError('Degenerate collinear RANSAC inliers')
    if geometric_validation == 'robust':
        info['processing_stage'] = 'geometry_validation'
        if im1 is None or im2 is None:
            raise ValueError('Image dimensions required for robust geometry validation')
        coverage = [inlier_coverage(points, image.shape) for points, image in ((src[ok], im1), (dst[ok], im2))]
        info['inlier_hull_coverage'] = coverage
        if min(coverage) < min_inlier_coverage:
            raise ValueError(f'Insufficient spatial inlier coverage: {coverage}; required >= {min_inlier_coverage}')
        projected_corners(H, im2.shape, inverse=True)
        info.update(validate_local_geometry(H, im1.shape, max_local_scale=max_local_scale,
                                            allow_reflection=allow_reflection))
    print('Number of inliers:', score)
    if display:
        cv.imshow("Ransac's results", cv_draw_matches(im1, im2, X1, X2, ok))
        cv.waitKey(1)
    return H, X1[:, ok], X2[:, ok]
