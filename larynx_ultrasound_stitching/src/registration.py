"""Normalized DLT and deterministic four-correspondence RANSAC."""

from dataclasses import dataclass
import math
import numpy as np


def transform_points(points: np.ndarray, H: np.ndarray) -> np.ndarray:
	p = np.c_[np.asarray(points, float), np.ones(len(points))]
	q = (H @ p.T).T
	return q[:, :2] / q[:, 2, None]


def _normalize(p):
	c = p.mean(0)
	d = np.mean(np.linalg.norm(p - c, axis=1))
	s = np.sqrt(2) / d if d > 1e-12 else 1
	T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])
	return transform_points(p, T), T


def estimate_homography_dlt(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
	"""Estimate source-to-destination H with Hartley-normalized DLT."""
	source = np.asarray(source, float)
	destination = np.asarray(destination, float)
	if (
		source.shape != destination.shape
		or source.ndim != 2
		or source.shape[1] != 2
		or len(source) < 4
	):
		raise ValueError("Need >=4 paired 2-D points")
	s, Ts = _normalize(source)
	d, Td = _normalize(destination)
	A = []
	for (x, y), (u, v) in zip(s, d):
		A.extend(
			[
				[-x, -y, -1, 0, 0, 0, u * x, u * y, u],
				[0, 0, 0, -x, -y, -1, v * x, v * y, v],
			]
		)
	_, _, vt = np.linalg.svd(np.asarray(A))
	Hn = vt[-1].reshape(3, 3)
	H = np.linalg.inv(Td) @ Hn @ Ts
	if abs(H[2, 2]) < 1e-12:
		raise np.linalg.LinAlgError("Degenerate homography")
	return H / H[2, 2]


def reprojection_errors(source, destination, H):
	return np.linalg.norm(transform_points(source, H) - destination, axis=1)


@dataclass(frozen=True)
class RansacResult:
	H: np.ndarray
	inliers: np.ndarray
	errors: np.ndarray
	iterations: int


def ransac_homography(
	source: np.ndarray, destination: np.ndarray, config: dict
) -> RansacResult:
	source = np.asarray(source, float)
	destination = np.asarray(destination, float)
	if len(source) < 4:
		raise ValueError("RANSAC needs at least four correspondences")
	rng = np.random.default_rng(int(config.get("seed", 42)))
	threshold = float(config["reprojection_threshold"])
	limit = int(config["max_iterations"])
	confidence = float(config["confidence"])
	best = None
	best_mean = np.inf
	i = 0
	while i < limit:
		ids = rng.choice(len(source), 4, replace=False)
		i += 1
		try:
			H = estimate_homography_dlt(source[ids], destination[ids])
			errors = reprojection_errors(source, destination, H)
		except (ValueError, np.linalg.LinAlgError):
			continue
		mask = np.isfinite(errors) & (errors < threshold)
		mean = float(errors[mask].mean()) if mask.any() else np.inf
		if (
			best is None
			or mask.sum() > best.sum()
			or (mask.sum() == best.sum() and mean < best_mean)
		):
			best = mask
			best_mean = mean
			w = mask.mean()
			if w > 0:
				limit = min(
					limit,
					max(
						i,
						int(
							math.ceil(
								math.log(1 - confidence)
								/ math.log(max(1e-12, 1 - w**4))
							)
						),
					),
				)
	if best is None or best.sum() < 4:
		raise RuntimeError("RANSAC could not find four inliers")
	H = estimate_homography_dlt(source[best], destination[best])
	return RansacResult(H, best, reprojection_errors(source, destination, H), i)


# Non-cropping panorama construction and warping
import cv2


@dataclass(frozen=True)
class WarpResult:
	reference: np.ndarray
	target: np.ndarray
	reference_mask: np.ndarray
	target_mask: np.ndarray
	translation: np.ndarray
	target_h_canvas: np.ndarray


def image_corners(shape):
	h, w = shape[:2]
	return np.array([[0, 0], [w, 0], [w, h], [0, h]], float)


def union_canvas(reference_shape, target_shape, H, max_pixels=100_000_000):
	allp = np.vstack(
		[
			image_corners(reference_shape),
			transform_points(image_corners(target_shape), H),
		]
	)
	lo = np.floor(allp.min(0))
	hi = np.ceil(allp.max(0))
	width, height = int(hi[0] - lo[0]), int(hi[1] - lo[1])
	if width <= 0 or height <= 0 or width * height > max_pixels:
		raise ValueError(f"Unsafe canvas {width}x{height}")
	T = np.array([[1, 0, -lo[0]], [0, 1, -lo[1]], [0, 0, 1]], float)
	return (width, height), T


def warp_to_union(reference, target, H, max_pixels=100_000_000) -> WarpResult:
	size, T = union_canvas(reference.shape, target.shape, H, max_pixels)
	rh, rw = reference.shape[:2]
	th, tw = target.shape[:2]
	ref = cv2.warpPerspective(reference, T, size)
	tgt = cv2.warpPerspective(target, T @ H, size)
	rm = (
		cv2.warpPerspective(
			np.ones((rh, rw), np.uint8), T, size, flags=cv2.INTER_NEAREST
		)
		> 0
	)
	tm = (
		cv2.warpPerspective(
			np.ones((th, tw), np.uint8), T @ H, size, flags=cv2.INTER_NEAREST
		)
		> 0
	)
	return WarpResult(ref, tgt, rm, tm, T, T @ H)
