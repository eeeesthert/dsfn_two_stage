"""Paper-equation and OpenCV real-valued Gabor feature banks."""

import cv2
import numpy as np


def build_gabor_filter_bank(config: dict) -> np.ndarray:
	scales = int(config["scales"])
	orientations = int(config["orientations"])
	sigma = float(config["sigma"])
	n = int(config.get("kernel_size", 31))
	n += 1 - n % 2
	yy, xx = np.mgrid[-n // 2 + 1 : n // 2 + 1, -n // 2 + 1 : n // 2 + 1]
	kernels = []
	for v in range(scales):
		kv = 2 ** (-(v + 2) / 2) * np.pi
		for u in range(orientations):
			phi = u * np.pi / orientations
			if config.get("backend", "paper") == "opencv":
				kernel = cv2.getGaborKernel(
					(n, n), sigma, phi, 2 * np.pi / kv, 1, 0, ktype=cv2.CV_32F
				)
			else:
				phase = kv * (xx * np.cos(phi) + yy * np.sin(phi))
				envelope = (kv**2 / sigma**2) * np.exp(
					-(kv**2) * (xx**2 + yy**2) / (2 * sigma**2)
				)
				kernel = envelope * (np.cos(phase) - np.exp(-(sigma**2) / 2))
			kernel = np.asarray(kernel, np.float32)
			kernel -= kernel.mean()
			kernels.append(kernel)
	return np.stack(kernels)


def apply_gabor_bank(image: np.ndarray, kernels: np.ndarray) -> np.ndarray:
	return np.stack(
		[
			cv2.filter2D(image, cv2.CV_32F, k, borderType=cv2.BORDER_REFLECT)
			for k in kernels
		]
	)


def compute_gabor_energy(
	responses: np.ndarray, window: int, downsample_factor: int = 1
) -> np.ndarray:
	if window < 1 or window % 2 == 0:
		raise ValueError("energy_window must be positive odd")
	energy = np.stack([cv2.blur(r * r, (window, window)) for r in responses]).astype(
		np.float32
	)
	# Bicubic reduced representation is exposed without sacrificing full maps used by refinement.
	if downsample_factor > 1:
		h, w = energy.shape[1:]
		_ = np.stack(
			[
				cv2.resize(
					e,
					(max(1, w // downsample_factor), max(1, h // downsample_factor)),
					interpolation=cv2.INTER_CUBIC,
				)
				for e in energy
			]
		)
	return energy


def sample_feature_map(feature_map: np.ndarray, points: np.ndarray) -> np.ndarray:
	maps = feature_map.astype(np.float32)
	h, w = maps.shape[1:]
	p = np.asarray(points, np.float32)
	out = []
	for x, y in p:
		if x < 0 or y < 0 or x > w - 1 or y > h - 1:
			out.append(np.zeros(len(maps), np.float32))
			continue
		vals = np.array(
			[cv2.getRectSubPix(m, (1, 1), (float(x), float(y)))[0, 0] for m in maps]
		)
		vals /= np.linalg.norm(vals) + 1e-12
		out.append(vals)
	return np.asarray(out, np.float32)


def gabor_feature_distance(ref_map, target_map, ref_points, target_points):
	r = sample_feature_map(ref_map, ref_points) - sample_feature_map(
		target_map, target_points
	)
	return r, float(np.sum(r * r))


# Replaceable saliency backends and Eq. (10) set variance
from scipy.spatial import cKDTree


def _norm(x):
	x = np.asarray(x, np.float32)
	lo, hi = float(x.min()), float(x.max())
	return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def spectral_residual(image):
	f = np.fft.fft2(image)
	log = np.log(np.abs(f) + 1e-8)
	avg = cv2.blur(log.astype(np.float32), (3, 3))
	residual = np.exp(log - avg + 1j * np.angle(f))
	sal = np.abs(np.fft.ifft2(residual)) ** 2
	return _norm(cv2.GaussianBlur(sal.astype(np.float32), (9, 9), 2.5))


def compute_saliency(image: np.ndarray, method="spectral_residual") -> np.ndarray:
	if method == "spectral_residual":
		return spectral_residual(image)
	if method == "gradient":
		return _norm(
			cv2.magnitude(
				cv2.Sobel(image, cv2.CV_32F, 1, 0), cv2.Sobel(image, cv2.CV_32F, 0, 1)
			)
		)
	if method == "intensity_variance":
		return _norm(cv2.blur(image * image, (9, 9)) - cv2.blur(image, (9, 9)) ** 2)
	if method == "fine_grained" and hasattr(cv2, "saliency"):
		return _norm(
			cv2.saliency.StaticSaliencyFineGrained_create().computeSaliency(image)[1]
		)
	raise ValueError(f"Unavailable saliency method: {method}")


def extract_salient_points(saliency, percentile, max_points, min_distance):
	threshold = np.percentile(saliency, percentile)
	size = 2 * int(min_distance) + 1
	maxima = (saliency == cv2.dilate(saliency, np.ones((size, size), np.uint8))) & (
		saliency > threshold
	)
	ys, xs = np.where(maxima)
	order = np.argsort(saliency[ys, xs])[::-1][:max_points]
	return np.c_[xs[order], ys[order]].astype(float)


def saliency_variance(points: np.ndarray) -> float:
	p = np.asarray(points, float)
	return 0.0 if len(p) == 0 else float(np.sqrt(np.sum((p - p.mean(0)) ** 2) / len(p)))


def local_saliency_variance(points: np.ndarray, k=8) -> np.ndarray:
	p = np.asarray(points, float)
	if not len(p):
		return np.empty(0)
	tree = cKDTree(p)
	_, ids = tree.query(p, k=min(k, len(p)))
	return np.array([saliency_variance(p[np.atleast_1d(i)]) for i in ids])


def match_salient_points(reference_points, target_points, H, max_distance):
	from .registration import transform_points

	projected = transform_points(target_points, H)
	tree = cKDTree(reference_points)
	d, idx = tree.query(projected, distance_upper_bound=max_distance)
	ok = np.isfinite(d) & (idx < len(reference_points))
	return reference_points[idx[ok]], target_points[ok]


# Local homography approximation to the under-specified paper refinement
from dataclasses import dataclass
from scipy.optimize import least_squares
from scipy.sparse.linalg import lsqr
from .registration import transform_points


@dataclass(frozen=True)
class RefinementResult:
	H: np.ndarray
	accepted: bool
	Eg_before: float
	Eg_after: float
	Es_before: float
	Es_after: float
	message: str


def _pack(H):
	return np.array(
		[H[0, 0], H[0, 1], H[0, 2], H[1, 0], H[1, 1], H[1, 2], H[2, 0], H[2, 1]]
	)


def _unpack(x):
	return np.array([[x[0], x[1], x[2]], [x[3], x[4], x[5]], [x[6], x[7], 1.0]])


def transform_quality(H, shape, cfg):
	h, w = shape[:2]
	corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)
	q = transform_points(corners, H)
	area = abs(cv2.contourArea(q.astype(np.float32))) / (w * h)
	det = np.linalg.det(H[:2, :2])
	ok = (
		np.isfinite(H).all()
		and abs(np.linalg.det(H)) > 1e-10
		and det > 0
		and cfg["min_area_ratio"] <= area <= cfg["max_area_ratio"]
		and max(abs(H[2, 0]), abs(H[2, 1])) <= cfg.get("max_perspective", np.inf)
	)
	return ok, {
		"determinant": float(np.linalg.det(H)),
		"area_ratio": float(area),
		"scale_x": float(np.linalg.norm(H[:2, 0])),
		"scale_y": float(np.linalg.norm(H[:2, 1])),
		"rotation_deg": float(np.degrees(np.arctan2(H[1, 0], H[0, 0]))),
		"shear": float(np.dot(H[:2, 0], H[:2, 1])),
		"perspective": float(np.linalg.norm(H[2, :2])),
	}


def refine_homography(
	H0,
	ref_energy,
	target_energy,
	ref_points,
	target_points,
	sal_ref,
	sal_target,
	cfg,
	check_cfg,
	target_shape,
	use_gabor=True,
	use_saliency=True,
):
	base = _pack(H0)
	ref_tex = sample_feature_map(ref_energy, ref_points)
	sg = saliency_variance(sal_ref)
	st = saliency_variance(sal_target)

	def components(x):
		H = _unpack(x)
		projected = transform_points(target_points, H)
		rg = (
			(ref_tex - sample_feature_map(target_energy, target_points)).ravel()
			if use_gabor
			else np.empty(0)
		)
		# Spatial correspondence modulated by paper Eq.(10) global set spread.
		rs = (
			((ref_points - projected) / (max(sg, st, 1.0))).ravel()
			if use_saliency
			else np.empty(0)
		)
		if cfg.get("normalize_residuals", True):
			if len(rg):
				rg = rg / (np.sqrt(np.mean(rg * rg)) + 1e-6)
			if len(rs):
				rs = rs / (np.sqrt(np.mean(rs * rs)) + 1e-6)
		return rg, rs

	def fun(x):
		rg, rs = components(x)
		return np.r_[np.sqrt(cfg["lambda_g"]) * rg, np.sqrt(cfg["lambda_s"]) * rs]

	rg0, rs0 = components(base)
	eg0 = float(rg0 @ rg0)
	es0 = float(rs0 @ rs0)
	if cfg.get("backend", "nonlinear") == "sparse_linear":
		eps = 1e-6
		r0 = fun(base)
		J = np.column_stack(
			[(fun(base + np.eye(8)[j] * eps) - r0) / eps for j in range(8)]
		)
		delta = lsqr(J, -r0, iter_lim=int(cfg["max_nfev"]))[0]
		candidate = _unpack(base + delta)
		msg = "sparse linearized LSQR"
	else:
		result = least_squares(
			fun,
			base,
			method="trf",
			max_nfev=int(cfg["max_nfev"]),
			ftol=float(cfg["ftol"]),
			xtol=float(cfg["xtol"]),
			gtol=float(cfg["gtol"]),
		)
		candidate = _unpack(result.x)
		msg = result.message
	ok, _ = transform_quality(candidate, target_shape, check_cfg)
	H = candidate if ok else H0
	rg1, rs1 = components(_pack(H))
	return RefinementResult(
		H, ok, eg0, float(rg1 @ rg1), es0, float(rs1 @ rs1), str(msg)
	)
