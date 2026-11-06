"""Linear weighted smoothing on the union canvas."""

import cv2
import numpy as np


def linear_blend(
	reference, target, reference_mask, target_mask, method="linear_distance"
):
	if reference.shape != target.shape or reference_mask.shape != reference.shape[:2]:
		raise ValueError("Shape mismatch")
	m1 = reference_mask.astype(bool)
	m2 = target_mask.astype(bool)
	union = m1 | m2
	if not union.any():
		raise ValueError("Empty union")
	if method == "linear_distance":
		d1 = cv2.distanceTransform(m1.astype(np.uint8), cv2.DIST_L2, 5)
		d2 = cv2.distanceTransform(m2.astype(np.uint8), cv2.DIST_L2, 5)
		w1 = d1 / (d1 + d2 + 1e-8)
	elif method == "intensity_weighted":
		a = reference.mean(2) if reference.ndim == 3 else reference
		b = target.mean(2) if target.ndim == 3 else target
		w1 = (a + 1e-6) / (a + b + 2e-6)
	else:
		raise ValueError(f"Unknown blending method {method}")
	w1 = np.where(m1 & ~m2, 1, np.where(m2 & ~m1, 0, w1))
	weights = w1[..., None] if reference.ndim == 3 else w1
	out = weights * reference + (1 - weights) * target
	out[~union] = 0
	return out.astype(np.float32), w1.astype(np.float32)


def crop_to_mask(image, mask):
	ys, xs = np.where(mask)
	if not len(xs):
		raise ValueError("Cannot crop empty mask")
	return image[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


# Paper metrics and regional engineering extensions
from skimage.metrics import structural_similarity


def _values(a, b, mask=None):
	a = np.asarray(a, float)
	b = np.asarray(b, float)
	if a.shape != b.shape:
		raise ValueError("Metric inputs differ in shape")
	if mask is not None:
		m = np.asarray(mask, bool)
		m = np.repeat(m[..., None], a.shape[2], 2) if a.ndim == 3 else m
		return a[m], b[m]
	return a.ravel(), b.ravel()


def compute_mse(a, b, mask=None):
	x, y = _values(a, b, mask)
	return float(np.mean((x - y) ** 2)) if len(x) else float("nan")


def compute_psnr(a, b, data_range=None, mask=None):
	mse = compute_mse(a, b, mask)
	if mse == 0:
		return float("inf")
	if data_range is None:
		data_range = 255.0 if np.issubdtype(np.asarray(a).dtype, np.integer) else 1.0
	return float(10 * np.log10(data_range**2 / mse))


def compute_ssim(a, b, data_range=None, mask=None):
	if mask is not None:
		x, y = _values(a, b, mask)
		mse = np.mean((x - y) ** 2)
		var = np.var(x) + np.var(y)
		return float(1 - mse / (var + 1e-12))
	if data_range is None:
		data_range = 255.0 if np.issubdtype(np.asarray(a).dtype, np.integer) else 1.0
	return float(
		structural_similarity(
			a,
			b,
			data_range=data_range,
			channel_axis=-1 if np.asarray(a).ndim == 3 else None,
		)
	)


def compute_ncc(a, b, mask=None):
	x, y = _values(a, b, mask)
	x = x - x.mean()
	y = y - y.mean()
	return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))


def evaluate_pair(warped_ref, warped_target, fused, region_mask):
	return {
		"mse": compute_mse(warped_ref, warped_target, region_mask),
		"psnr": compute_psnr(warped_ref, warped_target, mask=region_mask),
		"ssim": compute_ssim(warped_ref, warped_target, mask=region_mask),
		"ncc": compute_ncc(warped_ref, warped_target, region_mask),
	}


# Feature, registration, and debug visualizations
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from .features import FeatureSet, Match


def as_u8(x):
	x = np.asarray(x)
	return x if x.dtype == np.uint8 else np.clip(x * 255, 0, 255).astype(np.uint8)


def draw_slic(image, labels):
	return (mark_boundaries(image, labels, color=(1, 0, 0)) * 255).astype(np.uint8)


def draw_features(image, fs):
	out = cv2.cvtColor(as_u8(image), cv2.COLOR_GRAY2BGR)
	for f in fs.features:
		cv2.circle(
			out,
			(round(f.x), round(f.y)),
			2,
			(0, 255, 0) if f.source_type == "global" else (0, 165, 255),
			1,
		)
	return out


def draw_matches(ref, target, rf, tf, matches):
	a = cv2.cvtColor(as_u8(ref), cv2.COLOR_GRAY2BGR)
	b = cv2.cvtColor(as_u8(target), cv2.COLOR_GRAY2BGR)
	h = max(a.shape[0], b.shape[0])
	out = np.zeros((h, a.shape[1] + b.shape[1], 3), np.uint8)
	out[: a.shape[0], : a.shape[1]] = a
	out[: b.shape[0], a.shape[1] :] = b
	for m in matches:
		p = (
			round(rf.features[m.reference_index].x),
			round(rf.features[m.reference_index].y),
		)
		q = (
			round(tf.features[m.target_index].x) + a.shape[1],
			round(tf.features[m.target_index].y),
		)
		cv2.line(out, p, q, (0, 255, 0), 1)
	return out


def overlay(a, b, ma=None, mb=None):
	return np.clip(0.5 * a + 0.5 * b, 0, 1)


def checkerboard(a, b, tile=32):
	yy, xx = np.indices(a.shape[:2])
	m = ((xx // tile + yy // tile) % 2).astype(bool)
	return np.where(m[..., None], a, b) if a.ndim == 3 else np.where(m, a, b)


def montage(images, path):
	fig, axes = plt.subplots(8, 2, figsize=(14, 32))
	for ax, (title, img) in zip(axes.flat, images):
		ax.imshow(
			(
				cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
				if img.ndim == 3 and img.shape[2] == 3
				else img
			),
			cmap="gray",
		)
		ax.set_title(title)
		ax.axis("off")
	for ax in axes.flat[len(images) :]:
		ax.axis("off")
	fig.tight_layout()
	fig.savefig(path, dpi=120)
	plt.close(fig)
