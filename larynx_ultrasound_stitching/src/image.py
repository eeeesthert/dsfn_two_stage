"""Image and configuration I/O without modifying source files."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import cv2
import numpy as np
import yaml


@dataclass(frozen=True)
class LoadedImage:
	original: np.ndarray
	path: Path
	dtype: np.dtype
	minimum: float
	maximum: float


def load_image(path: str | Path) -> LoadedImage:
	"""Load PNG/JPEG/BMP/TIFF at unchanged depth and retain range metadata."""
	path = Path(path)
	if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
		raise ValueError(f"Unsupported image extension: {path.suffix}")
	image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
	if image is None:
		raise FileNotFoundError(f"Cannot read image: {path}")
	return LoadedImage(
		image.copy(), path, image.dtype, float(image.min()), float(image.max())
	)


def normalize_image(image: np.ndarray) -> np.ndarray:
	"""Return a float32 [0,1] copy using dtype range, preserving input."""
	x = image.astype(np.float32)
	if np.issubdtype(image.dtype, np.integer):
		info = np.iinfo(image.dtype)
		x = (x - info.min) / float(info.max - info.min)
	else:
		lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
		x = (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)
	return np.clip(x, 0, 1).astype(np.float32)


def to_grayscale(image: np.ndarray) -> np.ndarray:
	"""Convert BGR/BGRA input to grayscale. Pass through 2-D input."""
	if image.ndim == 2:
		return image.copy()
	if image.ndim != 3 or image.shape[2] not in (3, 4):
		raise ValueError("Expected grayscale, BGR, or BGRA image")
	code = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
	return cv2.cvtColor(image, code)


def load_config(path: str | Path) -> dict[str, Any]:
	with Path(path).open(encoding="utf8") as f:
		return yaml.safe_load(f)


def save_image(path: str | Path, image: np.ndarray) -> None:
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	x = image
	if np.issubdtype(x.dtype, np.floating):
		x = np.clip(x * 255, 0, 255).astype(np.uint8)
	if not cv2.imwrite(str(path), x):
		raise IOError(f"Could not write {path}")


# Ultrasound preprocessing


def enhance_ultrasound(gray: np.ndarray, config: dict) -> np.ndarray:
	"""Normalize, optionally apply CLAHE and mild Gaussian smoothing."""
	out = normalize_image(gray)
	if config.get("clahe", False):
		u8 = np.round(out * 255).astype(np.uint8)
		grid = int(config.get("clahe_grid_size", 8))
		out = (
			cv2.createCLAHE(float(config.get("clahe_clip_limit", 2.0)), (grid, grid))
			.apply(u8)
			.astype(np.float32)
			/ 255
		)
	sigma = float(config.get("gaussian_sigma", 0))
	if sigma > 0:
		out = cv2.GaussianBlur(out, (0, 0), sigma)
	return np.clip(out, 0, 1).astype(np.float32)
