"""Datasets used by both UDIS stages and by the ABUS adapter.

Keeping the related readers in one module makes the accepted input layouts easy
to discover.  Images are always returned as RGB ``float32`` tensors in
``[-1, 1]``.  ABUS slices are paired by slice name and are never resized.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from UDIS_PyTorch.models.alignment import DifferentiableDLT
from UDIS_PyTorch.models.alignment import homography_warp
from UDIS_PyTorch.utils.image import load_mask, load_rgb, resize_image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImagePair:
	"""One pair, including stable identifiers used in ABUS output paths."""

	image1: Path
	image2: Path
	case: str = ""
	slice_id: str = ""
	stage: str = ""


def _images(path: Path) -> list[Path]:
	if path.is_file():
		return [path]
	if not path.exists():
		return []
	return sorted(
		item for item in path.iterdir() if item.suffix.lower() in IMAGE_SUFFIXES
	)


def _slice_id(path: Path) -> str:
	return path.stem.removeprefix("slice_")


def scan_abus_pairs(root: str | Path, stage: str) -> list[ImagePair]:
	"""Read ``case*/input{1,2,3}`` ABUS folders without resizing or cropping.

	``stage='12'`` pairs input1 with input2 and ``stage='23'`` pairs input2
	with input3.  Same-named slices are preferred.  If names do not intersect,
	the two sorted sequences are zipped, matching the repository's existing ABUS
	loaders.
	"""
	if stage not in {"12", "23"}:
		raise ValueError("stage must be '12' or '23'")

	root = Path(root)
	left_name, right_name = (
		("input1", "input2") if stage == "12" else ("input2", "input3")
	)
	pairs: list[ImagePair] = []
	for case_dir in sorted(path for path in root.iterdir() if path.is_dir()):
		left = _images(case_dir / left_name) or _images(case_dir / f"{left_name}.jpg")
		right = _images(case_dir / right_name) or _images(
			case_dir / f"{right_name}.jpg"
		)
		left_by_id = {_slice_id(path): path for path in left}
		right_by_id = {_slice_id(path): path for path in right}
		common_ids = sorted(left_by_id.keys() & right_by_id.keys())

		if common_ids:
			path_pairs = [
				(left_by_id[key], right_by_id[key], key) for key in common_ids
			]
		else:
			path_pairs = [(a, b, _slice_id(a)) for a, b in zip(left, right)]

		for image1, image2, slice_id in path_pairs:
			pairs.append(ImagePair(image1, image2, case_dir.name, slice_id, stage))
	return pairs


class AlignmentDataset(Dataset):
	"""Real image pairs with independent paper-compatible photometric jitter."""

	def __init__(
		self,
		manifest: str | Path | None = None,
		brightness: tuple[float, float] = (0.7, 1.3),
		color: tuple[float, float] = (0.7, 1.3),
		augment: bool = True,
		abus_root: str | Path | None = None,
		stage: str = "12",
	) -> None:
		if (manifest is None) == (abus_root is None):
			raise ValueError("provide exactly one of manifest or abus_root")
		if abus_root is not None:
			self.pairs = scan_abus_pairs(abus_root, stage)
		else:
			self.pairs = self._read_manifest(Path(manifest))
		if not self.pairs:
			raise ValueError("no image pairs were found")
		self.brightness = brightness
		self.color = color
		self.augment = augment

	@staticmethod
	def _read_manifest(path: Path) -> list[ImagePair]:
		with path.open(newline="", encoding="utf8") as stream:
			rows = [row for row in csv.reader(stream) if row and row[0] != "image1"]
		return [
			ImagePair(Path(row[0]), Path(row[1]), slice_id=str(index))
			for index, row in enumerate(rows)
		]

	def __len__(self) -> int:
		return len(self.pairs)

	def _jitter(self, image: torch.Tensor) -> torch.Tensor:
		if not self.augment:
			return image
		color_gain = image.new_tensor([random.uniform(*self.color) for _ in range(3)])[
			:, None, None
		]
		brightness_gain = random.uniform(*self.brightness)
		return (image * brightness_gain * color_gain).clamp(-1, 1)

	def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
		pair = self.pairs[index]
		image1 = load_rgb(pair.image1)
		image2 = load_rgb(pair.image2)
		if image1.shape != image2.shape:
			raise ValueError(
				"paired images must have the same original size because ABUS input is not resized: "
				f"{pair.image1}={tuple(image1.shape[-2:])}, {pair.image2}={tuple(image2.shape[-2:])}"
			)
		return {
			"image1": image1,
			"image2": image2,
			"aug1": self._jitter(image1),
			"aug2": self._jitter(image2),
			"name": pair.slice_id,
			"case": pair.case,
			"stage": pair.stage,
		}


class ReconstructionDataset(Dataset):
	"""Read the frozen ``warp1/warp2/mask1/mask2`` Stage-1 result."""

	def __init__(self, root: str | Path, max_image_size: int = 1024) -> None:
		self.root = Path(root)
		self.files = sorted((self.root / "warp1").glob("*"))
		self.maximum = max_image_size

	def __len__(self) -> int:
		return len(self.files)

	def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
		name = self.files[index].name
		result = {
			"warp1": load_rgb(self.root / "warp1" / name),
			"warp2": load_rgb(self.root / "warp2" / name),
			"mask1": load_mask(self.root / "mask1" / name),
			"mask2": load_mask(self.root / "mask2" / name),
			"name": Path(name).stem,
		}
		height, width = result["warp1"].shape[-2:]
		if self.maximum and max(height, width) > self.maximum:
			scale = self.maximum / max(height, width)
			size = (
				max(8, int(height * scale) // 8 * 8),
				max(8, int(width * scale) // 8 * 8),
			)
			for key in ("warp1", "warp2", "mask1", "mask2"):
				result[key] = resize_image(result[key][None], size)[0]
		return result


class SyntheticHomographyDataset(Dataset):
	"""Generate no-parallax pretraining pairs with debug-only GT offsets."""

	def __init__(
		self, root: str | Path, patch_size: int = 128, perturbation: int = 16
	) -> None:
		self.files = sorted(
			path
			for path in Path(root).glob("**/*")
			if path.suffix.lower() in IMAGE_SUFFIXES
		)
		self.size = patch_size
		self.perturbation = perturbation
		self.dlt = DifferentiableDLT()

	def __len__(self) -> int:
		return len(self.files)

	def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
		image = load_rgb(self.files[index])
		height, width = image.shape[-2:]
		if min(height, width) < self.size:
			image = resize_image(
				image[None], (max(height, self.size), max(width, self.size))
			)[0]
			height, width = image.shape[-2:]
		top = random.randint(0, height - self.size)
		left = random.randint(0, width - self.size)
		image1 = image[:, top : top + self.size, left : left + self.size]
		offsets = torch.empty(1, 8).uniform_(-self.perturbation, self.perturbation)
		homography = self.dlt(offsets, self.size)
		image2 = homography_warp(image1[None], torch.linalg.inv(homography))[0]
		return {
			"image1": image1,
			"image2": image2,
			"aug1": image1,
			"aug2": image2,
			"gt_offsets": offsets[0],
			"name": self.files[index].stem,
		}
