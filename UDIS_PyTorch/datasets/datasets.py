"""Datasets for alignment, reconstruction, synthetic pretraining, and ABUS inference."""

import csv
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from UDIS_PyTorch.models.dlt import DifferentiableDLT
from UDIS_PyTorch.models.homography_warp import homography_warp
from UDIS_PyTorch.utils.utilities import load_mask, load_rgb, resize_image


class AlignmentDataset(Dataset):
	"""Read real image pairs from a CSV manifest."""

	def __init__(self, manifest, brightness=(0.7, 1.3), color=(0.7, 1.3), augment=True):
		with open(manifest, newline="", encoding="utf8") as stream:
			self.pairs = [
				tuple(row[:2]) for row in csv.reader(stream) if row and row[0] != "image1"
			]
		self.brightness = brightness
		self.color = color
		self.augment = augment

	def __len__(self):
		return len(self.pairs)

	def _jitter(self, image):
		if not self.augment:
			return image
		gain = image.new_tensor([random.uniform(*self.color) for _ in range(3)])[:, None, None]
		return (image * random.uniform(*self.brightness) * gain).clamp(-1, 1)

	def __getitem__(self, index):
		image1, image2 = map(load_rgb, self.pairs[index])
		return {
			"image1": image1,
			"image2": image2,
			"aug1": self._jitter(image1),
			"aug2": self._jitter(image2),
			"name": str(index),
		}


class ReconstructionDataset(Dataset):
	"""Read frozen Stage-1 warp and mask quadruples."""

	def __init__(self, root, max_image_size=1024):
		self.root = Path(root)
		self.files = sorted((self.root / "warp1").glob("*"))
		self.maximum = max_image_size

	def __len__(self):
		return len(self.files)

	def __getitem__(self, index):
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
			size = (int(height * scale) // 8 * 8, int(width * scale) // 8 * 8)
			for key in ("warp1", "warp2", "mask1", "mask2"):
				result[key] = resize_image(result[key][None], size)[0]
		return result


class SyntheticHomographyDataset(Dataset):
	"""Create synthetic no-parallax pairs, offsets remain debug-only metadata."""

	def __init__(self, root, patch_size=128, perturbation=16):
		self.files = [
			path
			for path in sorted(Path(root).glob("**/*"))
			if path.suffix.lower() in (".jpg", ".jpeg", ".png")
		]
		self.size = patch_size
		self.rho = perturbation
		self.dlt = DifferentiableDLT()

	def __len__(self):
		return len(self.files)

	def __getitem__(self, index):
		image = load_rgb(self.files[index])
		height, width = image.shape[-2:]
		if min(height, width) < self.size:
			image = resize_image(image[None], (max(height, self.size), max(width, self.size)))[0]
			height, width = image.shape[-2:]
		y = random.randint(0, height - self.size)
		x = random.randint(0, width - self.size)
		image1 = image[:, y : y + self.size, x : x + self.size]
		offsets = torch.empty(1, 8).uniform_(-self.rho, self.rho)
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


class ABUSPairDataset(Dataset):
	"""Enumerate ABUS case slices as the original UDIS pair inputs.

	Stage ``12`` pairs input1/input2 and stage ``23`` pairs input2/input3.
	The nipple coordinates are preserved as metadata, UDIS itself does not add a
	landmark branch, so the original method structure remains unchanged.
	"""

	def __init__(self, root, stage):
		if stage not in ("12", "23"):
			raise ValueError("stage must be '12' or '23'")
		self.stage = stage
		self.samples = []
		left_view, right_view = (("input1", "input2") if stage == "12" else ("input2", "input3"))
		for case_dir in sorted(Path(root).glob("case*")):
			if not case_dir.is_dir():
				continue
			left = self._images(case_dir, left_view)
			right = self._images(case_dir, right_view)
			for slice_name in sorted(left.keys() & right.keys()):
				self.samples.append(
					(case_dir, slice_name, left[slice_name], right[slice_name])
				)

	@staticmethod
	def _images(case_dir, view):
		directory = case_dir / view
		if directory.is_dir():
			return {
				path.stem: path
				for path in directory.iterdir()
				if path.suffix.lower() in (".jpg", ".jpeg", ".png")
			}
		single = case_dir / f"{view}.jpg"
		return {single.stem: single} if single.exists() else {}

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, index):
		case_dir, slice_name, left_path, right_path = self.samples[index]
		nipple_path = case_dir / "nipple_x.txt"
		nipple_x = []
		if nipple_path.exists():
			text = nipple_path.read_text(encoding="utf8").strip().strip("[]")
			nipple_x = [float(value) for value in text.replace(",", " ").split()]
		left = load_rgb(left_path)
		right = load_rgb(right_path)
		if left.shape != right.shape:
			raise ValueError(
				f"ABUS pair must have equal image sizes: {left_path} {tuple(left.shape)}, "
				f"{right_path} {tuple(right.shape)}"
			)
		return {
			"image1": left,
			"image2": right,
			"case": case_dir.name,
			"slice": slice_name,
			"stage": self.stage,
			"nipple_x": nipple_x,
		}
