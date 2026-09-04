"""Central image conversion and resize helpers."""

import random

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F


def resize_image(x: torch.Tensor, size):
	return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def load_rgb(path):
	a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
	return torch.from_numpy(a).permute(2, 0, 1) / 127.5 - 1


def load_mask(path):
	a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
	return torch.from_numpy(a).permute(2, 0, 1) / 255.0


def save_image(x, path, mask=False):
	x = x.detach().cpu().clamp(0, 1) if mask else ((x.detach().cpu().clamp(-1, 1) + 1) * 0.5)
	Image.fromarray((x.permute(1, 2, 0).numpy() * 255).round().astype("uint8")).save(path)


def save_checkpoint(path, model, optimizer, step, config, scheduler=None):
	"""Save all state required to resume either training stage."""
	torch.save(
		{
			"model": model.state_dict(),
			"optimizer": optimizer.state_dict(),
			"scheduler": scheduler.state_dict() if scheduler else None,
			"step": step,
			"config": config,
		},
		path,
	)


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu"):
	"""Restore model and optional training state."""
	state = torch.load(path, map_location=map_location)
	model.load_state_dict(state["model"])
	if optimizer and state.get("optimizer"):
		optimizer.load_state_dict(state["optimizer"])
	if scheduler and state.get("scheduler"):
		scheduler.load_state_dict(state["scheduler"])
	return state


def set_seed(seed=2020):
	"""Seed all random number generators used by the project."""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def log_images(writer, images, step):
	"""Log normalized image tensors to TensorBoard."""
	for name, value in images.items():
		writer.add_images(name, value.detach().cpu().clamp(-1, 1) * 0.5 + 0.5, step)
