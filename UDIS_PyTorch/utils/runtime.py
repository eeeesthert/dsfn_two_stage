"""Training runtime helpers kept together to avoid tiny utility modules."""

import random

import numpy as np
import torch


def set_seed(seed: int = 2020) -> None:
	"""Seed Python, NumPy, CPU PyTorch, and all CUDA devices."""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def save_checkpoint(path, model, optimizer, step, config, scheduler=None) -> None:
	"""Save all state required to resume a training run."""
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
	"""Restore a checkpoint and return its complete state dictionary."""
	state = torch.load(path, map_location=map_location)
	model.load_state_dict(state["model"])
	if optimizer and state.get("optimizer"):
		optimizer.load_state_dict(state["optimizer"])
	if scheduler and state.get("scheduler"):
		scheduler.load_state_dict(state["scheduler"])
	return state
