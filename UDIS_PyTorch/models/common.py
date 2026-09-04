"""Shared layers and paper-compatible Glorot initialization."""

import torch.nn as nn


def init_glorot(module: nn.Module) -> None:
	"""Initialize convolutional/linear layers with Glorot and zero biases."""
	for layer in module.modules():
		if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
			nn.init.xavier_uniform_(layer.weight)
			if layer.bias is not None:
				nn.init.zeros_(layer.bias)
