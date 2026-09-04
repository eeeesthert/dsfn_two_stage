"""Shared Siamese feature pyramid from the UDIS alignment stage."""

import torch
from torch import nn
from .common import init_glorot


def block(cin: int, cout: int) -> nn.Sequential:
	return nn.Sequential(
		nn.Conv2d(cin, cout, 3, padding=1),
		nn.ReLU(inplace=True),
		nn.Conv2d(cout, cout, 3, padding=1),
		nn.ReLU(inplace=True),
	)


class FeaturePyramid(nn.Module):
	"""Map [B,1,128,128] to F1..F4 at 1, 1/2, 1/4, 1/8 scale."""

	def __init__(self):
		super().__init__()
		self.b1 = block(1, 64)
		self.b2 = block(64, 64)
		self.b3 = block(64, 128)
		self.b4 = block(128, 128)
		self.pool = nn.MaxPool2d(2)
		init_glorot(self)

	def forward(self, x: torch.Tensor):
		f1 = self.b1(x)
		f2 = self.b2(self.pool(f1))
		f3 = self.b3(self.pool(f2))
		f4 = self.b4(self.pool(f3))
		return (f1, f2, f3, f4)
