"""True local correlation volume used by all three UDIS levels."""

import torch
from torch import nn
from torch.nn import functional as F


class CostVolume(nn.Module):
	def __init__(self, search_range: int, eps: float = 1e-6):
		super().__init__()
		self.search_range = search_range
		self.eps = eps

	@property
	def channels(self):
		return (2 * self.search_range + 1) ** 2

	def forward(self, feature1: torch.Tensor, feature2: torch.Tensor) -> torch.Tensor:
		if feature1.shape != feature2.shape:
			raise ValueError("feature shapes must match")
		a = feature1 / torch.sqrt(torch.sum(feature1.square(), 1, keepdim=True) + self.eps)
		b = feature2 / torch.sqrt(torch.sum(feature2.square(), 1, keepdim=True) + self.eps)
		r = self.search_range
		bp = F.pad(b, (r, r, r, r))
		h, w = a.shape[-2:]
		costs = []
		for dy in range(-r, r + 1):
			for dx in range(-r, r + 1):
				shifted = bp[:, :, dy + r : dy + r + h, dx + r : dx + r + w]
				costs.append(torch.mean(a * shifted, 1, keepdim=True))
		return F.leaky_relu(torch.cat(costs, 1), negative_slope=0.1)
