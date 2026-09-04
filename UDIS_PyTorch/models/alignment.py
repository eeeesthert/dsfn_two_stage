"""Full Stage-1 inference pipeline."""

import torch
from torch import nn
from .dlt import DifferentiableDLT
from .homography_net import HomographyNet
from .stitching_transform import StitchingDomainTransformer


class AlignmentPipeline(nn.Module):
	def __init__(self, homography_net=None):
		super().__init__()
		self.net = homography_net or HomographyNet()
		self.dlt = DifferentiableDLT()
		self.stitch = StitchingDomainTransformer()

	def forward(self, image1, image2):
		if image1.shape[-2:] != image2.shape[-2:]:
			raise ValueError("inputs must have equal spatial size")
		pred = self.net(image1, image2)
		h, w = image1.shape[-2:]
		scale = image1.new_tensor([w / 128, h / 128] * 4)
		delta = pred["delta_final"] * scale
		H = (
			self.dlt(delta, image1.new_full((image1.shape[0],), float(w - 1)))
			if h == w
			else self._rect_dlt(delta, w, h)
		)
		out = self.stitch(image1, image2, H)
		return {"delta": delta, "H": H, **{k: out[k] for k in ("warp1", "warp2", "mask1", "mask2")}}

	def _rect_dlt(self, delta, w, h):
		# Normalize anisotropic rectangle to a square DLT coordinate system.
		scale = delta.new_tensor([1, (w - 1) / (h - 1)] * 4)
		hn = self.dlt(delta * scale, w - 1)
		S = (
			delta.new_tensor([[1, 0, 0], [0, (w - 1) / (h - 1), 0], [0, 0, 1]])
			.unsqueeze(0)
			.expand(delta.shape[0], -1, -1)
		)
		return torch.linalg.inv(S) @ hn @ S
