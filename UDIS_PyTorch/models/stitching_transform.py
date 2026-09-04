"""Dynamic union-canvas stitching-domain transformer."""

import math
import torch
from torch import nn
from .dlt import transform_points
from .homography_warp import homography_warp


class StitchingDomainTransformer(nn.Module):
	"""Warp equal-sized pairs onto a complete union canvas (batch size one or common canvas)."""

	def __init__(self, multiple=8):
		super().__init__()
		self.multiple = multiple

	def forward(self, image1, image2, H):
		if image1.shape != image2.shape:
			raise ValueError("paired images must have equal shapes")
		b, c, h, w = image1.shape
		device, dtype = image1.device, image1.dtype
		corners = (
			torch.tensor(
				[[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], device=device, dtype=dtype
			)
			.unsqueeze(0)
			.expand(b, -1, -1)
		)
		moved = transform_points(H, corners)
		allp = torch.cat((corners, moved), 1)
		xmin = torch.floor(allp[..., 0].amin(1))
		ymin = torch.floor(allp[..., 1].amin(1))
		xmax = torch.ceil(allp[..., 0].amax(1))
		ymax = torch.ceil(allp[..., 1].amax(1))
		# A tensor batch needs one rectangular canvas, use the union over all samples.
		x0 = xmin.min()
		y0 = ymin.min()
		x1 = xmax.max()
		y1 = ymax.max()
		cw = max(self.multiple, int(math.ceil(float(x1 - x0 + 1) / self.multiple)) * self.multiple)
		ch = max(self.multiple, int(math.ceil(float(y1 - y0 + 1) / self.multiple)) * self.multiple)
		origin = torch.stack((x0, y0)).unsqueeze(0).expand(b, -1)
		eye = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(b, -1, -1)
		w1 = homography_warp(image1, eye, (ch, cw), origin)
		w2 = homography_warp(image2, H, (ch, cw), origin)
		m1 = homography_warp(torch.ones_like(image1), eye, (ch, cw), origin).clamp(0, 1)
		m2 = homography_warp(torch.ones_like(image2), H, (ch, cw), origin).clamp(0, 1)
		return {"warp1": w1, "warp2": w2, "mask1": m1, "mask2": m2, "origin": origin}
