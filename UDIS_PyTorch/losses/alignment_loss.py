"""Paper's 16:4:1 multiscale unsupervised alignment objective."""

import torch
from torch.nn import functional as F
from UDIS_PyTorch.models.dlt import DifferentiableDLT
from UDIS_PyTorch.models.homography_warp import homography_warp


class AlignmentLoss(torch.nn.Module):
	def __init__(self, weights=(16.0, 4.0, 1.0)):
		super().__init__()
		self.weights = weights
		self.dlt = DifferentiableDLT()

	def forward(self, image1, image2, pred):
		a = F.interpolate(image1, (128, 128), mode="bilinear", align_corners=False)
		b = F.interpolate(image2, (128, 128), mode="bilinear", align_corners=False)
		cumulative = (pred["delta1"], pred["delta1"] + pred["delta2"], pred["delta_final"])
		values = []
		warps = []
		for d in cumulative:
			H = self.dlt(d, 128)
			mask = homography_warp(torch.ones_like(b), H).clamp(0, 1)
			warp = homography_warp(b, H)
			values.append(F.l1_loss(warp, a * mask))
			warps.append(warp)
		return {
			"total": sum(w * v for w, v in zip(self.weights, values)),
			"level1": values[0],
			"level2": values[1],
			"level3": values[2],
			"warps": warps,
		}
