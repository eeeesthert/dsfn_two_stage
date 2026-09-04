"""Inference-only composition of independently trained UDIS stages."""

from torch import nn
from .alignment import AlignmentPipeline
from .reconstruction_net import ReconstructionNet


class UDIS(nn.Module):
	def __init__(self, alignment=None, reconstruction=None):
		super().__init__()
		self.alignment = alignment or AlignmentPipeline()
		self.reconstruction = reconstruction or ReconstructionNet()

	def forward(self, image1, image2):
		a = self.alignment(image1, image2)
		r = self.reconstruction(a["warp1"], a["warp2"])
		return {
			"H": a["H"],
			"warp1": a["warp1"],
			"warp2": a["warp2"],
			"lr": r["lr"],
			"stitched": r["hr"],
		}
