"""Stage 2 reconstruction network and frozen VGG feature extractor."""

import torch
from torch import nn
from torch.nn import functional as F

from UDIS_PyTorch.utils.image import resize_image

from .alignment import init_glorot


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
	return nn.Sequential(
		nn.Conv2d(in_channels, out_channels, 3, padding=1),
		nn.ReLU(inplace=True),
		nn.Conv2d(out_channels, out_channels, 3, padding=1),
		nn.ReLU(inplace=True),
	)


class ResidualBlock(nn.Module):
	def __init__(self):
		super().__init__()
		self.c1 = nn.Conv2d(64, 64, 3, padding=1)
		self.c2 = nn.Conv2d(64, 64, 3, padding=1)

	def forward(self, x):
		return F.relu(x + self.c2(F.relu(self.c1(x), inplace=True)), inplace=True)


class ReconstructionNet(nn.Module):
	def __init__(self, lr_size=256, num_res_blocks=8):
		super().__init__()
		self.lr_size = lr_size
		self.e1 = _conv_block(6, 64)
		self.e2 = _conv_block(64, 128)
		self.e3 = _conv_block(128, 256)
		self.e4 = _conv_block(256, 512)
		self.pool = nn.MaxPool2d(2)
		self.u3 = nn.ConvTranspose2d(512, 256, 2, 2)
		self.d3 = _conv_block(512, 256)
		self.u2 = nn.ConvTranspose2d(256, 128, 2, 2)
		self.d2 = _conv_block(256, 128)
		self.u1 = nn.ConvTranspose2d(128, 64, 2, 2)
		self.d1 = _conv_block(128, 64)
		self.lr_out = nn.Conv2d(64, 3, 3, padding=1)
		self.hr_in = nn.Conv2d(9, 64, 3, padding=1)
		self.res = nn.Sequential(*[ResidualBlock() for _ in range(num_res_blocks)])
		self.hr_global = nn.Conv2d(64, 64, 3, padding=1)
		self.hr_out = nn.Conv2d(64, 3, 3, padding=1)
		init_glorot(self)

	def forward(self, warp1, warp2):
		x = resize_image(torch.cat((warp1, warp2), 1), (self.lr_size, self.lr_size))
		s1 = self.e1(x)
		s2 = self.e2(self.pool(s1))
		s3 = self.e3(self.pool(s2))
		z = self.e4(self.pool(s3))
		z = self.d3(torch.cat((self.u3(z), s3), 1))
		z = self.d2(torch.cat((self.u2(z), s2), 1))
		z = self.d1(torch.cat((self.u1(z), s1), 1))
		lr = torch.tanh(self.lr_out(z))
		up = resize_image(lr, warp1.shape[-2:])
		f0 = F.relu(self.hr_in(torch.cat((warp1, warp2, up), 1)), inplace=True)
		z = F.relu(f0 + self.hr_global(self.res(f0)), inplace=True)
		hr = torch.tanh(self.hr_out(z))
		return {"lr": lr, "hr": hr}

class VGGPerceptualExtractor(nn.Module):
	def __init__(self, pretrained=True):
		super().__init__()
		from torchvision.models import VGG19_Weights, vgg19

		self.features = vgg19(
			weights=VGG19_Weights.IMAGENET1K_V1 if pretrained else None
		).features[:33]
		for p in self.parameters():
			p.requires_grad = False
		self.register_buffer(
			"mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
		)
		self.register_buffer(
			"std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
		)
		self.eval()

	def train(self, mode=True):
		super().train(False)
		return self

	def forward(self, x):
		z = ((x + 1) / 2 - self.mean) / self.std
		out = {}
		for i, layer in enumerate(self.features):
			z = layer(z)
			if i == 14:
				out["conv3_3"] = z
			if i == 32:
				out["conv5_3"] = z
		return out
