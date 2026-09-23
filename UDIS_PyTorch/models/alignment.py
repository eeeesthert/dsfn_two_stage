"""Stage 1: feature extraction, correlation, DLT, warping, and alignment.

Coordinate convention: corner offsets map source points to target points. DLT
returns a forward source-to-target homography. ``homography_warp`` inverts that
matrix because ``grid_sample`` performs target-to-source backward sampling.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


# Shared initialization and correlation

def init_glorot(module: nn.Module) -> None:
	"""Initialize convolutional/linear layers with Glorot and zero biases."""
	for layer in module.modules():
		if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
			nn.init.xavier_uniform_(layer.weight)
			if layer.bias is not None:
				nn.init.zeros_(layer.bias)

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
		a = feature1 / torch.sqrt(
			torch.sum(feature1.square(), 1, keepdim=True) + self.eps
		)
		b = feature2 / torch.sqrt(
			torch.sum(feature2.square(), 1, keepdim=True) + self.eps
		)
		r = self.search_range
		bp = F.pad(b, (r, r, r, r))
		h, w = a.shape[-2:]
		costs = []
		for dy in range(-r, r + 1):
			for dx in range(-r, r + 1):
				shifted = bp[:, :, dy + r : dy + r + h, dx + r : dx + r + w]
				costs.append(torch.mean(a * shifted, 1, keepdim=True))
		return F.leaky_relu(torch.cat(costs, 1), negative_slope=0.1)

# Differentiable projective geometry


class DifferentiableDLT(nn.Module):
	def __init__(self, eps: float = 0.0):
		super().__init__()
		self.eps = eps

	def forward(self, offsets: torch.Tensor, patch_size):
		b = offsets.shape[0]
		dtype, device = offsets.dtype, offsets.device
		if torch.is_tensor(patch_size):
			s = patch_size.to(device=device, dtype=dtype).reshape(b, 1)
			z = torch.zeros_like(s)
			corners = torch.stack(
				(
					torch.cat((z, z), 1),
					torch.cat((s, z), 1),
					torch.cat((z, s), 1),
					torch.cat((s, s), 1),
				),
				1,
			)
		else:
			corners = (
				torch.tensor(
					[
						[0, 0],
						[patch_size, 0],
						[0, patch_size],
						[patch_size, patch_size],
					],
					device=device,
					dtype=dtype,
				)
				.unsqueeze(0)
				.expand(b, -1, -1)
			)
		target = corners + offsets.reshape(b, 4, 2)
		x, y = corners[..., 0], corners[..., 1]
		u, v = target[..., 0], target[..., 1]
		zero = torch.zeros_like(x)
		one = torch.ones_like(x)
		rows = []
		rhs = []
		for i in range(4):
			rows += [
				torch.stack(
					(
						x[:, i],
						y[:, i],
						one[:, i],
						zero[:, i],
						zero[:, i],
						zero[:, i],
						-u[:, i] * x[:, i],
						-u[:, i] * y[:, i],
					),
					1,
				),
				torch.stack(
					(
						zero[:, i],
						zero[:, i],
						zero[:, i],
						x[:, i],
						y[:, i],
						one[:, i],
						-v[:, i] * x[:, i],
						-v[:, i] * y[:, i],
					),
					1,
				),
			]
			rhs += [u[:, i], v[:, i]]
		A = torch.stack(rows, 1)
		q = torch.stack(rhs, 1).unsqueeze(-1)
		if self.eps:
			A = A + self.eps * torch.eye(8, device=device, dtype=dtype).unsqueeze(0)
		h = torch.linalg.solve(A, q).squeeze(-1)
		return torch.cat((h, torch.ones(b, 1, device=device, dtype=dtype)), 1).reshape(
			b, 3, 3
		)


def transform_points(H: torch.Tensor, points: torch.Tensor, eps: float = 1e-6):
	"""Apply forward pixel homographies to [B,N,2] points."""
	p = torch.cat((points, torch.ones_like(points[..., :1])), 2)
	q = torch.bmm(H, p.transpose(1, 2)).transpose(1, 2)
	z = q[..., 2:]
	z = torch.where(
		z.abs() < eps,
		torch.where(z < 0, -torch.full_like(z, eps), torch.full_like(z, eps)),
		z,
	)
	return q[..., :2] / z

def pixel_to_normalized(x: torch.Tensor, size: int) -> torch.Tensor:
	"""Pixel centers to align_corners=True normalized coordinates."""
	return torch.zeros_like(x) if size <= 1 else 2 * x / (size - 1) - 1


def homography_warp(
	source: torch.Tensor,
	H_source_to_target: torch.Tensor,
	output_size=None,
	origin=None,
):
	"""Backward-sample source onto a target canvas. H maps source to target."""
	b, _, hs, ws = source.shape
	ho, wo = output_size or (hs, ws)
	device, dtype = source.device, source.dtype
	yy, xx = torch.meshgrid(
		torch.arange(ho, device=device, dtype=dtype),
		torch.arange(wo, device=device, dtype=dtype),
		indexing="ij",
	)
	if origin is not None:
		xx = xx + origin[:, 0, None, None]
		yy = yy + origin[:, 1, None, None]
	else:
		xx = xx.expand(b, -1, -1)
		yy = yy.expand(b, -1, -1)
	p = torch.stack((xx, yy, torch.ones_like(xx)), 1).reshape(b, 3, -1)
	q = torch.bmm(torch.linalg.inv(H_source_to_target), p)
	z = q[:, 2:3]
	z = torch.where(
		z.abs() < 1e-6,
		torch.where(z < 0, -torch.full_like(z, 1e-6), torch.full_like(z, 1e-6)),
		z,
	)
	gx = pixel_to_normalized(q[:, 0] / z[:, 0], ws)
	gy = pixel_to_normalized(q[:, 1] / z[:, 0], hs)
	grid = torch.stack((gx, gy), -1).reshape(b, ho, wo, 2)
	return F.grid_sample(
		source, grid, mode="bilinear", padding_mode="zeros", align_corners=True
	)

# Shared feature pyramid and coarse-to-fine regression


def _feature_block(cin: int, cout: int) -> nn.Sequential:
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
		self.b1 = _feature_block(1, 64)
		self.b2 = _feature_block(64, 64)
		self.b3 = _feature_block(64, 128)
		self.b4 = _feature_block(128, 128)
		self.pool = nn.MaxPool2d(2)
		init_glorot(self)

	def forward(self, x: torch.Tensor):
		f1 = self.b1(x)
		f2 = self.b2(self.pool(f1))
		f3 = self.b3(self.pool(f2))
		f4 = self.b4(self.pool(f3))
		return (f1, f2, f3, f4)

class Regressor(nn.Module):
	def __init__(self, channels, strides, spatial, hidden, dropout):
		super().__init__()
		layers = []
		for ci, co, st in zip(channels[:-1], channels[1:], strides):
			layers += [
				nn.Conv2d(ci, co, 3, stride=st, padding=1),
				nn.ReLU(inplace=True),
			]
		self.conv = nn.Sequential(*layers)
		out_sp = spatial
		for s in strides:
			out_sp = (out_sp + 2 - 3) // s + 1
		self.fc = nn.Sequential(
			nn.Linear(channels[-1] * out_sp * out_sp, hidden),
			nn.ReLU(inplace=True),
			nn.Dropout(dropout),
			nn.Linear(hidden, 8),
		)
		init_glorot(self)

	def forward(self, x):
		return self.fc(self.conv(x).flatten(1))


class HomographyNet(nn.Module):
	"""Return three residual corner offsets and their sum in 128 coordinates."""

	def __init__(self, dropout=0.5, search_ranges=(16, 8, 4)):
		super().__init__()
		r1, r2, r3 = search_ranges
		self.features = FeaturePyramid()
		self.dlt = DifferentiableDLT()
		self.cv1, self.cv2, self.cv3 = CostVolume(r1), CostVolume(r2), CostVolume(r3)
		self.reg1 = Regressor(
			[self.cv1.channels, 512, 512, 512], [1, 1, 1], 16, 1024, dropout
		)
		self.reg2 = Regressor(
			[self.cv2.channels, 256, 256, 256], [1, 1, 2], 32, 512, dropout
		)
		self.reg3 = Regressor(
			[self.cv3.channels, 128, 128, 128], [1, 2, 2], 64, 256, dropout
		)

	def forward(self, image1, image2):
		a = F.interpolate(
			image1, size=(128, 128), mode="bilinear", align_corners=False
		).mean(1, keepdim=True)
		b = F.interpolate(
			image2, size=(128, 128), mode="bilinear", align_corners=False
		).mean(1, keepdim=True)
		_, a2, a3, a4 = self.features(a)
		_, b2, b3, b4 = self.features(b)
		d1 = self.reg1(self.cv1(a4, b4))
		h1 = self.dlt(d1 / 4, 32)
		b3w = homography_warp(b3, h1)
		d2 = self.reg2(self.cv2(a3, b3w))
		d12 = d1 + d2
		h2 = self.dlt(d12 / 2, 64)
		b2w = homography_warp(b2, h2)
		d3 = self.reg3(self.cv3(a2, b2w))
		return {"delta1": d1, "delta2": d2, "delta3": d3, "delta_final": d12 + d3}

# Full-resolution stitching canvas and public Stage-1 pipeline


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
				[[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]],
				device=device,
				dtype=dtype,
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
		# A tensor batch needs one rectangle, so use the union over all samples.
		x0 = xmin.min()
		y0 = ymin.min()
		x1 = xmax.max()
		y1 = ymax.max()
		cw = max(
			self.multiple,
			int(math.ceil(float(x1 - x0 + 1) / self.multiple)) * self.multiple,
		)
		ch = max(
			self.multiple,
			int(math.ceil(float(y1 - y0 + 1) / self.multiple)) * self.multiple,
		)
		origin = torch.stack((x0, y0)).unsqueeze(0).expand(b, -1)
		eye = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(b, -1, -1)
		w1 = homography_warp(image1, eye, (ch, cw), origin)
		w2 = homography_warp(image2, H, (ch, cw), origin)
		m1 = homography_warp(torch.ones_like(image1), eye, (ch, cw), origin).clamp(0, 1)
		m2 = homography_warp(torch.ones_like(image2), H, (ch, cw), origin).clamp(0, 1)
		return {"warp1": w1, "warp2": w2, "mask1": m1, "mask2": m2, "origin": origin}

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
		return {
			"delta": delta,
			"H": H,
			**{k: out[k] for k in ("warp1", "warp2", "mask1", "mask2")},
		}

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
