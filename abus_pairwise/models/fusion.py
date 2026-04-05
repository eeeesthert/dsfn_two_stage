from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, dilation: int = 2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=dilation, dilation=dilation),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=dilation, dilation=dilation),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SoftSeamFusionUNet(nn.Module):
    """Dilated Conv U-Net to predict soft seam masks and blended output."""

    def __init__(self):
        super().__init__()
        self.e1 = DilatedBlock(6, 64, dilation=1)
        self.p1 = nn.MaxPool2d(2)
        self.e2 = DilatedBlock(64, 128, dilation=2)
        self.p2 = nn.MaxPool2d(2)
        self.b = DilatedBlock(128, 256, dilation=4)

        self.u2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.d2 = DilatedBlock(256, 128, dilation=2)
        self.u1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.d1 = DilatedBlock(128, 64, dilation=1)

        self.mask_head = nn.Conv2d(64, 1, 1)

    def forward(self, left_warp: torch.Tensor, right_warp: torch.Tensor) -> dict[str, torch.Tensor]:
        x = torch.cat([left_warp, right_warp], dim=1)

        e1 = self.e1(x)
        e2 = self.e2(self.p1(e1))
        b = self.b(self.p2(e2))

        up2 = self.u2(b)
        if up2.shape[-2:] != e2.shape[-2:]:
            up2 = F.interpolate(up2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.d2(torch.cat([up2, e2], dim=1))

        up1 = self.u1(d2)
        if up1.shape[-2:] != e1.shape[-2:]:
            up1 = F.interpolate(up1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.d1(torch.cat([up1, e1], dim=1))

        raw_right = torch.sigmoid(self.mask_head(d1))
        valid_left = (left_warp.sum(1, keepdim=True) > 0).float()
        valid_right = (right_warp.sum(1, keepdim=True) > 0).float()

        # Constrain masks by valid support, then renormalize.
        m_right = raw_right * valid_right
        m_left = (1.0 - raw_right) * valid_left
        norm = (m_left + m_right).clamp_min(1e-6)
        m_left = m_left / norm
        m_right = m_right / norm
        stitched = m_left * left_warp + m_right * right_warp

        return {
            "mask_left": m_left,
            "mask_right": m_right,
            "stitched": stitched,
        }
