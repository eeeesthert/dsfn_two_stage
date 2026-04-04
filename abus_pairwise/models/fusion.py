from __future__ import annotations

import torch
import torch.nn as nn


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

        d2 = self.d2(torch.cat([self.u2(b), e2], dim=1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], dim=1))

        m_right = torch.sigmoid(self.mask_head(d1))
        m_left = 1.0 - m_right
        stitched = m_left * left_warp + m_right * right_warp

        return {
            "mask_left": m_left,
            "mask_right": m_right,
            "stitched": stitched,
        }
