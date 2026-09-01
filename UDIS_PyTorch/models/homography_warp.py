"""Pixel-coordinate homography warping using differentiable grid_sample."""
import torch
from torch.nn import functional as F

def pixel_to_normalized(x: torch.Tensor, size: int) -> torch.Tensor:
    """Pixel centers to align_corners=True normalized coordinates."""
    return torch.zeros_like(x) if size<=1 else 2*x/(size-1)-1

def homography_warp(source: torch.Tensor, H_source_to_target: torch.Tensor, output_size=None, origin=None):
    """Backward-sample source onto target canvas; H is forward source->target."""
    b,_,hs,ws=source.shape; ho,wo=output_size or (hs,ws); device,dtype=source.device,source.dtype
    yy,xx=torch.meshgrid(torch.arange(ho,device=device,dtype=dtype),torch.arange(wo,device=device,dtype=dtype),indexing='ij')
    if origin is not None: xx=xx+origin[:,0,None,None]; yy=yy+origin[:,1,None,None]
    else: xx=xx.expand(b,-1,-1); yy=yy.expand(b,-1,-1)
    p=torch.stack((xx,yy,torch.ones_like(xx)),1).reshape(b,3,-1)
    q=torch.bmm(torch.linalg.inv(H_source_to_target),p); z=q[:,2:3]; z=torch.where(z.abs()<1e-6,torch.where(z<0,-torch.full_like(z,1e-6),torch.full_like(z,1e-6)),z)
    gx=pixel_to_normalized(q[:,0]/z[:,0],ws); gy=pixel_to_normalized(q[:,1]/z[:,0],hs)
    grid=torch.stack((gx,gy),-1).reshape(b,ho,wo,2)
    return F.grid_sample(source,grid,mode='bilinear',padding_mode='zeros',align_corners=True)
