"""Differentiable four-point DLT.

Convention: offsets move source corners P to target corners P'=P+delta.  The
returned H is a forward source-pixel -> target-pixel mapping.  Consequently
`homography_warp` inverts H because grid_sample performs backward sampling.
Corner order is top-left, top-right, bottom-left, bottom-right.
"""
import torch
from torch import nn
class DifferentiableDLT(nn.Module):
    def __init__(self, eps: float=0.0): super().__init__(); self.eps=eps
    def forward(self, offsets: torch.Tensor, patch_size):
        b=offsets.shape[0]; dtype,device=offsets.dtype,offsets.device
        if torch.is_tensor(patch_size):
            s=patch_size.to(device=device,dtype=dtype).reshape(b,1)
            z=torch.zeros_like(s); corners=torch.stack((torch.cat((z,z),1),torch.cat((s,z),1),torch.cat((z,s),1),torch.cat((s,s),1)),1)
        else:
            corners=torch.tensor([[0,0],[patch_size,0],[0,patch_size],[patch_size,patch_size]],device=device,dtype=dtype).unsqueeze(0).expand(b,-1,-1)
        target=corners+offsets.reshape(b,4,2); x,y=corners[...,0],corners[...,1]; u,v=target[...,0],target[...,1]
        zero=torch.zeros_like(x); one=torch.ones_like(x)
        rows=[]; rhs=[]
        for i in range(4):
            rows += [torch.stack((x[:,i],y[:,i],one[:,i],zero[:,i],zero[:,i],zero[:,i],-u[:,i]*x[:,i],-u[:,i]*y[:,i]),1),
                     torch.stack((zero[:,i],zero[:,i],zero[:,i],x[:,i],y[:,i],one[:,i],-v[:,i]*x[:,i],-v[:,i]*y[:,i]),1)]
            rhs += [u[:,i],v[:,i]]
        A=torch.stack(rows,1); q=torch.stack(rhs,1).unsqueeze(-1)
        if self.eps: A=A+self.eps*torch.eye(8,device=device,dtype=dtype).unsqueeze(0)
        h=torch.linalg.solve(A,q).squeeze(-1); return torch.cat((h,torch.ones(b,1,device=device,dtype=dtype)),1).reshape(b,3,3)

def transform_points(H: torch.Tensor, points: torch.Tensor, eps: float=1e-6):
    """Apply forward pixel homographies to [B,N,2] points."""
    p=torch.cat((points,torch.ones_like(points[...,:1])),2); q=torch.bmm(H,p.transpose(1,2)).transpose(1,2)
    z=q[...,2:]; z=torch.where(z.abs()<eps,torch.where(z<0,-torch.full_like(z,eps),torch.full_like(z,eps)),z)
    return q[...,:2]/z
