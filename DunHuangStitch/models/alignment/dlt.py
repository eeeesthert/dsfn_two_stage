"""Differentiable normalized batched DLT. H maps source/target coordinates to destination/reference."""
import torch

def transform_points(H: torch.Tensor, p: torch.Tensor, eps: float=1e-6)->torch.Tensor:
 q=torch.cat((p,torch.ones_like(p[...,:1])), -1); q=torch.bmm(q,H.transpose(1,2)); return q[...,:2]/torch.where(q[...,2:].abs()<eps, torch.full_like(q[...,2:],eps), q[...,2:])
def _norm(p,eps):
 c=p.mean(1,keepdim=True); d=torch.linalg.vector_norm(p-c,dim=-1).mean(1).clamp_min(eps); s=2**.5/d
 T=torch.zeros(p.shape[0],3,3,device=p.device,dtype=p.dtype); T[:,0,0]=s;T[:,1,1]=s;T[:,0,2]=-s*c[:,0,0];T[:,1,2]=-s*c[:,0,1];T[:,2,2]=1
 return transform_points(T,p,eps),T
def dlt_homography(src:torch.Tensor,dst:torch.Tensor,eps:float=1e-6)->torch.Tensor:
 dtype=src.dtype; s,T1=_norm(src.float(),eps); d,T2=_norm(dst.float(),eps); x,y=s.unbind(-1);u,v=d.unbind(-1); z=torch.zeros_like(x);o=torch.ones_like(x)
 A=torch.stack([x,y,o,z,z,z,-u*x,-u*y,z,z,z,x,y,o,-v*x,-v*y],-1).reshape(-1,8,8); b=torch.stack([u,v],-1).reshape(-1,8,1)
 h=torch.linalg.solve(A+eps*torch.eye(8,device=A.device)[None],b).squeeze(-1); Hn=torch.cat((h,torch.ones_like(h[:,:1])),1).reshape(-1,3,3); H=torch.linalg.inv(T2)@Hn@T1; return (H/H[:,2:3,2:3].clamp_min(eps)).to(dtype)
def image_corners(b,h,w,device,dtype):
 p=torch.tensor([[0,0],[w-1,0],[w-1,h-1],[0,h-1]],device=device,dtype=dtype); return p[None].expand(b,-1,-1)
