import torch
from torch import nn
from torch.nn import functional as F
def sobel(x,eps=1e-6):
 kx=x.new_tensor([[-1,0,1],[-2,0,2],[-1,0,1]]).reshape(1,1,3,3);ky=kx.transpose(-1,-2);c=x.shape[1];gx=F.conv2d(x,kx.expand(c,1,3,3),padding=1,groups=c);gy=F.conv2d(x,ky.expand(c,1,3,3),padding=1,groups=c);return torch.sqrt(gx.square()+gy.square()+eps)
class SeamLoss(nn.Module):
 def __init__(self,alpha_edge=1.,lf_mode="paper_literal",lambda_b=1.,lambda_o=1.,lambda_f=1.):super().__init__();self.alpha=alpha_edge;self.mode=lf_mode;self.ws=(lambda_b,lambda_o,lambda_f)
 def forward(self,a,b,ma,mb,out):
  f=out["stitched"];s=out["seam_mask_r_raw"];mbr=ma*sobel(mb);mbt=mb*sobel(ma);lb=((f-a).abs()*mbr).mean()+((f-b).abs()*mbt).mean();ld=torch.linalg.vector_norm(a-b,dim=1,keepdim=True);le=torch.linalg.vector_norm(sobel(a)-sobel(b),dim=1,keepdim=True);energy=ld+self.alpha*le;dh=(s[:,:,:,1:]-s[:,:,:,:-1]).abs();dv=(s[:,:,1:]-s[:,:,:-1]).abs();lo=(dh*(energy[:,:,:,1:]+energy[:,:,:,:-1])).mean()+(dv*(energy[:,:,1:]+energy[:,:,:-1])).mean();fh=f[:,:,:,1:]-f[:,:,:,:-1];fv=f[:,:,1:]-f[:,:,:-1];fh=fh.abs() if self.mode=="stable_abs" else fh;fv=fv.abs() if self.mode=="stable_abs" else fv;lf=(dh*fh).mean()+(dv*fv).mean();total=self.ws[0]*lb+self.ws[1]*lo+self.ws[2]*lf;return {"Lb":lb,"Lo":lo,"Lf":lf,"Ls":total,"loss":total}
