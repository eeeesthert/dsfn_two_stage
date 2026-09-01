import torch
from torch.nn import functional as F
def cost_volume(ref,tgt,radius):
 ref=F.normalize(ref,dim=1);tgt=F.normalize(tgt,dim=1); b,c,h,w=ref.shape;p=F.pad(tgt,(radius,)*4); out=[]
 for dy in range(2*radius+1):
  for dx in range(2*radius+1): out.append((ref*p[:,:,dy:dy+h,dx:dx+w]).sum(1))
 return torch.stack(out,1)
