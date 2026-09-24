"""grid_sample warper. H maps input coordinates into output coordinates."""
import torch
from torch.nn import functional as F
def warp_image(x,H,output_size=None,mode="bilinear"):
 b,c,ih,iw=x.shape; oh,ow=output_size or (ih,iw); ys,xs=torch.meshgrid(torch.arange(oh,device=x.device,dtype=torch.float32),torch.arange(ow,device=x.device,dtype=torch.float32),indexing="ij"); q=torch.stack((xs,ys,torch.ones_like(xs)),0).reshape(3,-1)[None].expand(b,-1,-1); p=torch.linalg.inv(H.float())@q; den=p[:,2:];den=torch.where(den.abs()<1e-6,torch.full_like(den,1e-6),den);p=p[:,:2]/den; gx=2*p[:,0]/max(iw-1,1)-1; gy=2*p[:,1]/max(ih-1,1)-1; grid=torch.stack((gx,gy),-1).reshape(b,oh,ow,2).to(x.dtype); return F.grid_sample(x,grid,mode=mode,padding_mode="zeros",align_corners=True)
def warp_mask(x,H,output_size=None): return warp_image(x,H,output_size,"bilinear").clamp(0,1)
