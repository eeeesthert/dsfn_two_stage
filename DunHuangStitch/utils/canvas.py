import math,torch
from models.alignment.dlt import image_corners,transform_points
from models.alignment.warper import warp_image,warp_mask
def compute_union_canvas(ref,tgt,H):
 b,_,hr,wr=ref.shape;ht,wt=tgt.shape[-2:];rc=image_corners(b,hr,wr,ref.device,ref.dtype);tc=transform_points(H,image_corners(b,ht,wt,ref.device,ref.dtype));allp=torch.cat((rc,tc),1);mn=torch.floor(allp.amin(1));mx=torch.ceil(allp.amax(1));oh=int((mx[:,1]-mn[:,1]+1).max());ow=int((mx[:,0]-mn[:,0]+1).max());T=torch.eye(3,device=ref.device,dtype=ref.dtype)[None].repeat(b,1,1);T[:,0,2]=-mn[:,0];T[:,1,2]=-mn[:,1];onesr=torch.ones(b,1,hr,wr,device=ref.device);onest=torch.ones(b,1,ht,wt,device=ref.device);return warp_image(ref,T,(oh,ow)),warp_image(tgt,T@H,(oh,ow)),warp_mask(onesr,T,(oh,ow)),warp_mask(onest,T@H,(oh,ow)),T
