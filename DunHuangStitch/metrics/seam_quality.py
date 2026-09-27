import torch
from torch.nn import functional as F
from .alignment_metrics import ssim
def seam_quality(a,b,f,seam,radius=5,gamma=1000.):
 edge=(seam[:,:,:,1:]-seam[:,:,:,:-1]).abs();edge=F.pad(edge,(0,1));d=F.max_pool2d(edge,2*radius+1,1,radius);aa=a*d;bb=b*d;ff=f*d;ep=(2-ssim(ff,aa)-ssim(ff,bb))/2;ept=(torch.linalg.vector_norm(ff-aa,dim=1)+torch.linalg.vector_norm(ff-bb,dim=1)).mean()/2;return {"E_patch":ep,"E_point":ept,"Q_seam":gamma*ep+ept}
