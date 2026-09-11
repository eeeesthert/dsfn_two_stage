"""Paper metrics and explicitly-labelled regional engineering extensions."""
import numpy as np
from skimage.metrics import structural_similarity

def _values(a,b,mask=None):
 a=np.asarray(a,float); b=np.asarray(b,float)
 if a.shape!=b.shape: raise ValueError('Metric inputs differ in shape')
 if mask is not None:
  m=np.asarray(mask,bool); m=np.repeat(m[...,None],a.shape[2],2) if a.ndim==3 else m; return a[m],b[m]
 return a.ravel(),b.ravel()
def compute_mse(a,b,mask=None):
 x,y=_values(a,b,mask); return float(np.mean((x-y)**2)) if len(x) else float('nan')
def compute_psnr(a,b,data_range=None,mask=None):
 mse=compute_mse(a,b,mask)
 if mse==0:return float('inf')
 if data_range is None: data_range=255. if np.issubdtype(np.asarray(a).dtype,np.integer) else 1.
 return float(10*np.log10(data_range**2/mse))
def compute_ssim(a,b,data_range=None,mask=None):
 if mask is not None:
  x,y=_values(a,b,mask); mse=np.mean((x-y)**2); var=np.var(x)+np.var(y); return float(1-mse/(var+1e-12))
 if data_range is None:data_range=255. if np.issubdtype(np.asarray(a).dtype,np.integer) else 1.
 return float(structural_similarity(a,b,data_range=data_range,channel_axis=-1 if np.asarray(a).ndim==3 else None))
def compute_ncc(a,b,mask=None):
 x,y=_values(a,b,mask); x=x-x.mean(); y=y-y.mean(); return float(x@y/(np.linalg.norm(x)*np.linalg.norm(y)+1e-12))
def evaluate_pair(warped_ref,warped_target,fused,region_mask):
 return {'mse':compute_mse(warped_ref,warped_target,region_mask),'psnr':compute_psnr(warped_ref,warped_target,mask=region_mask),'ssim':compute_ssim(warped_ref,warped_target,mask=region_mask),'ncc':compute_ncc(warped_ref,warped_target,region_mask)}
