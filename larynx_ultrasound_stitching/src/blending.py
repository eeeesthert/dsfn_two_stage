"""Linear weighted smoothing on the union canvas."""
import cv2
import numpy as np

def linear_blend(reference,target,reference_mask,target_mask,method='linear_distance'):
 if reference.shape!=target.shape or reference_mask.shape!=reference.shape[:2]: raise ValueError('Shape mismatch')
 m1=reference_mask.astype(bool); m2=target_mask.astype(bool); union=m1|m2
 if not union.any(): raise ValueError('Empty union')
 if method=='linear_distance':
  d1=cv2.distanceTransform(m1.astype(np.uint8),cv2.DIST_L2,5); d2=cv2.distanceTransform(m2.astype(np.uint8),cv2.DIST_L2,5); w1=d1/(d1+d2+1e-8)
 elif method=='intensity_weighted':
  a=reference.mean(2) if reference.ndim==3 else reference; b=target.mean(2) if target.ndim==3 else target; w1=(a+1e-6)/(a+b+2e-6)
 else: raise ValueError(f'Unknown blending method {method}')
 w1=np.where(m1&~m2,1,np.where(m2&~m1,0,w1)); weights=w1[...,None] if reference.ndim==3 else w1; out=weights*reference+(1-weights)*target; out[~union]=0
 return out.astype(np.float32),w1.astype(np.float32)
def crop_to_mask(image,mask):
 ys,xs=np.where(mask); 
 if not len(xs): raise ValueError('Cannot crop empty mask')
 return image[ys.min():ys.max()+1,xs.min():xs.max()+1]
