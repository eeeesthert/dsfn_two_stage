"""Replaceable saliency backends and Eq. (10) set variance."""
import cv2
import numpy as np
from scipy.spatial import cKDTree

def _norm(x):
 x=np.asarray(x,np.float32); lo,hi=float(x.min()),float(x.max()); return (x-lo)/(hi-lo) if hi>lo else np.zeros_like(x)
def spectral_residual(image):
 f=np.fft.fft2(image); log=np.log(np.abs(f)+1e-8); avg=cv2.blur(log.astype(np.float32),(3,3)); residual=np.exp(log-avg+1j*np.angle(f)); sal=np.abs(np.fft.ifft2(residual))**2; return _norm(cv2.GaussianBlur(sal.astype(np.float32),(9,9),2.5))
def compute_saliency(image:np.ndarray,method='spectral_residual')->np.ndarray:
 if method=='spectral_residual': return spectral_residual(image)
 if method=='gradient':
  return _norm(cv2.magnitude(cv2.Sobel(image,cv2.CV_32F,1,0),cv2.Sobel(image,cv2.CV_32F,0,1)))
 if method=='intensity_variance': return _norm(cv2.blur(image*image,(9,9))-cv2.blur(image,(9,9))**2)
 if method=='fine_grained' and hasattr(cv2,'saliency'): return _norm(cv2.saliency.StaticSaliencyFineGrained_create().computeSaliency(image)[1])
 raise ValueError(f'Unavailable saliency method: {method}')
def extract_salient_points(saliency,percentile,max_points,min_distance):
 threshold=np.percentile(saliency,percentile); size=2*int(min_distance)+1; maxima=(saliency==cv2.dilate(saliency,np.ones((size,size),np.uint8)))&(saliency>threshold); ys,xs=np.where(maxima); order=np.argsort(saliency[ys,xs])[::-1][:max_points]; return np.c_[xs[order],ys[order]].astype(float)
def saliency_variance(points:np.ndarray)->float:
 p=np.asarray(points,float); return 0.0 if len(p)==0 else float(np.sqrt(np.sum((p-p.mean(0))**2)/len(p)))
def local_saliency_variance(points:np.ndarray,k=8)->np.ndarray:
 p=np.asarray(points,float)
 if not len(p): return np.empty(0)
 tree=cKDTree(p); _,ids=tree.query(p,k=min(k,len(p))); return np.array([saliency_variance(p[np.atleast_1d(i)]) for i in ids])
def match_salient_points(reference_points,target_points,H,max_distance):
 from .homography import transform_points
 projected=transform_points(target_points,H); tree=cKDTree(reference_points); d,idx=tree.query(projected,distance_upper_bound=max_distance); ok=np.isfinite(d)&(idx<len(reference_points)); return reference_points[idx[ok]],target_points[ok]
