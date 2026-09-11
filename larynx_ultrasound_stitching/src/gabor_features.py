"""Paper-equation and OpenCV real-valued Gabor feature banks."""
import cv2
import numpy as np

def build_gabor_filter_bank(config:dict)->np.ndarray:
    scales=int(config['scales']); orientations=int(config['orientations']); sigma=float(config['sigma']); n=int(config.get('kernel_size',31)); n+=1-n%2
    yy,xx=np.mgrid[-n//2+1:n//2+1,-n//2+1:n//2+1]; kernels=[]
    for v in range(scales):
      kv=2**(-(v+2)/2)*np.pi
      for u in range(orientations):
        phi=u*np.pi/orientations
        if config.get('backend','paper')=='opencv': kernel=cv2.getGaborKernel((n,n),sigma,phi,2*np.pi/kv,1,0,ktype=cv2.CV_32F)
        else:
          phase=kv*(xx*np.cos(phi)+yy*np.sin(phi)); envelope=(kv**2/sigma**2)*np.exp(-(kv**2)*(xx**2+yy**2)/(2*sigma**2)); kernel=envelope*(np.cos(phase)-np.exp(-sigma**2/2))
        kernel=np.asarray(kernel,np.float32); kernel-=kernel.mean(); kernels.append(kernel)
    return np.stack(kernels)
def apply_gabor_bank(image:np.ndarray,kernels:np.ndarray)->np.ndarray: return np.stack([cv2.filter2D(image,cv2.CV_32F,k,borderType=cv2.BORDER_REFLECT) for k in kernels])
def compute_gabor_energy(responses:np.ndarray,window:int,downsample_factor:int=1)->np.ndarray:
    if window<1 or window%2==0: raise ValueError('energy_window must be positive odd')
    energy=np.stack([cv2.blur(r*r,(window,window)) for r in responses]).astype(np.float32)
    # Bicubic reduced representation is exposed without sacrificing full maps used by refinement.
    if downsample_factor>1:
      h,w=energy.shape[1:]; _=np.stack([cv2.resize(e,(max(1,w//downsample_factor),max(1,h//downsample_factor)),interpolation=cv2.INTER_CUBIC) for e in energy])
    return energy
def sample_feature_map(feature_map:np.ndarray,points:np.ndarray)->np.ndarray:
    maps=feature_map.astype(np.float32); h,w=maps.shape[1:]; p=np.asarray(points,np.float32); out=[]
    for x,y in p:
      if x<0 or y<0 or x>w-1 or y>h-1: out.append(np.zeros(len(maps),np.float32)); continue
      vals=np.array([cv2.getRectSubPix(m,(1,1),(float(x),float(y)))[0,0] for m in maps]); vals/=np.linalg.norm(vals)+1e-12; out.append(vals)
    return np.asarray(out,np.float32)
def gabor_feature_distance(ref_map,target_map,ref_points,target_points):
    r=sample_feature_map(ref_map,ref_points)-sample_feature_map(target_map,target_points); return r,float(np.sum(r*r))
