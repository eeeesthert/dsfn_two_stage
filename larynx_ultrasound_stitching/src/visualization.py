"""Feature, registration, and mandatory debug visualizations."""
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import mark_boundaries
from .sift_features import FeatureSet
from .feature_matching import Match

def as_u8(x):
 x=np.asarray(x); return x if x.dtype==np.uint8 else np.clip(x*255,0,255).astype(np.uint8)
def draw_slic(image,labels): return (mark_boundaries(image,labels,color=(1,0,0))*255).astype(np.uint8)
def draw_features(image,fs):
 out=cv2.cvtColor(as_u8(image),cv2.COLOR_GRAY2BGR)
 for f in fs.features: cv2.circle(out,(round(f.x),round(f.y)),2,(0,255,0) if f.source_type=='global' else (0,165,255),1)
 return out
def draw_matches(ref,target,rf,tf,matches):
 a=cv2.cvtColor(as_u8(ref),cv2.COLOR_GRAY2BGR); b=cv2.cvtColor(as_u8(target),cv2.COLOR_GRAY2BGR); h=max(a.shape[0],b.shape[0]); out=np.zeros((h,a.shape[1]+b.shape[1],3),np.uint8); out[:a.shape[0],:a.shape[1]]=a; out[:b.shape[0],a.shape[1]:]=b
 for m in matches:
  p=(round(rf.features[m.reference_index].x),round(rf.features[m.reference_index].y)); q=(round(tf.features[m.target_index].x)+a.shape[1],round(tf.features[m.target_index].y)); cv2.line(out,p,q,(0,255,0),1)
 return out
def overlay(a,b,ma=None,mb=None): return np.clip(.5*a+.5*b,0,1)
def checkerboard(a,b,tile=32):
 yy,xx=np.indices(a.shape[:2]); m=((xx//tile+yy//tile)%2).astype(bool); return np.where(m[...,None],a,b) if a.ndim==3 else np.where(m,a,b)
def montage(images,path):
 fig,axes=plt.subplots(8,2,figsize=(14,32));
 for ax,(title,img) in zip(axes.flat,images): ax.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB) if img.ndim==3 and img.shape[2]==3 else img,cmap='gray'); ax.set_title(title); ax.axis('off')
 for ax in axes.flat[len(images):]:ax.axis('off')
 fig.tight_layout(); fig.savefig(path,dpi=120); plt.close(fig)
