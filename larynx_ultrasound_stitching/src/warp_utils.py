"""Non-cropping panorama canvas construction and warping."""
from dataclasses import dataclass
import cv2
import numpy as np
from .homography import transform_points
@dataclass(frozen=True)
class WarpResult:
    reference:np.ndarray; target:np.ndarray; reference_mask:np.ndarray; target_mask:np.ndarray; translation:np.ndarray; target_h_canvas:np.ndarray

def image_corners(shape):
    h,w=shape[:2]; return np.array([[0,0],[w,0],[w,h],[0,h]],float)
def union_canvas(reference_shape,target_shape,H,max_pixels=100_000_000):
    allp=np.vstack([image_corners(reference_shape),transform_points(image_corners(target_shape),H)])
    lo=np.floor(allp.min(0)); hi=np.ceil(allp.max(0)); width,height=int(hi[0]-lo[0]),int(hi[1]-lo[1])
    if width<=0 or height<=0 or width*height>max_pixels: raise ValueError(f'Unsafe canvas {width}x{height}')
    T=np.array([[1,0,-lo[0]],[0,1,-lo[1]],[0,0,1]],float); return (width,height),T

def warp_to_union(reference,target,H,max_pixels=100_000_000)->WarpResult:
    size,T=union_canvas(reference.shape,target.shape,H,max_pixels); rh,rw=reference.shape[:2]; th,tw=target.shape[:2]
    ref=cv2.warpPerspective(reference,T,size); tgt=cv2.warpPerspective(target,T@H,size)
    rm=cv2.warpPerspective(np.ones((rh,rw),np.uint8),T,size,flags=cv2.INTER_NEAREST)>0
    tm=cv2.warpPerspective(np.ones((th,tw),np.uint8),T@H,size,flags=cv2.INTER_NEAREST)>0
    return WarpResult(ref,tgt,rm,tm,T,T@H)
