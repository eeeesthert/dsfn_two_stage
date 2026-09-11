"""Conservative ultrasound preprocessing (unreported details are configurable)."""
import cv2
import numpy as np
from .io_utils import normalize_image

def enhance_ultrasound(gray: np.ndarray, config: dict) -> np.ndarray:
    """Normalize, optionally apply CLAHE and mild Gaussian smoothing."""
    out=normalize_image(gray)
    if config.get('clahe',False):
        u8=np.round(out*255).astype(np.uint8)
        grid=int(config.get('clahe_grid_size',8))
        out=cv2.createCLAHE(float(config.get('clahe_clip_limit',2.0)),(grid,grid)).apply(u8).astype(np.float32)/255
    sigma=float(config.get('gaussian_sigma',0))
    if sigma>0: out=cv2.GaussianBlur(out,(0,0),sigma)
    return np.clip(out,0,1).astype(np.float32)
