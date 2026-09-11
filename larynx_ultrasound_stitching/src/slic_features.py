"""SLIC super-pixel segmentation."""
import numpy as np
from skimage.segmentation import slic,find_boundaries

def compute_slic(image:np.ndarray,config:dict)->tuple[np.ndarray,np.ndarray]:
    """Return zero-based SLIC labels and boundary mask."""
    if image.ndim!=2: raise ValueError('SLIC feature image must be grayscale')
    labels=slic(image,n_segments=int(config['n_segments']),compactness=float(config['compactness']),sigma=float(config['sigma']),enforce_connectivity=bool(config['enforce_connectivity']),channel_axis=None,start_label=0)
    return labels.astype(np.int32),find_boundaries(labels,mode='thick')
