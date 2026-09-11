"""Global and masked super-pixel SIFT features."""
from dataclasses import dataclass
import cv2
import numpy as np

@dataclass
class Feature:
    x: float; y: float; scale: float; orientation: float; response: float
    descriptor: np.ndarray; source_type: str='global'; superpixel_id: int=-1; octave: int=0

@dataclass
class FeatureSet:
    features: list[Feature]
    @property
    def descriptors(self)->np.ndarray:
        return np.asarray([f.descriptor for f in self.features],np.float32) if self.features else np.empty((0,128),np.float32)
    @property
    def points(self)->np.ndarray: return np.asarray([[f.x,f.y] for f in self.features],np.float64)

def _sift(cfg:dict)->cv2.SIFT:
    if not hasattr(cv2,'SIFT_create'): raise RuntimeError('OpenCV build does not provide SIFT')
    return cv2.SIFT_create(nfeatures=int(cfg['nfeatures']),contrastThreshold=float(cfg['contrast_threshold']),edgeThreshold=float(cfg['edge_threshold']),sigma=float(cfg['sigma']))

def _u8(image:np.ndarray)->np.ndarray:
    return np.clip(image*255,0,255).astype(np.uint8) if image.dtype!=np.uint8 else image

def _convert(kps:list, desc:np.ndarray|None, source:str, sid:int)->FeatureSet:
    if desc is None:return FeatureSet([])
    return FeatureSet([Feature(k.pt[0],k.pt[1],k.size,k.angle,k.response,d.copy(),source,sid,k.octave) for k,d in zip(kps,desc)])

def extract_global_sift(image:np.ndarray,config:dict)->FeatureSet:
    """Extract whole-image SIFT features."""
    k,d=_sift(config).detectAndCompute(_u8(image),None); return _convert(k,d,'global',-1)

def extract_region_sift(image:np.ndarray,labels:np.ndarray,config:dict)->FeatureSet:
    """Detect SIFT independently inside each exact super-pixel mask."""
    detector=_sift(config); result=[]; u8=_u8(image)
    for sid in np.unique(labels):
        ys,xs=np.where(labels==sid)
        if not len(xs): continue
        x0,x1,y0,y1=xs.min(),xs.max()+1,ys.min(),ys.max()+1
        roi=u8[y0:y1,x0:x1]; mask=np.where(labels[y0:y1,x0:x1]==sid,255,0).astype(np.uint8)
        kps,desc=detector.detectAndCompute(roi,mask)
        fs=_convert(kps,desc,'region',int(sid))
        for f in fs.features: f.x+=x0; f.y+=y0
        result.extend(fs.features)
    return FeatureSet(result)

def merge_features(global_set:FeatureSet,region_set:FeatureSet,duplicate_radius:float,cosine_threshold:float)->FeatureSet:
    """Merge duplicate detections using configurable spatial/cosine criteria (assumption)."""
    kept=list(global_set.features)
    for f in region_set.features:
        duplicate=-1
        for i,g in enumerate(kept):
            if np.hypot(f.x-g.x,f.y-g.y)<duplicate_radius:
                sim=float(np.dot(f.descriptor,g.descriptor)/(np.linalg.norm(f.descriptor)*np.linalg.norm(g.descriptor)+1e-12))
                if sim>cosine_threshold: duplicate=i; break
        if duplicate<0: kept.append(f)
        elif f.response>kept[duplicate].response: kept[duplicate]=f
    return FeatureSet(kept)

def save_feature_set(path:str,fs:FeatureSet)->None:
    np.savez(path,points=fs.points,descriptors=fs.descriptors,size=[f.scale for f in fs.features],angle=[f.orientation for f in fs.features],response=[f.response for f in fs.features],octave=[f.octave for f in fs.features],source_type=[f.source_type for f in fs.features],superpixel_id=[f.superpixel_id for f in fs.features])
