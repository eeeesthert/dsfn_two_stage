"""Documented local homography approximation to under-specified paper refinement."""
from dataclasses import dataclass
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse.linalg import lsqr
from .gabor_features import sample_feature_map
from .homography import transform_points
from .saliency import saliency_variance
@dataclass(frozen=True)
class RefinementResult:
 H:np.ndarray; accepted:bool; Eg_before:float; Eg_after:float; Es_before:float; Es_after:float; message:str

def _pack(H): return np.array([H[0,0],H[0,1],H[0,2],H[1,0],H[1,1],H[1,2],H[2,0],H[2,1]])
def _unpack(x): return np.array([[x[0],x[1],x[2]],[x[3],x[4],x[5]],[x[6],x[7],1.]])
def transform_quality(H,shape,cfg):
 h,w=shape[:2]; corners=np.array([[0,0],[w,0],[w,h],[0,h]],float); q=transform_points(corners,H); area=abs(cv2.contourArea(q.astype(np.float32)))/(w*h)
 det=np.linalg.det(H[:2,:2]); ok=np.isfinite(H).all() and abs(np.linalg.det(H))>1e-10 and det>0 and cfg['min_area_ratio']<=area<=cfg['max_area_ratio'] and max(abs(H[2,0]),abs(H[2,1]))<=cfg.get('max_perspective',np.inf)
 return ok,{'determinant':float(np.linalg.det(H)),'area_ratio':float(area),'scale_x':float(np.linalg.norm(H[:2,0])),'scale_y':float(np.linalg.norm(H[:2,1])),'rotation_deg':float(np.degrees(np.arctan2(H[1,0],H[0,0]))),'shear':float(np.dot(H[:2,0],H[:2,1])),'perspective':float(np.linalg.norm(H[2,:2]))}
def refine_homography(H0,ref_energy,target_energy,ref_points,target_points,sal_ref,sal_target,cfg,check_cfg,target_shape,use_gabor=True,use_saliency=True):
 base=_pack(H0); ref_tex=sample_feature_map(ref_energy,ref_points); sg=saliency_variance(sal_ref); st=saliency_variance(sal_target)
 def components(x):
  H=_unpack(x); projected=transform_points(target_points,H); rg=(ref_tex-sample_feature_map(target_energy,target_points)).ravel() if use_gabor else np.empty(0)
  # Spatial correspondence modulated by paper Eq.(10) global set spread.
  rs=((ref_points-projected)/(max(sg,st,1.0))).ravel() if use_saliency else np.empty(0)
  if cfg.get('normalize_residuals',True):
   if len(rg): rg=rg/(np.sqrt(np.mean(rg*rg))+1e-6)
   if len(rs): rs=rs/(np.sqrt(np.mean(rs*rs))+1e-6)
  return rg,rs
 def fun(x):
  rg,rs=components(x); return np.r_[np.sqrt(cfg['lambda_g'])*rg,np.sqrt(cfg['lambda_s'])*rs]
 rg0,rs0=components(base); eg0=float(rg0@rg0); es0=float(rs0@rs0)
 if cfg.get('backend','nonlinear')=='sparse_linear':
  eps=1e-6; r0=fun(base); J=np.column_stack([(fun(base+np.eye(8)[j]*eps)-r0)/eps for j in range(8)]); delta=lsqr(J,-r0,iter_lim=int(cfg['max_nfev']))[0]; candidate=_unpack(base+delta); msg='sparse linearized LSQR'
 else:
  result=least_squares(fun,base,method='trf',max_nfev=int(cfg['max_nfev']),ftol=float(cfg['ftol']),xtol=float(cfg['xtol']),gtol=float(cfg['gtol'])); candidate=_unpack(result.x); msg=result.message
 ok,_=transform_quality(candidate,target_shape,check_cfg); H=candidate if ok else H0; rg1,rs1=components(_pack(H)); return RefinementResult(H,ok,eg0,float(rg1@rg1),es0,float(rs1@rs1),str(msg))
