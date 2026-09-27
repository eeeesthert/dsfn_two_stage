import torch
from torch import nn
from .focus import Focus
from .hab import HAB
from .patch_merging import PatchMerging
from .aprb import APRB
from .homography_prediction import HomographyPredictionBlock,convert_offset_between_scales
from .dlt import dlt_homography,image_corners
from .warper import warp_image,warp_mask
class FeatureExtractor(nn.Module):
 def __init__(self,channels,hab):
  super().__init__(); c0,c1,c2=channels; hs=hab["num_heads"];kw=dict(window=hab["window_size"],mlp_ratio=hab["mlp_ratio"],dw_kernel=hab["dw_kernel"],residual=hab.get("residual",True));self.focus=Focus(3,c0);self.h0=nn.Sequential(HAB(c0,hs[0],**kw),HAB(c0,hs[0],**kw));self.m1=PatchMerging(c0,c1);self.h1=nn.Sequential(HAB(c1,hs[1],**kw),HAB(c1,hs[1],**kw));self.m2=PatchMerging(c1,c2);self.h2=nn.Sequential(HAB(c2,hs[2],**kw),HAB(c2,hs[2],**kw));self.m3=PatchMerging(c2,c2)
 def forward(self,x): f=self.h0(self.focus(x));fine=self.m1(f);mid=self.m2(self.h1(fine));coarse=self.m3(self.h2(mid));return [coarse,mid,fine]
class AlignmentModel(nn.Module):
 def __init__(self,cfg):
  super().__init__();m=cfg["model"];self.extractor=FeatureExtractor(m["feature_channels"],m["hab"]);a=m["aprb"];h=m["hpb"];self.aprbs=nn.ModuleList([APRB(r,a["output_channels"],a["attention_reduction"],a.get("conv_layers",2)) for r in m["search_ranges"]]);self.hpbs=nn.ModuleList([HomographyPredictionBlock(a["output_channels"],h["hidden_channels"],h.get("conv_layers",2)) for _ in range(3)]);self.eps=m["homography"].get("eps",1e-6)
 def forward(self,ref,tgt):
  fr=self.extractor(ref);ft=self.extractor(tgt);b,_,h,w=ref.shape;src=image_corners(b,h,w,ref.device,ref.dtype);cum=torch.zeros_like(src);out={"features_ref":fr,"features_tgt":ft};prev=None
  for i,(a,p) in enumerate(zip(self.aprbs,self.hpbs),1):
   tf=ft[i-1] if prev is None else warp_image(ft[i-1],prev,fr[i-1].shape[-2:]); d=p(a(fr[i-1],tf));df=convert_offset_between_scales(d,fr[i-1].shape[-2:],(h,w));cum=cum+df;H=dlt_homography(src,src+cum,self.eps);out[f"delta{i}"]=d;out[f"offset{i}"]=cum;out[f"H{i}"]=H;out[f"warped{i}"]=warp_image(tgt,H);out[f"valid_mask{i}"]=warp_mask(torch.ones(b,1,h,w,device=ref.device,dtype=ref.dtype),H);prev=H
  out["aligned_reference"]=ref;out["aligned_target"]=out["warped3"];return out
