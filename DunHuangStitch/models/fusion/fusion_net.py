import torch
from torch import nn
from .shallow_feature_extractor import ShallowFeatureExtractor
from .reconstruction_unet import ReconstructionUNet
from .seam_generator import SeamGenerator
def weighted_fusion(a,b,ma,mb,s,eps=1e-6):
 wr=s*ma;wt=(1-s)*mb;den=wr+wt;out=(wr*a+wt*b)/(den+eps);out=torch.where((ma>0)&(mb<=0),a,out);out=torch.where((mb>0)&(ma<=0),b,out);return out*(den>0).to(out.dtype)
class FusionModel(nn.Module):
 def __init__(self,cfg):super().__init__();m=cfg["model"];self.mode=m["difference_mode"];self.eps=m["eps"];self.encoder=ShallowFeatureExtractor(m["channels"][0]);self.recon=ReconstructionUNet(m["channels"],m["attention"]);self.seam=SeamGenerator(m["channels"][0])
 def forward(self,a,b,ma,mb):
  fa=self.encoder(a);fb=self.encoder(b);d=fa-fb if self.mode=="signed" else (fa-fb).abs();sf=self.recon(d);raw=self.seam(sf,a.shape[-2:]);sr=raw*ma;st=(1-raw)*mb;stitched=weighted_fusion(a,b,ma,mb,raw,self.eps);return {"feature_wr":fa,"feature_wt":fb,"feature_diff":d,"seam_feature":sf,"seam_mask_r_raw":raw,"seam_mask_r":sr,"seam_mask_t":st,"stitched":stitched,"M_wr":ma,"M_wt":mb}
