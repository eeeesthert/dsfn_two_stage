import torch
from torch import nn
from torch.nn import functional as F
from .repconv import RepConv
from .feature_perception import FeaturePerception
class ReconstructionUNet(nn.Module):
 def __init__(self,chs,att):
  super().__init__();self.down=nn.ModuleList([RepConv(chs[i],chs[i+1],2) for i in range(4)]);self.bot=FeaturePerception(chs[-1],att["heads"],att["window_size"],att["mlp_ratio"]);self.up=nn.ModuleList([RepConv(chs[i+1],chs[i]) for i in range(3,-1,-1)])
 def forward(self,x):
  skips=[]
  for d in self.down:skips.append(x);x=d(x)
  x=self.bot(x)
  for u,s in zip(self.up,reversed(skips)):x=F.interpolate(x,size=s.shape[-2:],mode="bilinear",align_corners=False);x=u(x)+s
  return x
