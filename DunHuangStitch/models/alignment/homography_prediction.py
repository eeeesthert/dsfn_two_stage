import torch
from torch import nn
from .hab import HAB
class HomographyPredictionBlock(nn.Module):
 def __init__(self,inc,hidden=64,layers=2,heads=1,window=4):
  super().__init__();mods=[HAB(inc,heads,window)]
  for i in range(layers):mods += [nn.Conv2d(inc if i==0 else hidden,hidden,3,2,1),nn.GELU()]
  self.body=nn.Sequential(*mods,nn.AdaptiveAvgPool2d(1));self.head=nn.Linear(hidden,8);nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
 def forward(self,x):return self.head(self.body(x).flatten(1)).reshape(-1,4,2)
def convert_offset_between_scales(offset,stage_hw,full_hw):
 sh,sw=stage_hw;fh,fw=full_hw;return offset*offset.new_tensor([fw/sw,fh/sh])
