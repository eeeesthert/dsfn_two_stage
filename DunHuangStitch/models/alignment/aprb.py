from torch import nn
from .cost_volume import cost_volume
class APRB(nn.Module):
 def __init__(self,radius,outc=49,reduction=4,layers=2):
  super().__init__();self.radius=radius;d=(2*radius+1)**2;hid=max(d//reduction,1);self.att=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(d,hid,1),nn.GELU(),nn.Conv2d(hid,d,1),nn.Sigmoid());mods=[]
  for i in range(layers): mods += [nn.Conv2d(d if i==0 else outc,outc,3,padding=1),nn.GELU()]
  self.compress=nn.Sequential(*mods)
 def forward(self,a,b):
  v=cost_volume(a,b,self.radius);return self.compress(v*self.att(v))
