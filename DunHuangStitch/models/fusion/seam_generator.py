from torch import nn
from torch.nn import functional as F
from .repconv import RepConv
class SeamGenerator(nn.Module):
 def __init__(self,c):super().__init__();self.r1=RepConv(c,c);self.r2=RepConv(c,c);self.head=nn.Conv2d(c,1,1)
 def forward(self,x,size):x=F.interpolate(x,scale_factor=2,mode="bilinear",align_corners=False);x=self.r1(x);x=F.interpolate(x,size=size,mode="bilinear",align_corners=False);return self.head(self.r2(x)).sigmoid()
