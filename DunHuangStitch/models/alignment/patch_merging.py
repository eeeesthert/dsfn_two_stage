from torch import nn
class PatchMerging(nn.Module):
 def __init__(self,inc,outc): super().__init__();self.op=nn.Sequential(nn.Conv2d(inc,outc,3,2,1),nn.GELU())
 def forward(self,x): return self.op(x)
