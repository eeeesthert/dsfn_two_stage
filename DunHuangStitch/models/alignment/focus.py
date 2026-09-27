import torch
from torch import nn
class Focus(nn.Module):
 def __init__(self,inc,outc): super().__init__(); self.proj=nn.Sequential(nn.Conv2d(inc*4,outc,3,padding=1),nn.GELU())
 def forward(self,x):
  if x.shape[-2]%2 or x.shape[-1]%2: x=torch.nn.functional.pad(x,(0,x.shape[-1]%2,0,x.shape[-2]%2))
  return self.proj(torch.cat((x[:,:,0::2,0::2],x[:,:,1::2,0::2],x[:,:,0::2,1::2],x[:,:,1::2,1::2]),1))
