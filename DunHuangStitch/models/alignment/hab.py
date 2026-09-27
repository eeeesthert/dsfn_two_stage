import math,torch
from torch import nn
from torch.nn import functional as F
class WindowAttention(nn.Module):
 def __init__(self,c,heads=4,window=8,mlp_ratio=2.): super().__init__(); self.w=window;self.a=nn.MultiheadAttention(c,heads,batch_first=True);self.m=nn.Sequential(nn.Linear(c,int(c*mlp_ratio)),nn.GELU(),nn.Linear(int(c*mlp_ratio),c))
 def forward(self,x):
  b,c,h,w=x.shape; ph=(-h)%self.w;pw=(-w)%self.w;y=F.pad(x,(0,pw,0,ph));hh,ww=y.shape[-2:]; y=y.permute(0,2,3,1).reshape(b,hh//self.w,self.w,ww//self.w,self.w,c).permute(0,1,3,2,4,5).reshape(-1,self.w*self.w,c); y=self.a(y,y,y,need_weights=False)[0];y=y+self.m(y);y=y.reshape(b,hh//self.w,ww//self.w,self.w,self.w,c).permute(0,1,3,2,4,5).reshape(b,hh,ww,c).permute(0,3,1,2);return y[:,:,:h,:w]
class HAB(nn.Module):
 def __init__(self,c,heads=4,window=8,mlp_ratio=2.,dw_kernel=7,residual=True): super().__init__();self.local=nn.Sequential(nn.Conv2d(c,c,dw_kernel,padding=dw_kernel//2,groups=c),nn.GELU());self.attn=WindowAttention(c,heads,window,mlp_ratio);self.proj=nn.Conv2d(c,c,1);self.residual=residual
 def forward(self,x):
  y=self.proj(self.local(x)+self.attn(x));return x+y if self.residual else y
