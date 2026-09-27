import torch
from torch import nn
class RepConv(nn.Module):
 def __init__(self,inc,outc,stride=1,deploy=False,activation=True):
  super().__init__();self.inc=inc;self.outc=outc;self.stride=stride;self.act=nn.GELU() if activation else nn.Identity();self.deploy=deploy
  if deploy:self.reparam=nn.Conv2d(inc,outc,3,stride,1,bias=True)
  else:self.b3=self._b(inc,outc,3,stride,1);self.b1=self._b(inc,outc,1,stride,0);self.id=nn.BatchNorm2d(inc) if inc==outc and stride==1 else None
 def _b(self,a,b,k,s,p):return nn.Sequential(nn.Conv2d(a,b,k,s,p,bias=False),nn.BatchNorm2d(b))
 def forward(self,x):return self.act(self.reparam(x) if self.deploy else self.b3(x)+self.b1(x)+(self.id(x) if self.id else 0))
 def _fuse(self,b):
  if b is None:return 0,0
  if isinstance(b,nn.BatchNorm2d):
   k=torch.zeros(self.outc,self.inc,3,3,device=b.weight.device);k[range(self.outc),range(self.inc),1,1]=1;bn=b
  else:k=b[0].weight;bn=b[1];k=torch.nn.functional.pad(k,[1,1,1,1]) if k.shape[-1]==1 else k
  std=(bn.running_var+bn.eps).sqrt();return k*(bn.weight/std).reshape(-1,1,1,1),bn.bias-bn.running_mean*bn.weight/std
 def switch_to_deploy(self):
  if self.deploy:return
  ks,bs=zip(*(self._fuse(x) for x in (self.b3,self.b1,self.id)));self.reparam=nn.Conv2d(self.inc,self.outc,3,self.stride,1,bias=True).to(self.b3[0].weight.device);self.reparam.weight.data=sum(ks);self.reparam.bias.data=sum(bs);del self.b3,self.b1,self.id;self.deploy=True
