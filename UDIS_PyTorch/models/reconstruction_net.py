"""UDIS low-resolution deformation and high-resolution refinement branches."""
import torch
from torch import nn
from torch.nn import functional as F
from .common import init_glorot
from UDIS_PyTorch.utils.image import resize_image

def cb(a,b): return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.ReLU(inplace=True),nn.Conv2d(b,b,3,padding=1),nn.ReLU(inplace=True))
class ResidualBlock(nn.Module):
    def __init__(self): super().__init__(); self.c1=nn.Conv2d(64,64,3,padding=1); self.c2=nn.Conv2d(64,64,3,padding=1)
    def forward(self,x): return F.relu(x+self.c2(F.relu(self.c1(x),inplace=True)),inplace=True)
class ReconstructionNet(nn.Module):
    def __init__(self,lr_size=256,num_res_blocks=8):
        super().__init__(); self.lr_size=lr_size; self.e1=cb(6,64); self.e2=cb(64,128); self.e3=cb(128,256); self.e4=cb(256,512); self.pool=nn.MaxPool2d(2)
        self.u3=nn.ConvTranspose2d(512,256,2,2); self.d3=cb(512,256); self.u2=nn.ConvTranspose2d(256,128,2,2); self.d2=cb(256,128); self.u1=nn.ConvTranspose2d(128,64,2,2); self.d1=cb(128,64); self.lr_out=nn.Conv2d(64,3,3,padding=1)
        self.hr_in=nn.Conv2d(9,64,3,padding=1); self.res=nn.Sequential(*[ResidualBlock() for _ in range(num_res_blocks)]); self.hr_global=nn.Conv2d(64,64,3,padding=1); self.hr_out=nn.Conv2d(64,3,3,padding=1); init_glorot(self)
    def forward(self,warp1,warp2):
        x=resize_image(torch.cat((warp1,warp2),1),(self.lr_size,self.lr_size)); s1=self.e1(x); s2=self.e2(self.pool(s1)); s3=self.e3(self.pool(s2)); z=self.e4(self.pool(s3)); z=self.d3(torch.cat((self.u3(z),s3),1)); z=self.d2(torch.cat((self.u2(z),s2),1)); z=self.d1(torch.cat((self.u1(z),s1),1)); lr=torch.tanh(self.lr_out(z))
        up=resize_image(lr,warp1.shape[-2:]); f0=F.relu(self.hr_in(torch.cat((warp1,warp2,up),1)),inplace=True); z=F.relu(f0+self.hr_global(self.res(f0)),inplace=True); hr=torch.tanh(self.hr_out(z)); return {'lr':lr,'hr':hr}
