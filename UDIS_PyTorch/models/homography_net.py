"""UDIS three-level coarse-to-fine four-corner homography regressor."""
import torch
from torch import nn
from torch.nn import functional as F
from .common import init_glorot
from .cost_volume import CostVolume
from .dlt import DifferentiableDLT
from .feature_pyramid import FeaturePyramid
from .homography_warp import homography_warp
class Regressor(nn.Module):
    def __init__(self, channels, strides, spatial, hidden, dropout):
        super().__init__(); layers=[]
        for ci,co,st in zip(channels[:-1],channels[1:],strides): layers += [nn.Conv2d(ci,co,3,stride=st,padding=1),nn.ReLU(inplace=True)]
        self.conv=nn.Sequential(*layers); out_sp=spatial
        for s in strides: out_sp=(out_sp+2-3)//s+1
        self.fc=nn.Sequential(nn.Linear(channels[-1]*out_sp*out_sp,hidden),nn.ReLU(inplace=True),nn.Dropout(dropout),nn.Linear(hidden,8)); init_glorot(self)
    def forward(self,x): return self.fc(self.conv(x).flatten(1))
class HomographyNet(nn.Module):
    """Return three residual corner offsets and their sum in 128 coordinates."""
    def __init__(self, dropout=.5, search_ranges=(16,8,4)):
        super().__init__(); r1,r2,r3=search_ranges; self.features=FeaturePyramid(); self.dlt=DifferentiableDLT()
        self.cv1,self.cv2,self.cv3=CostVolume(r1),CostVolume(r2),CostVolume(r3)
        self.reg1=Regressor([self.cv1.channels,512,512,512],[1,1,1],16,1024,dropout)
        self.reg2=Regressor([self.cv2.channels,256,256,256],[1,1,2],32,512,dropout)
        self.reg3=Regressor([self.cv3.channels,128,128,128],[1,2,2],64,256,dropout)
    def forward(self,image1,image2):
        a=F.interpolate(image1,size=(128,128),mode='bilinear',align_corners=False).mean(1,keepdim=True)
        b=F.interpolate(image2,size=(128,128),mode='bilinear',align_corners=False).mean(1,keepdim=True)
        _,a2,a3,a4=self.features(a); _,b2,b3,b4=self.features(b)
        d1=self.reg1(self.cv1(a4,b4)); h1=self.dlt(d1/4,32); b3w=homography_warp(b3,h1)
        d2=self.reg2(self.cv2(a3,b3w)); d12=d1+d2; h2=self.dlt(d12/2,64); b2w=homography_warp(b2,h2)
        d3=self.reg3(self.cv3(a2,b2w))
        return {'delta1':d1,'delta2':d2,'delta3':d3,'delta_final':d12+d3}
