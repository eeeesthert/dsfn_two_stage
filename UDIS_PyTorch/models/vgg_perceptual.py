"""Frozen ImageNet VGG19 feature extractor (conv3_3=14, conv5_3=32)."""
import torch
from torch import nn
class VGGPerceptualExtractor(nn.Module):
    def __init__(self,pretrained=True):
        super().__init__()
        from torchvision.models import VGG19_Weights,vgg19
        self.features=vgg19(weights=VGG19_Weights.IMAGENET1K_V1 if pretrained else None).features[:33]
        for p in self.parameters(): p.requires_grad=False
        self.register_buffer('mean',torch.tensor([.485,.456,.406]).view(1,3,1,1)); self.register_buffer('std',torch.tensor([.229,.224,.225]).view(1,3,1,1)); self.eval()
    def train(self,mode=True): super().train(False); return self
    def forward(self,x):
        z=((x+1)/2-self.mean)/self.std; out={}
        for i,layer in enumerate(self.features):
            z=layer(z)
            if i==14: out['conv3_3']=z
            if i==32: out['conv5_3']=z
        return out
