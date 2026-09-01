"""Central image conversion and resize helpers."""
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

def resize_image(x: torch.Tensor, size): return F.interpolate(x,size=size,mode='bilinear',align_corners=False)
def load_rgb(path):
    a=np.asarray(Image.open(path).convert('RGB'),dtype=np.float32); return torch.from_numpy(a).permute(2,0,1)/127.5-1
def load_mask(path):
    a=np.asarray(Image.open(path).convert('RGB'),dtype=np.float32); return torch.from_numpy(a).permute(2,0,1)/255.0
def save_image(x,path,mask=False):
    x=x.detach().cpu().clamp(0,1) if mask else ((x.detach().cpu().clamp(-1,1)+1)*.5)
    Image.fromarray((x.permute(1,2,0).numpy()*255).round().astype('uint8')).save(path)
