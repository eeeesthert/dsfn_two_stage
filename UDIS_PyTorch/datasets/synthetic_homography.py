"""Synthetic no-parallax pretraining pairs; GT offsets are debug-only."""
import random
from pathlib import Path
import torch
from torch.utils.data import Dataset
from UDIS_PyTorch.utils.image import load_rgb,resize_image
from UDIS_PyTorch.models.dlt import DifferentiableDLT
from UDIS_PyTorch.models.homography_warp import homography_warp
class SyntheticHomographyDataset(Dataset):
    def __init__(self,root,patch_size=128,perturbation=16): self.files=sorted(Path(root).glob('**/*')); self.files=[p for p in self.files if p.suffix.lower() in ('.jpg','.jpeg','.png')]; self.size=patch_size; self.rho=perturbation; self.dlt=DifferentiableDLT()
    def __len__(self): return len(self.files)
    def __getitem__(self,i):
        image=load_rgb(self.files[i]); h,w=image.shape[-2:]
        if min(h,w)<self.size: image=resize_image(image[None],(max(h,self.size),max(w,self.size)))[0]; h,w=image.shape[-2:]
        y=random.randint(0,h-self.size); x=random.randint(0,w-self.size); a=image[:,y:y+self.size,x:x+self.size]; d=torch.empty(1,8).uniform_(-self.rho,self.rho); H=self.dlt(d,self.size); b=homography_warp(a[None],torch.linalg.inv(H))[0]
        return {'image1':a,'image2':b,'aug1':a,'aug2':b,'gt_offsets':d[0],'name':self.files[i].stem}
