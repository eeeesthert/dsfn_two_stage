"""Manifest-based real-pair dataset with independent UDIS photometric jitter."""
import csv, random
import torch
from torch.utils.data import Dataset
from UDIS_PyTorch.utils.image import load_rgb
class AlignmentDataset(Dataset):
    def __init__(self,manifest,brightness=(.7,1.3),color=(.7,1.3),augment=True):
        with open(manifest,newline='',encoding='utf8') as f: self.pairs=[tuple(r[:2]) for r in csv.reader(f) if r and r[0]!='image1']
        self.brightness,self.color,self.augment=brightness,color,augment
    def __len__(self): return len(self.pairs)
    def _jitter(self,x):
        if not self.augment:return x
        gain=x.new_tensor([random.uniform(*self.color) for _ in range(3)])[:,None,None]; return (x*random.uniform(*self.brightness)*gain).clamp(-1,1)
    def __getitem__(self,i):
        a,b=map(load_rgb,self.pairs[i]); return {'image1':a,'image2':b,'aug1':self._jitter(a),'aug2':self._jitter(b),'name':str(i)}
