from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
def load_image(p,size=None):
 im=Image.open(p).convert("RGB");im=im.resize((size[1],size[0])) if size else im;return torch.from_numpy(np.asarray(im).copy()).permute(2,0,1).float()/255
class ImagePairDataset(Dataset):
 def __init__(self,root,split="train",size=(128,128),pair_list=None):
  self.root=Path(root)/split;self.size=size
  if pair_list:self.pairs=[tuple((self.root/x for x in line.split())) for line in Path(pair_list).read_text().splitlines() if line.strip()]
  else:self.pairs=[(d/"reference.png",d/"target.png") for d in sorted(self.root.iterdir()) if d.is_dir()]
 def __len__(self):return len(self.pairs)
 def __getitem__(self,i):a,b=self.pairs[i];return {"reference":load_image(a,self.size),"target":load_image(b,self.size),"name":a.parent.name}
