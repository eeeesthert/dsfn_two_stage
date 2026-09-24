from pathlib import Path
from torch.utils.data import Dataset
from .image_pair_dataset import load_image
class AlignedPairDataset(Dataset):
 def __init__(self,root,split="train",size=None):self.items=[d for d in sorted((Path(root)/split).iterdir()) if d.is_dir()];self.size=size
 def __len__(self):return len(self.items)
 def __getitem__(self,i):d=self.items[i];return {"I_wr":load_image(d/"reference.png",self.size),"I_wt":load_image(d/"target.png",self.size),"M_wr":load_image(d/"mask_reference.png",self.size)[:1],"M_wt":load_image(d/"mask_target.png",self.size)[:1],"name":d.name}
