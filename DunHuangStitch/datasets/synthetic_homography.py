import random,torch
from torch.utils.data import Dataset
from .image_pair_dataset import load_image
from models.alignment.dlt import image_corners,dlt_homography
from models.alignment.warper import warp_image
class SyntheticHomographyDataset(Dataset):
 def __init__(self,paths,size=(128,128),perturb_range=32):self.paths=paths;self.size=size;self.rho=perturb_range
 def __len__(self):return len(self.paths)
 def __getitem__(self,i):
  a=load_image(self.paths[i],self.size);h,w=self.size;src=image_corners(1,h,w,a.device,a.dtype);off=(torch.rand_like(src)*2-1)*self.rho;H=dlt_homography(src,src+off);b=warp_image(a[None],torch.linalg.inv(H))[0];return {"reference":a,"target":b,"H_gt":H[0],"offset_gt":off[0]}
