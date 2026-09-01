"""Dataset reading frozen Stage-1 warp/mask quadruples."""
from pathlib import Path
from torch.utils.data import Dataset
from UDIS_PyTorch.utils.image import load_mask,load_rgb,resize_image
class ReconstructionDataset(Dataset):
    def __init__(self,root,max_image_size=1024): self.root=Path(root); self.files=sorted((self.root/'warp1').glob('*')); self.maximum=max_image_size
    def __len__(self): return len(self.files)
    def __getitem__(self,i):
        n=self.files[i].name; out={'warp1':load_rgb(self.root/'warp1'/n),'warp2':load_rgb(self.root/'warp2'/n),'mask1':load_mask(self.root/'mask1'/n),'mask2':load_mask(self.root/'mask2'/n),'name':Path(n).stem}; h,w=out['warp1'].shape[-2:]
        if self.maximum and max(h,w)>self.maximum:
            scale=self.maximum/max(h,w); size=(int(h*scale)//8*8,int(w*scale)//8*8)
            for k in ('warp1','warp2','mask1','mask2'): out[k]=resize_image(out[k][None],size)[0]
        return out
