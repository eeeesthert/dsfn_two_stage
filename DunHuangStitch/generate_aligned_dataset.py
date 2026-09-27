import argparse,torch,numpy as np
from torch.utils.data import DataLoader
from datasets.image_pair_dataset import ImagePairDataset
from models.alignment.alignment_net import AlignmentModel
from utils.checkpoint import load_checkpoint
from utils.canvas import compute_union_canvas
from utils.image import save_tensor
def main():
 p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--input_root",required=True);p.add_argument("--output_root",required=True);p.add_argument("--split",default="train");p.add_argument("--device",default="cuda");a=p.parse_args();dev=torch.device(a.device if torch.cuda.is_available() else "cpu");ck=torch.load(a.checkpoint,map_location="cpu",weights_only=False);m=AlignmentModel(ck["config"]).to(dev).eval();load_checkpoint(a.checkpoint,m,map_location=dev)
 for b in DataLoader(ImagePairDataset(a.input_root,a.split,size=ck["config"]["data"]["image_size"]),1):
  r,t=b["reference"].to(dev),b["target"].to(dev);H=m(r,t)["H3"];wr,wt,mr,mt,_=compute_union_canvas(r,t,H);from pathlib import Path;d=Path(a.output_root)/a.split/b["name"][0];d.mkdir(parents=True,exist_ok=True);save_tensor(wr[0],d/"reference.png");save_tensor(wt[0],d/"target.png");save_tensor(mr[0],d/"mask_reference.png");save_tensor(mt[0],d/"mask_target.png");np.save(d/"homography.npy",H[0].detach().cpu().numpy())
if __name__=="__main__":main()
