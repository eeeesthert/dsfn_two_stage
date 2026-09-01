import argparse,torch
from datasets.image_pair_dataset import load_image
from models.alignment.alignment_net import AlignmentModel
from models.fusion.fusion_net import FusionModel
from utils.checkpoint import load_checkpoint
from utils.canvas import compute_union_canvas
from utils.image import save_tensor
def main():
 p=argparse.ArgumentParser();p.add_argument("--reference",required=True);p.add_argument("--target",required=True);p.add_argument("--alignment_checkpoint",required=True);p.add_argument("--fusion_checkpoint",required=True);p.add_argument("--output",required=True);p.add_argument("--device",default="cuda");a=p.parse_args();dev=torch.device(a.device if torch.cuda.is_available() else "cpu");ac=torch.load(a.alignment_checkpoint,map_location="cpu",weights_only=False);fc=torch.load(a.fusion_checkpoint,map_location="cpu",weights_only=False);am=AlignmentModel(ac["config"]).to(dev).eval();fm=FusionModel(fc["config"]).to(dev).eval();load_checkpoint(a.alignment_checkpoint,am,map_location=dev);load_checkpoint(a.fusion_checkpoint,fm,map_location=dev);r=load_image(a.reference)[None].to(dev);t=load_image(a.target)[None].to(dev)
 with torch.no_grad():H=am(r,t)["H3"];wr,wt,mr,mt,_=compute_union_canvas(r,t,H);o=fm(wr,wt,mr,mt)
 from pathlib import Path;d=Path(a.output).parent;save_tensor(o["stitched"][0],a.output)
 for n,x in (("warped_reference",wr),("warped_target",wt),("mask_reference",mr),("mask_target",mt),("seam_reference",o["seam_mask_r"]),("seam_target",o["seam_mask_t"])):save_tensor(x[0],d/f"{n}.png")
if __name__=="__main__":main()
