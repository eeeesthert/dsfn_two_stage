import argparse,torch
from torch.utils.data import DataLoader
from datasets.aligned_pair_dataset import AlignedPairDataset
from models.fusion.fusion_net import FusionModel
from losses.seam_loss import SeamLoss
from utils.config import config
from utils.checkpoint import save_checkpoint,load_checkpoint
def main():
 p=argparse.ArgumentParser();p.add_argument("--config",default="configs/fusion.yaml");p.add_argument("--resume");p.add_argument("--output",default="checkpoints/fusion");a=p.parse_args();c=config(a.config);dev=torch.device(c["device"] if torch.cuda.is_available() else "cpu");m=FusionModel(c).to(dev);tr=c["training"];opt=torch.optim.AdamW(m.parameters(),lr=tr["lr"],betas=tuple(tr["betas"]),weight_decay=tr["weight_decay"]);sch=torch.optim.lr_scheduler.ExponentialLR(opt,tr["scheduler_gamma"]);start=0;best=1e9
 if a.resume:q=load_checkpoint(a.resume,m,opt,sch,dev);start=q["epoch"]+1;best=q["best_metric"]
 dl=DataLoader(AlignedPairDataset(c["data"]["root"]),tr["batch_size"],shuffle=True,num_workers=tr["workers"]);crit=SeamLoss(**c["fusion_loss"]);sc=torch.amp.GradScaler("cuda",enabled=c["amp"] and dev.type=="cuda")
 for e in range(start,tr["epochs"]):
  total=0
  for b in dl:
   x=[b[k].to(dev) for k in ("I_wr","I_wt","M_wr","M_wt")];opt.zero_grad();out=m(*x);loss=crit(*x,out);sc.scale(loss["loss"]).backward();sc.step(opt);sc.update();total+=loss["loss"].item()
  if e>=tr["warmup_epochs"]:sch.step()
  avg=total/max(len(dl),1);print(e,avg);save_checkpoint(f"{a.output}/last.pt",m,opt,sch,e,best,c)
  if avg<best:best=avg;save_checkpoint(f"{a.output}/best.pt",m,opt,sch,e,best,c)
if __name__=="__main__":main()
