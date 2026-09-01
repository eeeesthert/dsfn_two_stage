import argparse,torch
from torch.utils.data import DataLoader
from datasets.image_pair_dataset import ImagePairDataset
from models.alignment.alignment_net import AlignmentModel
from losses.alignment_loss import AlignmentLoss
from utils.config import config
from utils.checkpoint import save_checkpoint,load_checkpoint
from utils.seed import set_seed
def main():
 p=argparse.ArgumentParser();p.add_argument("--config",default="configs/alignment.yaml");p.add_argument("--resume");p.add_argument("--output",default="checkpoints/alignment");a=p.parse_args();c=config(a.config);set_seed(c["seed"]);dev=torch.device(c["device"] if torch.cuda.is_available() else "cpu");model=AlignmentModel(c).to(dev);opt=torch.optim.AdamW(model.parameters(),lr=c["training"]["lr"],betas=tuple(c["training"]["betas"]),weight_decay=c["training"]["weight_decay"]);sch=torch.optim.lr_scheduler.ExponentialLR(opt,c["training"]["scheduler_gamma"]);start=0;best=1e9
 if a.resume:q=load_checkpoint(a.resume,model,opt,sch,dev);start=q["epoch"]+1;best=q["best_metric"]
 ds=ImagePairDataset(c["data"]["root"],size=c["data"]["image_size"]);dl=DataLoader(ds,c["training"]["batch_size"],shuffle=True,num_workers=c["training"]["workers"]);crit=AlignmentLoss(c["loss"]["stage_weights"]);scaler=torch.amp.GradScaler("cuda",enabled=c["amp"] and dev.type=="cuda")
 for epoch in range(start,c["training"]["epochs"]):
  total=0.
  for b in dl:
   r,t=b["reference"].to(dev),b["target"].to(dev);opt.zero_grad();
   with torch.autocast(dev.type,enabled=scaler.is_enabled()):out=model(r,t);losses=crit(r,out)
   scaler.scale(losses["loss"]).backward();scaler.step(opt);scaler.update();total+=losses["loss"].item()
  if epoch>=c["training"]["warmup_epochs"]:sch.step()
  avg=total/max(len(dl),1);print(epoch,opt.param_groups[0]["lr"],avg);save_checkpoint(f"{a.output}/last.pt",model,opt,sch,epoch,best,c)
  if avg<best:best=avg;save_checkpoint(f"{a.output}/best.pt",model,opt,sch,epoch,best,c)
if __name__=="__main__":main()
