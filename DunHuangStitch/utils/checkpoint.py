from pathlib import Path
import torch
def save_checkpoint(path,model,optimizer,scheduler,epoch,best_metric,config):Path(path).parent.mkdir(parents=True,exist_ok=True);torch.save({"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict(),"scheduler_state_dict":scheduler.state_dict() if scheduler else None,"epoch":epoch,"best_metric":best_metric,"config":config},path)
def load_checkpoint(path,model,optimizer=None,scheduler=None,map_location="cpu"):
 c=torch.load(path,map_location=map_location,weights_only=False);model.load_state_dict(c["model_state_dict"]);optimizer and optimizer.load_state_dict(c["optimizer_state_dict"]);scheduler and c.get("scheduler_state_dict") and scheduler.load_state_dict(c["scheduler_state_dict"]);return c
