import torch
from torch import nn
class AlignmentLoss(nn.Module):
 def __init__(self,weights=(1.,4.,16.),eps=1e-6):super().__init__();self.weights=weights;self.eps=eps
 def forward(self,ref,out):
  parts={};total=ref.new_zeros(())
  for i,w in enumerate(self.weights,1):
   m=out[f"valid_mask{i}"];v=(torch.abs(ref-out[f"warped{i}"])*m).sum()/(m.sum()*ref.shape[1]+self.eps);parts[f"stage{i}_loss"]=v;total=total+w*v
  parts["loss"]=total;return parts
