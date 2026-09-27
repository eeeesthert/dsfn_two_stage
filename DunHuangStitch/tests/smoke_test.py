import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
import torch
from models.alignment.alignment_net import AlignmentModel
from models.fusion.fusion_net import FusionModel
from losses.alignment_loss import AlignmentLoss
from losses.seam_loss import SeamLoss

def main():
 ac={"model":{"feature_channels":[8,12,16],"focus":{"out_channels":8},"hab":{"num_heads":[1,1,1],"window_size":4,"mlp_ratio":2.,"dw_kernel":7},"search_ranges":[2,1,1],"aprb":{"output_channels":8,"attention_reduction":4,"conv_layers":1},"hpb":{"hidden_channels":8,"conv_layers":1},"homography":{"eps":1e-6}}};m=AlignmentModel(ac);opt=torch.optim.Adam(m.parameters(),1e-4)
 for _ in range(2):
  r=torch.rand(1,3,32,32);t=torch.roll(r,1,-1);o=m(r,t);l=AlignmentLoss()(r,o)['loss'];opt.zero_grad();l.backward();opt.step()
 fc={"model":{"channels":[8,12,16,24,32],"difference_mode":"signed","eps":1e-6,"attention":{"heads":4,"window_size":4,"mlp_ratio":2.}},"fusion_loss":{"alpha_edge":1.,"lf_mode":"paper_literal","lambda_b":1.,"lambda_o":1.,"lambda_f":1.}};fm=FusionModel(fc);ma=torch.ones(1,1,32,32);wt=o['warped3'].detach(); vm=o['valid_mask3'].detach();fo=fm(r,wt,ma,vm);loss=SeamLoss(**fc['fusion_loss'])(r,wt,ma,vm,fo)['loss'];loss.backward();assert torch.isfinite(loss);print('smoke pipeline passed',float(loss))
if __name__=='__main__':main()
