import torch
from UDIS_PyTorch.models.reconstruction_net import ReconstructionNet
def test_shapes_range_backward():
 m=ReconstructionNet(lr_size=32,num_res_blocks=1);a=torch.randn(1,3,16,24,requires_grad=True);o=m(a,a);assert o['lr'].shape==(1,3,32,32) and o['hr'].shape==(1,3,16,24);assert o['lr'].abs().max()<=1 and o['hr'].abs().max()<=1;(o['lr'].mean()+o['hr'].mean()).backward();assert a.grad is not None
