import torch
from UDIS_PyTorch.models.homography_warp import homography_warp
def test_forward_translation_direction():
 x=torch.zeros(1,1,7,7);x[0,0,2,2]=1;H=torch.tensor([[[1.,0,2],[0,1.,1],[0,0,1.]]]);y=homography_warp(x,H);assert y[0,0,3,4]>.99
