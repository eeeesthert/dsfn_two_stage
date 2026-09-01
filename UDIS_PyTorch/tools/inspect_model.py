"""Print architecture shapes and parameter counts without retaining gradients."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[2]))
import torch
from UDIS_PyTorch.models.cost_volume import CostVolume
from UDIS_PyTorch.models.feature_pyramid import FeaturePyramid
from UDIS_PyTorch.models.homography_net import HomographyNet
from UDIS_PyTorch.models.reconstruction_net import ReconstructionNet
with torch.no_grad():
 p=FeaturePyramid();f=p(torch.randn(1,1,128,128));print('Feature Pyramid:',[tuple(x.shape) for x in f]);print('Cost Volumes:',[(r,(1,CostVolume(r).channels,s,s)) for r,s in ((16,16),(8,32),(4,64))]);h=HomographyNet();r=ReconstructionNet();print('HomographyNet parameters:',sum(x.numel() for x in h.parameters()));print('Reconstruction parameters:',sum(x.numel() for x in r.parameters()));del h
 x=torch.randn(1,3,64,64);o=r(x,x);print('Reconstruction:',{k:tuple(v.shape) for k,v in o.items()})
print('Full Homography forward is covered by tests/test_homography_net.py when sufficient RAM is available.')
