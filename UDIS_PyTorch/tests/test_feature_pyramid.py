import torch
from UDIS_PyTorch.models.feature_pyramid import FeaturePyramid
def test_shapes():
 o=FeaturePyramid()(torch.randn(1,1,128,128));assert [x.shape for x in o]==[(1,64,128,128),(1,64,64,64),(1,128,32,32),(1,128,16,16)]
