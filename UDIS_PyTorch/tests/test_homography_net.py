import os,pytest,torch
from UDIS_PyTorch.models.homography_net import HomographyNet
@pytest.mark.skipif(os.environ.get('UDIS_FULL_MODEL_TEST')!='1',reason='exact FC layers require about 1 GB RAM; set UDIS_FULL_MODEL_TEST=1')
def test_outputs():
 m=HomographyNet().eval();o=m(torch.randn(1,3,128,128),torch.randn(1,3,128,128));assert all(o[k].shape==(1,8) for k in o);assert torch.allclose(o['delta_final'],o['delta1']+o['delta2']+o['delta3'])
