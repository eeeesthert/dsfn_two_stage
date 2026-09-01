import pytest,torch
from UDIS_PyTorch.models.stitching_transform import StitchingDomainTransformer
@pytest.mark.parametrize('tx,ty',[(0,0),(12,4),(-12,-4)])
def test_union(tx,ty):
 x=torch.ones(1,3,24,32);H=torch.tensor([[[1.,0,tx],[0,1.,ty],[0,0,1.]]]);o=StitchingDomainTransformer()(x,x,H);h,w=o['warp1'].shape[-2:];assert h%8==0 and w%8==0 and w>=32+abs(tx);assert 0<=o['mask1'].min()<=o['mask1'].max()<=1
