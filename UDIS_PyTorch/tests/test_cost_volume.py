import pytest,torch
from UDIS_PyTorch.models.cost_volume import CostVolume
@pytest.mark.parametrize('r,c',[(16,1089),(8,289),(4,81)])
def test_shapes_and_backward(r,c):
 a=torch.randn(1,2,3,4,requires_grad=True);b=torch.randn_like(a,requires_grad=True);o=CostVolume(r)(a,b);assert o.shape==(1,c,3,4);o.sum().backward();assert a.grad is not None and b.grad is not None
