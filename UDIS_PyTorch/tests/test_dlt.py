import torch
from UDIS_PyTorch.models.dlt import DifferentiableDLT,transform_points
def test_identity(): assert torch.allclose(DifferentiableDLT()(torch.zeros(1,8),128),torch.eye(3)[None],atol=1e-4)
def test_translation():
 d=torch.tensor([[5.,-3.]*4]);H=DifferentiableDLT()(d,128);p=torch.tensor([[[0.,0.],[128.,128.]]]);assert torch.allclose(transform_points(H,p),p+torch.tensor([5.,-3.]),atol=1e-4)
def test_gradient():
 d=torch.zeros(1,8,requires_grad=True);DifferentiableDLT()(d,128).sum().backward();assert d.grad is not None and torch.isfinite(d.grad).all()
