import torch
from models.alignment.dlt import dlt_homography,transform_points
from models.alignment.warper import warp_image
from models.alignment.cost_volume import cost_volume
from models.fusion.repconv import RepConv
from models.fusion.fusion_net import FusionModel,weighted_fusion
from losses.seam_loss import SeamLoss

def test_dlt_identity_translation_perspective():
 p=torch.tensor([[[0.,0.],[31,0],[31,31],[0,31]]],requires_grad=True)
 for H in [torch.eye(3)[None],torch.tensor([[[1.,0,4],[0,1,3],[0,0,1]]]),torch.tensor([[[1.,.1,2],[.05,1,3],[.001,.002,1]]])]:
  q=transform_points(H,p);e=dlt_homography(p,q);assert torch.allclose(transform_points(e,p),q,atol=2e-3)
 e.sum().backward();assert p.grad is not None

def test_warp_identity_gradient():
 x=torch.rand(1,1,8,8,requires_grad=True);y=warp_image(x,torch.eye(3)[None]);assert torch.allclose(x,y,atol=1e-6);y.sum().backward();assert x.grad is not None

def test_cost_volume_shape_backward():
 a=torch.rand(1,3,5,6,requires_grad=True);v=cost_volume(a,a,2);assert v.shape==(1,25,5,6);v.sum().backward()

def test_repconv_deploy():
 m=RepConv(4,4).eval();x=torch.rand(2,4,8,8);a=m(x);m.switch_to_deploy();assert (a-m(x)).abs().max()<1e-4

def test_fusion_and_loss():
 c={"model":{"channels":[8,12,16,24,32],"difference_mode":"signed","eps":1e-6,"attention":{"heads":4,"window_size":4,"mlp_ratio":2.}},"fusion_loss":{"alpha_edge":1.,"lf_mode":"paper_literal","lambda_b":1.,"lambda_o":1.,"lambda_f":1.}};m=FusionModel(c);a=torch.rand(1,3,32,32);b=torch.rand_like(a);ma=torch.ones(1,1,32,32);mb=torch.ones_like(ma);o=m(a,b,ma,mb);assert o['stitched'].shape==a.shape;assert o['seam_mask_r_raw'].min()>=0 and o['seam_mask_r_raw'].max()<=1;l=SeamLoss(**c['fusion_loss'])(a,b,ma,mb,o);assert torch.isfinite(l['loss']);l['loss'].backward()

def test_weighted_regions():
 a=torch.ones(1,1,1,4);b=2*a;ma=torch.tensor([[[[1.,0,1,0]]]]);mb=torch.tensor([[[[0.,1,1,0]]]]);o=weighted_fusion(a,b,ma,mb,torch.full_like(ma,.25));assert torch.allclose(o.flatten(),torch.tensor([1.,2.,1.75,0.]))
