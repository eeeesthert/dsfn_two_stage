import torch
def rmse(a,b,m=None):
 d=(a-b).square();d=d*m if m is not None else d;return torch.sqrt(d.sum()/((m.sum()*a.shape[1] if m is not None else d.numel())+1e-8))
def psnr(a,b,m=None):return -20*torch.log10(rmse(a,b,m).clamp_min(1e-8))
def ssim(a,b):
 ux=a.mean();uy=b.mean();vx=a.var();vy=b.var();cov=((a-ux)*(b-uy)).mean();return ((2*ux*uy+.01**2)*(2*cov+.03**2))/((ux**2+uy**2+.01**2)*(vx+vy+.03**2))
