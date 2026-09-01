"""Cross-boundary seam, VGG content, and LR/HR consistency objectives."""
import torch
from torch import nn
from torch.nn import functional as F
from UDIS_PyTorch.utils.image import resize_image

def boundary(mask):
    m=mask[:,:1].clamp(0,1); e=torch.zeros_like(m); e[:,:,1:]+=abs(m[:,:,1:]-m[:,:,:-1]); e[:,:,:,1:]+=abs(m[:,:,:,1:]-m[:,:,:,:-1]); k=torch.ones(1,1,3,3,device=m.device,dtype=m.dtype)
    for _ in range(3): e=F.conv2d(e,k,padding=1).clamp(0,1)
    return e

def seam_masks(m1,m2): return boundary(m2)*m1[:,:1],boundary(m1)*m2[:,:1]
class ReconstructionLoss(nn.Module):
    def __init__(self,vgg,seam_weight=2.,content_weight=1e-6,lr_weight=100.,hr_weight=1.,consistency_weight=1.,lr_size=256,vgg_size=224): super().__init__(); self.vgg=vgg; self.sw=seam_weight; self.cw=content_weight; self.lw=lr_weight; self.hw=hr_weight; self.kw=consistency_weight; self.lrs=lr_size; self.vs=vgg_size
    def _content(self,s,w,m,layer): return F.mse_loss(self.vgg(resize_image(s*m,(self.vs,self.vs)))[layer],self.vgg(resize_image(w*m,(self.vs,self.vs)))[layer])
    def forward(self,out,w1,w2,m1,m2):
        sm1,sm2=seam_masks(m1,m2); w1l,w2l=resize_image(w1,(self.lrs,self.lrs)),resize_image(w2,(self.lrs,self.lrs)); m1l,m2l=resize_image(m1[:,:1],(self.lrs,self.lrs)),resize_image(m2[:,:1],(self.lrs,self.lrs)); s1l,s2l=seam_masks(m1l,m2l)
        lrs=F.l1_loss(out['lr']*s1l,w1l*s1l)+F.l1_loss(out['lr']*s2l,w2l*s2l); lrc=self._content(out['lr'],w1l,m1l,'conv5_3')+self._content(out['lr'],w2l,m2l,'conv5_3'); hrs=F.l1_loss(out['hr']*sm1,w1*sm1)+F.l1_loss(out['hr']*sm2,w2*sm2); hrc=self._content(out['hr'],w1,m1[:,:1],'conv3_3')+self._content(out['hr'],w2,m2[:,:1],'conv3_3'); ll=self.sw*lrs+self.cw*lrc; hl=self.sw*hrs+self.cw*hrc; consistency=F.l1_loss(resize_image(out['hr'],(self.lrs,self.lrs)),out['lr']); total=self.lw*ll+self.hw*hl+self.kw*consistency
        return {'total_loss':total,'lr_loss':ll,'hr_loss':hl,'consistency_loss':consistency,'lr_seam_loss':lrs,'lr_content_loss':lrc,'hr_seam_loss':hrs,'hr_content_loss':hrc,'seam1':sm1,'seam2':sm2}
