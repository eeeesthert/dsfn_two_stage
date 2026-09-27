"""Intermediate-result writers for alignment and fusion debugging."""
from pathlib import Path
from .image import save_tensor
def save_alignment_debug(ref,tgt,out,root):
 root=Path(root); pairs={'reference':ref,'target':tgt,**{f'stage{i}_warped':out[f'warped{i}'] for i in range(1,4)}}
 pairs['final_overlap_difference']=(ref-out['warped3']).abs()*out['valid_mask3']
 for n,x in pairs.items():save_tensor(x[0],root/f'{n}.png')
def save_fusion_debug(a,b,out,root):
 root=Path(root);items={'I_wr':a,'I_wt':b,'seam_mask':out['seam_mask_r'],'inverse_seam':out['seam_mask_t'],'stitched':out['stitched']}
 for n,x in items.items():save_tensor(x[0],root/f'{n}.png')
