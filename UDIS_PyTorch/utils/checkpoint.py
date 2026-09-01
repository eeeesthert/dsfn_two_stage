"""Portable training checkpoint helpers."""
import torch
def save_checkpoint(path,model,optimizer,step,config,scheduler=None): torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict() if scheduler else None,'step':step,'config':config},path)
def load_checkpoint(path,model,optimizer=None,scheduler=None,map_location='cpu'):
    state=torch.load(path,map_location=map_location); model.load_state_dict(state['model']);
    if optimizer and state.get('optimizer'): optimizer.load_state_dict(state['optimizer'])
    if scheduler and state.get('scheduler'): scheduler.load_state_dict(state['scheduler'])
    return state
