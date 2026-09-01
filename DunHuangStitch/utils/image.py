from pathlib import Path
import numpy as np,torch
from PIL import Image
def save_tensor(x,path):Path(path).parent.mkdir(parents=True,exist_ok=True);a=(x.detach().cpu().clamp(0,1).permute(1,2,0).numpy()*255).astype(np.uint8);Image.fromarray(a.squeeze()).save(path)
