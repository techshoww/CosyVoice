import os
import torch
import numpy as np
from glob import glob

# paths = glob("*.pth")
paths = ["speech_feat_50.pth",  "speech_feat_58.pth", "hift_cache_source.pth"]
for p in paths:
    name = os.path.splitext(p)[0]
    a = torch.load(p)
    # name1 = "_".join(name.split("_")[0:-1])
    np.save(name+".npy", a.detach().cpu().numpy())
    
    os.system(f"tar -cvf {name}.tar {name}.npy")