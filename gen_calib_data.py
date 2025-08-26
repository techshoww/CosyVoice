import os
import torch
import numpy as np
from glob import glob

paths = glob("*.pth")
for p in paths:
    if "llm_decoder_input" in p:
        continue
    name = os.path.splitext(p)[0]
    a = torch.load(p)
    np.save(name+".npy", a.detach().cpu().numpy())
    
    os.system(f"tar -cvf {name}.tar {name}.npy")


paths = glob("llm_decoder_input_*.pth")
for p in paths:
    name = os.path.splitext(p)[0]
    a = torch.load(p)
    np.save(name+".npy", a.detach().cpu().numpy())

    os.system(f"tar -cvf llm_decoder_input.tar llm_decoder_input_*.npy")