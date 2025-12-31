import os
import torch
import numpy as np
from glob import glob

paths = glob("*.pth")

calibs = []

for p in paths:
    if "llm_decoder_input" in p:
        continue
    name = os.path.splitext(p)[0]
    calibs.append(name.split("_time")[0])
    a = torch.load(p)
    np.save(name+".npy", a.detach().cpu().numpy())
    
calib_set = set(calibs)
for name in calib_set:
    cnt = calibs.count(name)
    os.system(f"tar -cvf {name}_cnt{cnt}.tar {name}*.npy")

for name in calib_set:
    os.system(f"rm {name}*.npy {name}*.pth")

paths = glob("llm_decoder_input_*.pth")
for p in paths:
    name = os.path.splitext(p)[0]
    a = torch.load(p)
    np.save(name+".npy", a.detach().cpu().numpy())

os.system(f"tar -cvf llm_decoder_input.tar llm_decoder_input_*.npy")
os.system("rm llm_decoder_input_*.npy llm_decoder_input_*.pth")