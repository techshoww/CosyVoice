import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import torch
import numpy as np 
import os
import sys 

cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, load_vllm=False, fp16=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


llm_embedding = cosyvoice.model.llm.llm_embedding.weight.cpu().detach().numpy()
speech_embedding = cosyvoice.model.llm.speech_embedding.weight.cpu().detach().numpy()
flow_input_embedding = cosyvoice.model.flow.input_embedding.weight.cpu().detach().numpy()

print("llm_embedding",llm_embedding.shape)
print("speech_embedding",speech_embedding.shape)

np.save("llm.llm_embedding.npy", llm_embedding)
np.save("llm.speech_embedding.npy", speech_embedding)
np.save("flow.input_embedding.npy", flow_input_embedding)

with open("llm.llm_embedding.float32.bin", "wb") as f:
    f.write(llm_embedding.astype(np.float32).tobytes())

with open("llm.speech_embedding.float32.bin", "wb") as f:
    f.write(speech_embedding.astype(np.float32).tobytes())

with open("flow.input_embedding.float32.bin", "wb") as f:
    f.write(flow_input_embedding.astype(np.float32).tobytes())

os.system("./tools/fp32_to_bf16 llm.llm_embedding.float32.bin llm.llm_embedding.float16.bin")
os.system("./tools/fp32_to_bf16 llm.speech_embedding.float32.bin llm.speech_embedding.float16.bin")
os.system("./tools/fp32_to_bf16 flow.input_embedding.float32.bin flow.input_embedding.float16.bin")