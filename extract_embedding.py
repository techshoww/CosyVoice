import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice3
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import torch
import numpy as np 
import os
import sys 
os.environ["export_onnx"] = "True"
cosyvoice = CosyVoice3('../Fun-CosyVoice3-0.5B-2512', load_trt=False, load_vllm=False, fp16=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


speech_embedding = cosyvoice.model.llm.speech_embedding.weight.cpu().detach().numpy()
flow_input_embedding = cosyvoice.model.flow.input_embedding.weight.cpu().detach().numpy()

print("speech_embedding",speech_embedding.shape)

np.save("llm.speech_embedding.npy", speech_embedding)
np.save("flow.input_embedding.npy", flow_input_embedding)


with open("llm.speech_embedding.float32.bin", "wb") as f:
    f.write(speech_embedding.astype(np.float32).tobytes())

with open("flow.input_embedding.float32.bin", "wb") as f:
    f.write(flow_input_embedding.astype(np.float32).tobytes())

os.system("./tools/fp32_to_bf16 llm.speech_embedding.float32.bin llm.speech_embedding.float16.bin")
os.system("./tools/fp32_to_bf16 flow.input_embedding.float32.bin flow.input_embedding.float16.bin")