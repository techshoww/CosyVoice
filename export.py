import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import torch
import onnx
from onnx.shape_inference import infer_shapes
import onnxsim
import onnx
from onnx import helper
import numpy as np 
import os
import sys 

def export_onnx(model, input, input_names, output_names, onnx_output):

    torch.onnx.export(
        model,
        input,
        onnx_output,
        input_names=input_names,
        output_names=output_names,
        opset_version=16,
    )

    # onnx_model = onnx.load(onnx_output)
    # print("IR 版本:", onnx_model.ir_version)
    # print("操作集:", onnx_model.opset_import)
    # onnx_model = infer_shapes(onnx_model)
    # # convert model
    # model_simp, check = onnxsim.simplify(onnx_model)
    # assert check, "Simplified ONNX model could not be validated"
    # onnx.save(model_simp, onnx_output)
    # print("onnx simpilfy successed, and model saved in {}".format(onnx_output))


cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, load_vllm=False, fp16=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

flow = cosyvoice.model.flow 
flow.forward = flow.inference_export

token = torch.ones([1,103], dtype=torch.int32)
token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(device)

prompt_token = torch.ones([1,75], dtype=torch.int32)
prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(device)


prompt_feat = torch.ones([1,150,80],dtype=torch.float32)
prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(device)

embedding = torch.ones([1,192], dtype=torch.float32)

finalize = False


token = token.to(device)
prompt_token = prompt_token.to(device)
prompt_feat = prompt_feat.to(device)
embedding = embedding.to(device)

inputs = (token, token_len, prompt_token, prompt_token_len, prompt_feat, prompt_feat_len, embedding, finalize)
input_names = ["token", "token_len", "prompt_token", "prompt_token_len", "prompt_feat", "prompt_feat_len", "embedding", "finalize"]
output_names = ["mel"]
onnx_output = "flow_103.onnx"

export_onnx(flow, inputs, input_names, output_names, onnx_output)