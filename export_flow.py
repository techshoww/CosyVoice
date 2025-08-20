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

    onnx_model = onnx.load(onnx_output)
    print("IR 版本:", onnx_model.ir_version)
    print("操作集:", onnx_model.opset_import)
    onnx_model = infer_shapes(onnx_model)
    # convert model
    model_simp, check = onnxsim.simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, onnx_output)
    print("onnx simpilfy successed, and model saved in {}".format(onnx_output))

os.environ["export_onnx"] = "True"
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, load_vllm=False, fp16=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

flow = cosyvoice.model.flow 


speech_token_len = int(sys.argv[1])                                 # 28 53 78 78 78 ... 50
prompt_token_len = 75

token_embedding = torch.ones([1,speech_token_len+prompt_token_len, 512], dtype=torch.float32)                          
prompt_feat = torch.ones([1,150,80],dtype=torch.float32)
embedding = torch.ones([1,192], dtype=torch.float32)
finalize = eval(sys.argv[2])
print("finalize",finalize)


token_embedding = token_embedding.to(device)
prompt_feat = prompt_feat.to(device)
embedding = embedding.to(device)

inputs = (token_embedding,  prompt_feat,  embedding)
input_names = ["token_embedding",  "prompt_feat",  "embedding"]
output_names = ["mel"]
if not finalize:
    flow.forward = flow.inference_export
    onnx_output = f"flow_{speech_token_len}.onnx"
else:
    flow.forward = flow.inference_export_final
    onnx_output = f"flow_{speech_token_len}_final.onnx"

export_onnx(flow, inputs, input_names, output_names, onnx_output)