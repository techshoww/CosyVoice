import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice3
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
        opset_version=17,
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

cosyvoice = CosyVoice3('../Fun-CosyVoice3-0.5B-2512', load_trt=False, load_vllm=False, fp16=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = cosyvoice.model.llm.llm_decoder


x = torch.ones([1, 896], dtype=torch.float32).to(device)


inputs = (x,)
input_names = ["x"]
output_names = ["y"]

onnx_output = f"llm_decoder.onnx"

export_onnx(model, inputs, input_names, output_names, onnx_output)


# norm = cosyvoice.model.llm.llm.model.model.norm

# onnx_output = f"postnorm.onnx"

# export_onnx(norm, inputs, input_names, output_names, onnx_output)