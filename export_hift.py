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
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, load_vllm=False, fp16=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

hift = cosyvoice.model.hift 


# first = True 
# mel_len =  50

# first = False 
# mel_len =  58

first = eval(sys.argv[1])
mel_len = int(sys.argv[2])

mel = torch.ones([1,80,mel_len], dtype=torch.float32).to(device)
if not first:
    hift_cache_source = torch.ones([1,1,3840],dtype=torch.float32).to(device)
else:
    hift_cache_source = torch.zeros(1, 1, 0).to(device)

inputs = (mel,)
input_names = ["mel"]
output_names = ["s"]
if not first:
    onnx_output_p1 = f"hift_p1_{mel_len}.onnx"
    onnx_output_p2 = f"hift_p2_{mel_len}.onnx"
else:
    onnx_output_p1 = f"hift_p1_{mel_len}_first.onnx"
    onnx_output_p2 = f"hift_p2_{mel_len}_first.onnx"


hift.forward = hift.inference_part1

export_onnx(hift, inputs, input_names, output_names, onnx_output_p1)


s = torch.ones(1,1, 480*mel_len, dtype=torch.float32).to(device)

inputs = (mel, s, hift_cache_source)
input_names = ["mel", "s", "hift_cache_source"]
output_names = ["audio"]

hift.forward = hift.inference_part2
export_onnx(hift, inputs, input_names, output_names, onnx_output_p2)