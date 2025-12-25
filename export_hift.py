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

hift = cosyvoice.model.hift 




final = eval(sys.argv[1])
mel_len = int(sys.argv[2])

mel = torch.ones([1,80,mel_len], dtype=torch.float32).to(device)


inputs = (mel, )
input_names = ["mel"]
output_names = ["s"]
if not final:
    onnx_output_p1 = f"hift_p1_{mel_len}.onnx"
    onnx_output_p2 = f"hift_p2_{mel_len}.onnx"
else:
    onnx_output_p1 = f"hift_p1_{mel_len}_final.onnx"
    onnx_output_p2 = f"hift_p2_{mel_len}_final.onnx"


if not final:
    hift.forward = hift.inference_part1
else:
    hift.forward = hift.inference_part1_final

export_onnx(hift, inputs, input_names, output_names, onnx_output_p1)


if not final:
    hift.forward = hift.inference_part2
    s = torch.ones(1,1, 480*(mel_len-3), dtype=torch.float32).to(device)
else:
    hift.forward = hift.inference_part2_final
    s = torch.ones(1,1, 480*mel_len, dtype=torch.float32).to(device)


inputs = (mel, s)
input_names = ["mel", "s"]
output_names = ["audio"]


export_onnx(hift, inputs, input_names, output_names, onnx_output_p2)