"""Export the one-step distilled CosyVoice2 flow estimator to ONNX.

The distilled checkpoint only contains ``flow.decoder.estimator`` weights under
the ``student`` key.  The rest of the CosyVoice2 model is still loaded from the
original model directory so the estimator architecture and deployment I/O stay
identical to the original AX model.
"""

import argparse
import os
import sys
from pathlib import Path

import onnx
import onnxsim
import torch
from onnx.shape_inference import infer_shapes


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR / "third_party" / "Matcha-TTS"))

from cosyvoice.cli.cosyvoice import CosyVoice2


DEFAULT_MODEL_DIR = SCRIPT_DIR / "pretrained_models" / "CosyVoice2-0.5B"
DEFAULT_STUDENT_CHECKPOINT = Path(
    "/data/tmp/yongqiang/nfs/lhj/huggingface/cosyvoice2_rknn/original/model/"
    "offline1_v3_student/checkpoint-final.pt"
)


class StreamingEstimatorExportWrapper(torch.nn.Module):
    """Freeze the estimator's non-tensor ``streaming`` argument for ONNX."""

    def __init__(self, estimator: torch.nn.Module):
        super().__init__()
        self.estimator = estimator

    def forward(self, x, mask, mu, t, spks, cond):
        # Keep this equal to the legacy exporter, which passed True as the
        # estimator's final argument.  It is intentionally not an ONNX input.
        return self.estimator(x, mask, mu, t, spks, cond, streaming=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export the one-step distilled CosyVoice2 flow estimator to ONNX."
    )
    parser.add_argument("x_len", type=int, help="Static mel-frame length, e.g. 200, 250, or 300")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"Original CosyVoice2-0.5B model directory (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--student-checkpoint",
        type=Path,
        default=DEFAULT_STUDENT_CHECKPOINT,
        help=f"Distilled estimator checkpoint (default: {DEFAULT_STUDENT_CHECKPOINT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="ONNX output path (default: flow_estimator_<x_len>.onnx in the current directory)",
    )
    return parser.parse_args()


def load_distilled_estimator(estimator: torch.nn.Module, checkpoint_path: Path) -> int:
    """Load and validate the estimator-only state dict in a distillation checkpoint."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Distilled checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "student" not in checkpoint:
        raise KeyError(
            "Expected a distillation checkpoint containing an estimator state dict under 'student'."
        )
    if not isinstance(checkpoint["student"], dict):
        raise TypeError("checkpoint['student'] must be an estimator state dict")

    student_steps = checkpoint.get("student_steps")
    if student_steps != 1:
        raise ValueError(
            f"Expected a one-step distilled checkpoint (student_steps=1), got {student_steps!r}."
        )

    # The checkpoint was trained directly from this estimator architecture, so
    # strict loading prevents silently exporting an incomplete/wrong checkpoint.
    estimator.load_state_dict(checkpoint["student"], strict=True)
    print(
        "Loaded distilled estimator: "
        f"{checkpoint_path} (student_steps={student_steps}, "
        f"guidance={checkpoint.get('student_guidance_mode', 'unknown')})"
    )
    return student_steps


def export_onnx(model, inputs, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        inputs,
        str(output_path),
        input_names=["x", "mask", "mu", "t", "spks", "cond"],
        output_names=["y"],
        opset_version=16,
    )

    onnx_model = onnx.load(str(output_path))
    print("IR version:", onnx_model.ir_version)
    print("Opset:", onnx_model.opset_import)
    onnx_model = infer_shapes(onnx_model)
    simplified_model, check = onnxsim.simplify(onnx_model)
    if not check:
        raise RuntimeError("Simplified ONNX model could not be validated")
    onnx.save(simplified_model, str(output_path))
    print(f"ONNX simplification succeeded: {output_path}")


def main():
    args = parse_args()
    if args.x_len <= 0:
        raise ValueError(f"x_len must be positive, got {args.x_len}")
    if not args.model_dir.is_dir():
        raise FileNotFoundError(f"Original model directory not found: {args.model_dir}")

    os.environ["export_onnx"] = "True"
    cosyvoice = CosyVoice2(
        str(args.model_dir), load_jit=False, load_trt=False, load_vllm=False, fp16=False
    )
    estimator = cosyvoice.model.flow.decoder.estimator
    student_steps = load_distilled_estimator(estimator, args.student_checkpoint)
    estimator.eval()

    device = cosyvoice.model.device
    x = torch.ones((2, 80, args.x_len), dtype=torch.float32, device=device)
    mask = torch.ones((2, 1, args.x_len), dtype=torch.float32, device=device)
    mu = torch.ones((2, 80, args.x_len), dtype=torch.float32, device=device)
    t = torch.ones((2,), dtype=torch.float32, device=device)
    spks = torch.ones((2, 80), dtype=torch.float32, device=device)
    cond = torch.ones((2, 80, args.x_len), dtype=torch.float32, device=device)

    output_path = args.output or Path.cwd() / f"flow_estimator_1_step_{args.x_len}.onnx"
    with torch.inference_mode():
        export_onnx(
            StreamingEstimatorExportWrapper(estimator),
            (x, mask, mu, t, spks, cond),
            output_path,
        )
    print(f"Exported one-step distilled flow estimator (student_steps={student_steps})")


if __name__ == "__main__":
    main()
