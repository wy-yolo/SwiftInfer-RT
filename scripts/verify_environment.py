#!/usr/bin/env python3
import json
import os
import platform
import shutil
import subprocess

import onnx
import onnxruntime as ort
import torch
import transformers


def command(args):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()


result = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "compute_capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
    "onnx": onnx.__version__,
    "onnxruntime": ort.__version__,
    "ort_providers": ort.get_available_providers(),
    "transformers": transformers.__version__,
    "nvcc": shutil.which("nvcc"),
    "compute_sanitizer": shutil.which("compute-sanitizer"),
    "cmake": shutil.which("cmake"),
    "ninja": shutil.which("ninja"),
    "tensorrt_root": os.environ.get("TENSORRT_ROOT"),
}
try:
    import tensorrt as trt
    result["tensorrt"] = trt.__version__
except Exception as error:
    result["tensorrt_error"] = repr(error)
if result["nvcc"]:
    result["nvcc_version"] = command([result["nvcc"], "--version"]).splitlines()[-1]
print(json.dumps(result, indent=2, ensure_ascii=False))

required = [
    result["cuda_available"],
    result["compute_capability"] == (12, 0),
    "CUDAExecutionProvider" in result["ort_providers"],
    result["nvcc"] and "minillm-rt" in result["nvcc"],
]
raise SystemExit(0 if all(required) else 1)

