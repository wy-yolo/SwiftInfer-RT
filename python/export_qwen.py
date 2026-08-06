#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path
from typing import Iterable, Sequence

import onnx
import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM


def legacy_cache(cache) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    if hasattr(cache, "to_legacy_cache"):
        cache = cache.to_legacy_cache()
    return tuple((layer[0], layer[1]) for layer in cache)


def build_dynamic_cache(pairs: Sequence[tuple[torch.Tensor, torch.Tensor]], config):
    try:
        from transformers.cache_utils import DynamicCache
        return DynamicCache(ddp_cache_data=tuple(pairs), config=config)
    except Exception:
        pass
    return tuple(pairs)


class PrefillWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, position_ids):
        output = self.model(
            input_ids=input_ids.to(torch.long),
            attention_mask=attention_mask.to(torch.long),
            position_ids=position_ids.to(torch.long),
            use_cache=True,
            return_dict=True,
        )
        flattened: list[torch.Tensor] = [output.logits[:, -1, :]]
        for key, value in legacy_cache(output.past_key_values):
            flattened.extend((key, value))
        return tuple(flattened)


class DecodeWrapper(nn.Module):
    def __init__(self, model: nn.Module, num_layers: int):
        super().__init__()
        self.model = model
        self.num_layers = num_layers

    def forward(self, input_ids, attention_mask, position_ids, *past):
        pairs = [(past[2 * i], past[2 * i + 1]) for i in range(self.num_layers)]
        output = self.model(
            input_ids=input_ids.to(torch.long),
            attention_mask=attention_mask.to(torch.long),
            position_ids=position_ids.to(torch.long),
            past_key_values=build_dynamic_cache(pairs, self.model.config),
            use_cache=True,
            return_dict=True,
        )
        flattened: list[torch.Tensor] = [output.logits[:, -1, :]]
        for key, value in legacy_cache(output.past_key_values):
            flattened.extend((key[..., -1:, :], value[..., -1:, :]))
        return tuple(flattened)


def output_names(num_layers: int) -> list[str]:
    names = ["logits"]
    for layer in range(num_layers):
        names.extend((f"present_key_{layer}", f"present_value_{layer}"))
    return names


def externalize(path: Path) -> None:
    unloaded = onnx.load(path, load_external_data=False)
    original_external_files = {
        entry.value
        for initializer in unloaded.graph.initializer
        for entry in initializer.external_data
        if entry.key == "location"
    }
    model = onnx.load(path, load_external_data=True)
    data_name = path.name + ".data"
    data_path = path.parent / data_name
    if data_path.exists():
        data_path.unlink()
    onnx.save_model(
        model,
        path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_name,
        size_threshold=1024,
        convert_attribute=False,
    )
    onnx.checker.check_model(str(path))
    for location in original_external_files:
        if location != data_name:
            (path.parent / location).unlink(missing_ok=True)


def require_cuda_device(device_arg: str) -> torch.device:
    device = torch.device(device_arg)
    if device.type != "cuda":
        raise SystemExit(
            f"ONNX export is GPU-only; received --device={device_arg!r}"
        )
    if not torch.cuda.is_available():
        raise SystemExit("ONNX export requires CUDA, but torch.cuda.is_available() is false")
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index < 0 or index >= torch.cuda.device_count():
        raise SystemExit(
            f"CUDA device index {index} is invalid; found {torch.cuda.device_count()} device(s)"
        )
    torch.cuda.set_device(index)
    return torch.device("cuda", index)


def gpu_free_mib(device_index: int) -> int:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
            f"--id={device_index}",
        ],
        text=True,
    )
    return int(output.strip().splitlines()[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("artifacts/model"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--dummy-sequence", type=int, default=8)
    parser.add_argument("--dummy-past", type=int, default=8)
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="CUDA device used for tracing; CPU export is intentionally unsupported",
    )
    parser.add_argument("--min-free-gb", type=float, default=16.0)
    args = parser.parse_args()
    if args.output is None:
        args.output = Path("artifacts/onnx") / args.precision
    args.output.mkdir(parents=True, exist_ok=True)
    export_device = require_cuda_device(args.device)
    free_mib = gpu_free_mib(export_device.index)
    required_mib = int(args.min_free_gb * 1024)
    if free_mib < required_mib:
        raise SystemExit(
            f"GPU export gate: {free_mib} MiB free, {required_mib} MiB required"
        )

    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    model_dtype = torch.float16 if args.precision == "fp16" else torch.float32
    load_kwargs = {"local_files_only": True, "attn_implementation": "eager"}
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=model_dtype, **load_kwargs
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=model_dtype, **load_kwargs
        )
    model.eval().to(export_device)
    if any(parameter.device != export_device for parameter in model.parameters()):
        raise RuntimeError("not all model parameters were transferred to the export GPU")
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    layers = int(config.num_hidden_layers)
    kv_heads = int(config.num_key_value_heads)
    head_dim = int(config.hidden_size // config.num_attention_heads)
    names = output_names(layers)

    seq = args.dummy_sequence
    input_ids = torch.ones((1, seq), dtype=torch.int32, device=export_device)
    attention_mask = torch.ones((1, seq), dtype=torch.int32, device=export_device)
    position_ids = torch.arange(seq, dtype=torch.int32, device=export_device).unsqueeze(0)
    prefill_path = args.output / "prefill.onnx"
    prefill_axes = {
        "input_ids": {1: "sequence"},
        "attention_mask": {1: "sequence"},
        "position_ids": {1: "sequence"},
        "logits": {0: "batch"},
    }
    for layer in range(layers):
        prefill_axes[f"present_key_{layer}"] = {0: "batch", 2: "sequence"}
        prefill_axes[f"present_value_{layer}"] = {0: "batch", 2: "sequence"}
    torch.onnx.export(
        PrefillWrapper(model),
        (input_ids, attention_mask, position_ids),
        str(prefill_path),
        input_names=["input_ids", "attention_mask", "position_ids"],
        output_names=names,
        dynamic_axes=prefill_axes,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    externalize(prefill_path)

    past_len = args.dummy_past
    token = torch.ones((1, 1), dtype=torch.int32, device=export_device)
    decode_mask = torch.ones((1, past_len + 1), dtype=torch.int32, device=export_device)
    decode_position = torch.tensor([[past_len]], dtype=torch.int32, device=export_device)
    past = []
    for _ in range(layers):
        shape = (1, kv_heads, past_len, head_dim)
        past.extend((torch.zeros(shape, dtype=model_dtype, device=export_device),
                     torch.zeros(shape, dtype=model_dtype, device=export_device)))
    past_names = []
    for layer in range(layers):
        past_names.extend((f"past_key_{layer}", f"past_value_{layer}"))
    decode_axes = {
        "input_ids": {0: "batch"},
        "attention_mask": {0: "batch", 1: "total_sequence"},
        "position_ids": {0: "batch"},
        "logits": {0: "batch"},
    }
    for layer in range(layers):
        decode_axes[f"past_key_{layer}"] = {0: "batch", 2: "past_sequence"}
        decode_axes[f"past_value_{layer}"] = {0: "batch", 2: "past_sequence"}
        decode_axes[f"present_key_{layer}"] = {0: "batch"}
        decode_axes[f"present_value_{layer}"] = {0: "batch"}
    decode_path = args.output / "decode.onnx"
    torch.onnx.export(
        DecodeWrapper(model, layers),
        (token, decode_mask, decode_position, *past),
        str(decode_path),
        input_names=["input_ids", "attention_mask", "position_ids", *past_names],
        output_names=names,
        dynamic_axes=decode_axes,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    externalize(decode_path)

    metadata = {
        "model_type": config.model_type,
        "num_layers": layers,
        "hidden_size": int(config.hidden_size),
        "num_attention_heads": int(config.num_attention_heads),
        "num_kv_heads": kv_heads,
        "head_dim": head_dim,
        "vocab_size": int(config.vocab_size),
        "precision": args.precision,
        "torch_dtype": str(model_dtype),
        "opset": 17,
        "export_device": str(export_device),
        "export_gpu": torch.cuda.get_device_name(export_device),
        "torch_cuda_version": torch.version.cuda,
    }
    (args.output / "model_spec.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
