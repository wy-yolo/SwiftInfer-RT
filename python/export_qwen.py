#!/usr/bin/env python3
import argparse
import json
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("artifacts/model"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/onnx"))
    parser.add_argument("--dummy-sequence", type=int, default=8)
    parser.add_argument("--dummy-past", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    load_kwargs = {"local_files_only": True, "attn_implementation": "eager"}
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float16, **load_kwargs
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, **load_kwargs
        )
    model.eval().to(args.device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    layers = int(config.num_hidden_layers)
    kv_heads = int(config.num_key_value_heads)
    head_dim = int(config.hidden_size // config.num_attention_heads)
    names = output_names(layers)

    seq = args.dummy_sequence
    input_ids = torch.ones((1, seq), dtype=torch.int32, device=args.device)
    attention_mask = torch.ones((1, seq), dtype=torch.int32, device=args.device)
    position_ids = torch.arange(seq, dtype=torch.int32, device=args.device).unsqueeze(0)
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
        prefill_path,
        input_names=["input_ids", "attention_mask", "position_ids"],
        output_names=names,
        dynamic_axes=prefill_axes,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    externalize(prefill_path)

    past_len = args.dummy_past
    token = torch.ones((1, 1), dtype=torch.int32, device=args.device)
    decode_mask = torch.ones((1, past_len + 1), dtype=torch.int32, device=args.device)
    decode_position = torch.tensor([[past_len]], dtype=torch.int32, device=args.device)
    past = []
    for _ in range(layers):
        shape = (1, kv_heads, past_len, head_dim)
        past.extend((torch.zeros(shape, dtype=torch.float16, device=args.device),
                     torch.zeros(shape, dtype=torch.float16, device=args.device)))
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
        decode_path,
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
        "opset": 17,
    }
    (args.output / "model_spec.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
