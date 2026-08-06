#!/usr/bin/env python3
"""Export an analysis-only Qwen prefill graph with every hidden state exposed."""

import argparse
import json
from pathlib import Path

import onnx
import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM


class HiddenStatePrefill(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, position_ids):
        output = self.model(
            input_ids=input_ids.to(torch.long),
            attention_mask=attention_mask.to(torch.long),
            position_ids=position_ids.to(torch.long),
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        return (output.logits[:, -1, :], *output.hidden_states)


class LayerDetailPrefill(nn.Module):
    CAPTURE_NAMES = (
        "input_layernorm",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "self_attn",
        "post_attention_layernorm",
        "gate_proj",
        "up_proj",
        "down_proj",
        "mlp",
        "layer_output",
    )

    def __init__(self, model: nn.Module, layer_index: int):
        super().__init__()
        self.model = model
        self.captured = {}
        layer = model.model.layers[layer_index]
        modules = {
            "input_layernorm": layer.input_layernorm,
            "q_proj": layer.self_attn.q_proj,
            "k_proj": layer.self_attn.k_proj,
            "v_proj": layer.self_attn.v_proj,
            "o_proj": layer.self_attn.o_proj,
            "self_attn": layer.self_attn,
            "post_attention_layernorm": layer.post_attention_layernorm,
            "gate_proj": layer.mlp.gate_proj,
            "up_proj": layer.mlp.up_proj,
            "down_proj": layer.mlp.down_proj,
            "mlp": layer.mlp,
            "layer_output": layer,
        }
        for name, module in modules.items():
            module.register_forward_hook(self._hook(name))

    def _hook(self, name):
        def save(_module, _inputs, output):
            self.captured[name] = output[0] if isinstance(output, (tuple, list)) else output

        return save

    def forward(self, input_ids, attention_mask, position_ids):
        self.captured.clear()
        output = self.model(
            input_ids=input_ids.to(torch.long),
            attention_mask=attention_mask.to(torch.long),
            position_ids=position_ids.to(torch.long),
            use_cache=False,
            return_dict=True,
        )
        return (output.logits[:, -1, :], *(self.captured[name] for name in self.CAPTURE_NAMES))


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
    parser.add_argument("--output", type=Path, default=Path("artifacts/diagnostics"))
    parser.add_argument("--dummy-sequence", type=int, default=8)
    parser.add_argument("--detail-layer", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        attn_implementation="eager",
        local_files_only=True,
    ).eval().cuda()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    sequence = args.dummy_sequence
    input_ids = torch.ones((1, sequence), dtype=torch.int32, device="cuda")
    attention_mask = torch.ones((1, sequence), dtype=torch.int32, device="cuda")
    position_ids = torch.arange(sequence, dtype=torch.int32, device="cuda").unsqueeze(0)
    hidden_count = int(config.num_hidden_layers) + 1
    if args.detail_layer is None:
        wrapper = HiddenStatePrefill(model)
        output_names = ["logits", *(f"hidden_{index}" for index in range(hidden_count))]
        path = args.output / "prefill_hidden.onnx"
    else:
        if not 0 <= args.detail_layer < int(config.num_hidden_layers):
            raise SystemExit("detail layer is outside the model")
        wrapper = LayerDetailPrefill(model, args.detail_layer)
        output_names = ["logits", *LayerDetailPrefill.CAPTURE_NAMES]
        path = args.output / f"prefill_layer_{args.detail_layer}.onnx"
    dynamic_axes = {
        "input_ids": {1: "sequence"},
        "attention_mask": {1: "sequence"},
        "position_ids": {1: "sequence"},
        "logits": {0: "batch"},
    }
    for name in output_names[1:]:
        dynamic_axes[name] = {0: "batch", 1: "sequence"}

    torch.onnx.export(
        wrapper,
        (input_ids, attention_mask, position_ids),
        path,
        input_names=["input_ids", "attention_mask", "position_ids"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    externalize(path)
    metadata = {
        "kind": "analysis_only",
        "num_layers": int(config.num_hidden_layers),
        "detail_layer": args.detail_layer,
        "outputs": output_names,
        "dtype": "float16",
        "opset": 17,
    }
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
