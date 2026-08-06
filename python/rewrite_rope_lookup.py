#!/usr/bin/env python3
"""Replace exported Qwen RoPE Sin/Cos computation with constant lookup tables.

TensorRT's runtime trigonometric implementation can lose substantial accuracy at
large positions.  This post-export rewrite computes Qwen's RoPE table with
PyTorch (CUDA by default) and replaces the two final RoPE casts with ONNX Gather
nodes indexed by the public ``position_ids`` input.  The model interface and all
model outputs remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper


COS_NODE = "/model/model/rotary_emb/Cast_4"
SIN_NODE = "/model/model/rotary_emb/Cast_5"
COS_TABLE = "minillm.rope_cos_table"
SIN_TABLE = "minillm.rope_sin_table"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_tables(config: dict, max_positions: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    hidden_size = int(config["hidden_size"])
    attention_heads = int(config["num_attention_heads"])
    if hidden_size % attention_heads:
        raise ValueError("hidden_size must be divisible by num_attention_heads")
    head_dim = hidden_size // attention_heads
    if head_dim % 2:
        raise ValueError("RoPE head dimension must be even")
    rope_scaling = config.get("rope_scaling")
    if rope_scaling not in (None, {}):
        raise ValueError(f"rope_scaling is not supported by this rewrite: {rope_scaling!r}")

    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA table generation requested but torch.cuda.is_available() is false")
    base = float(config.get("rope_theta", 10000.0))
    with torch.inference_mode():
        # Match transformers.models.qwen2.modeling_qwen2.Qwen2RotaryEmbedding:
        # frequency construction and trigonometry are evaluated in FP32, then
        # converted to the activation dtype used by the exported FP16 model.
        indices = torch.arange(0, head_dim, 2, dtype=torch.int64, device=torch_device)
        inv_freq = 1.0 / (base ** (indices.float() / head_dim))
        positions = torch.arange(max_positions, dtype=torch.int64, device=torch_device)
        inv_expanded = inv_freq[None, :, None].float()
        pos_expanded = positions[None, None, :].float()
        frequencies = (inv_expanded @ pos_expanded).transpose(1, 2)
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        cosine = embedding.cos().to(torch.float16).squeeze(0).cpu().numpy()
        sine = embedding.sin().to(torch.float16).squeeze(0).cpu().numpy()
    if cosine.shape != (max_positions, head_dim) or sine.shape != cosine.shape:
        raise RuntimeError(f"unexpected RoPE table shapes: cos={cosine.shape}, sin={sine.shape}")
    if not np.isfinite(cosine).all() or not np.isfinite(sine).all():
        raise RuntimeError("generated RoPE table contains NaN or Inf")
    return cosine, sine


def prune_dead_graph(model: onnx.ModelProto) -> tuple[int, int]:
    """Remove nodes and initializers that cannot reach a public graph output."""
    nodes = list(model.graph.node)
    producer: dict[str, int] = {}
    for index, node in enumerate(nodes):
        for output in node.output:
            if output:
                producer[output] = index

    pending = [output.name for output in model.graph.output]
    live_values = set(pending)
    live_nodes: set[int] = set()
    while pending:
        value = pending.pop()
        index = producer.get(value)
        if index is None or index in live_nodes:
            continue
        live_nodes.add(index)
        for input_name in nodes[index].input:
            if input_name and input_name not in live_values:
                live_values.add(input_name)
                pending.append(input_name)

    removed_nodes = len(nodes) - len(live_nodes)
    del model.graph.node[:]
    model.graph.node.extend(node for index, node in enumerate(nodes) if index in live_nodes)

    initializers = list(model.graph.initializer)
    removed_initializers = sum(initializer.name not in live_values for initializer in initializers)
    del model.graph.initializer[:]
    model.graph.initializer.extend(
        initializer for initializer in initializers if initializer.name in live_values
    )
    return removed_nodes, removed_initializers


def rewrite_model(source: Path, destination: Path, cosine: np.ndarray, sine: np.ndarray) -> dict:
    model = onnx.load_model(source, load_external_data=True)
    replacements = {
        COS_NODE: (COS_TABLE, cosine),
        SIN_NODE: (SIN_TABLE, sine),
    }
    found: set[str] = set()
    rewritten_nodes = []
    for node in model.graph.node:
        replacement = replacements.get(node.name)
        if replacement is None:
            rewritten_nodes.append(node)
            continue
        table_name, _ = replacement
        if len(node.output) != 1:
            raise RuntimeError(f"{node.name} has unexpected outputs: {list(node.output)}")
        rewritten_nodes.append(
            helper.make_node(
                "Gather",
                [table_name, "position_ids"],
                list(node.output),
                axis=0,
                name=node.name,
            )
        )
        found.add(node.name)
    missing = set(replacements) - found
    if missing:
        raise RuntimeError(f"RoPE nodes were not found in {source}: {sorted(missing)}")

    del model.graph.node[:]
    model.graph.node.extend(rewritten_nodes)
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(cosine, COS_TABLE),
            numpy_helper.from_array(sine, SIN_TABLE),
        ]
    )
    removed_nodes, removed_initializers = prune_dead_graph(model)
    onnx.checker.check_model(model, full_check=True)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rope-lookup-", dir=destination.parent) as temporary:
        temporary_dir = Path(temporary)
        temporary_model = temporary_dir / destination.name
        external_name = destination.name + ".data"
        onnx.save_model(
            model,
            temporary_model,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=external_name,
            size_threshold=1024,
            convert_attribute=False,
        )
        onnx.checker.check_model(str(temporary_model), full_check=True)
        temporary_data = temporary_dir / external_name
        if not temporary_data.is_file():
            raise RuntimeError(f"external data was not generated for {destination}")
        final_data = destination.with_name(external_name)
        os.replace(temporary_data, final_data)
        os.replace(temporary_model, destination)

    return {
        "source": str(source.resolve()),
        "destination": str(destination.resolve()),
        "source_sha256": sha256(source),
        "onnx_sha256": sha256(destination),
        "external_data": str(destination.with_name(destination.name + ".data").resolve()),
        "external_data_sha256": sha256(destination.with_name(destination.name + ".data")),
        "removed_nodes": removed_nodes,
        "removed_initializers": removed_initializers,
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path("artifacts/onnx/fp16"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/onnx/fp16_rope_lookup")
    )
    parser.add_argument("--model-config", type=Path, default=Path("artifacts/model/config.json"))
    parser.add_argument("--max-positions", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.max_positions < 1:
        raise ValueError("--max-positions must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing = [
        path
        for kind in ("prefill", "decode")
        for path in (
            args.output_dir / f"{kind}.onnx",
            args.output_dir / f"{kind}.onnx.data",
        )
        if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing lookup artifacts: " + ", ".join(map(str, existing))
        )

    config = json.loads(args.model_config.read_text())
    cosine, sine = build_tables(config, args.max_positions, args.device)
    results = []
    try:
        for kind in ("prefill", "decode"):
            results.append(
                rewrite_model(
                    args.source_dir / f"{kind}.onnx",
                    args.output_dir / f"{kind}.onnx",
                    cosine,
                    sine,
                )
            )
    except Exception:
        # Outputs are new generated artifacts; remove the partial set so a
        # failed rewrite can never be mistaken for a complete pair.
        for path in args.output_dir.glob("*.onnx*"):
            path.unlink(missing_ok=True)
        raise

    for auxiliary in ("model_spec.json",):
        source = args.source_dir / auxiliary
        if source.is_file():
            shutil.copy2(source, args.output_dir / auxiliary)
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "generation_device": args.device,
        "generation_cuda_device": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
        "max_positions": args.max_positions,
        "head_dim": cosine.shape[1],
        "dtype": str(cosine.dtype),
        "rope_theta": float(config.get("rope_theta", 10000.0)),
        "rope_scaling": config.get("rope_scaling"),
        "table_cos_sha256": hashlib.sha256(cosine.tobytes()).hexdigest(),
        "table_sin_sha256": hashlib.sha256(sine.tobytes()).hexdigest(),
        "models": results,
    }
    manifest_path = args.output_dir / "rope_lookup_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
