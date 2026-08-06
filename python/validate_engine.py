#!/usr/bin/env python3
"""Validate TensorRT profile endpoints against HF and stable ORT CUDA."""

import argparse
import gc
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import tensorrt as trt
import torch
from transformers import AutoModelForCausalLM

from validate_onnx_matrix import dynamic_cache, legacy_cache, tensor_metrics


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
PREFILL_CASES = ((1, 1), (1, 256), (1, 4096))
DECODE_CASES = ((1, 1), (8, 256), (32, 4095))
APPROVED_GATE = {
    "max_fp32_rank": 3,
    "max_fp32_gap": 0.125,
    "min_cosine": 0.999,
    "min_mean_top5_overlap": 0.95,
    "nrmse": "diagnostic_only",
}


def gpu_gate(min_free_gb: float) -> None:
    free_mib = int(
        subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
                "--id=0",
            ],
            text=True,
        ).strip()
    )
    process_text = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    foreign = []
    for line in process_text.splitlines():
        if not line.strip():
            continue
        pid, name, used = line.split(", ", 2)
        if int(pid) != os.getpid() and int(used) >= 64:
            foreign.append({"pid": int(pid), "name": name, "used_memory_mib": int(used)})
    if foreign:
        raise SystemExit(f"GPU gate: foreign compute processes are active: {foreign}")
    required_mib = int(min_free_gb * 1024)
    if free_mib < required_mib:
        raise SystemExit(f"GPU gate: {free_mib} MiB free, {required_mib} MiB required")


def ort_session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=[("CUDAExecutionProvider", {"use_tf32": "0"}), "CPUExecutionProvider"],
    )
    if "CUDAExecutionProvider" not in session.get_providers():
        raise RuntimeError("ORT CUDAExecutionProvider is unavailable")
    return session


def torch_dtype(dtype: trt.DataType) -> torch.dtype:
    mapping = {
        trt.float16: torch.float16,
        trt.float32: torch.float32,
        trt.int32: torch.int32,
        trt.int64: torch.int64,
        trt.bool: torch.bool,
    }
    if dtype not in mapping:
        raise TypeError(f"unsupported TensorRT dtype: {dtype}")
    return mapping[dtype]


class EngineRunner:
    def __init__(self, path: Path):
        self.runtime = trt.Runtime(TRT_LOGGER)
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize {path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"failed to create execution context for {path}")

    def run(self, feed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        tensors: dict[str, torch.Tensor] = {}
        for name, value in feed.items():
            expected_dtype = torch_dtype(self.engine.get_tensor_dtype(name))
            tensor = torch.from_numpy(np.ascontiguousarray(value)).to(
                device="cuda", dtype=expected_dtype
            )
            if not self.context.set_input_shape(name, tuple(tensor.shape)):
                raise RuntimeError(f"failed to set TensorRT shape for {name}: {tensor.shape}")
            tensors[name] = tensor
            if not self.context.set_tensor_address(name, tensor.data_ptr()):
                raise RuntimeError(f"failed to bind TensorRT input {name}")

        output_names = [
            self.engine.get_tensor_name(index)
            for index in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(index))
            == trt.TensorIOMode.OUTPUT
        ]
        for name in output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            if not shape or any(dimension < 0 for dimension in shape):
                raise RuntimeError(f"unresolved TensorRT output shape for {name}: {shape}")
            tensor = torch.empty(
                shape, dtype=torch_dtype(self.engine.get_tensor_dtype(name)), device="cuda"
            )
            tensors[name] = tensor
            if not self.context.set_tensor_address(name, tensor.data_ptr()):
                raise RuntimeError(f"failed to bind TensorRT output {name}")

        # Input copies above are enqueued on PyTorch's current stream. Execute
        # TensorRT on that same stream so the bindings are ready before enqueue.
        stream = torch.cuda.current_stream()
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 failed")
        stream.synchronize()
        return {name: tensors[name].cpu().numpy() for name in output_names}


def output_metrics(reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray]) -> dict:
    if reference.keys() != candidate.keys():
        raise RuntimeError(
            f"output names differ: ORT={sorted(reference)} TRT={sorted(candidate)}"
        )
    tensors = {}
    for name in reference:
        tensors[name] = tensor_metrics(
            reference[name], candidate[name], 5e-2, 5e-2, name == "logits"
        )
    logits = tensors["logits"]
    reference_rows = reference["logits"].astype(np.float32).reshape(
        reference["logits"].shape[0], -1
    )
    candidate_rows = candidate["logits"].astype(np.float32).reshape(
        candidate["logits"].shape[0], -1
    )
    candidate_tokens = candidate_rows.argmax(axis=1)
    ranks = []
    gaps = []
    for row, token in zip(reference_rows, candidate_tokens, strict=True):
        score = row[token]
        ranks.append(int(1 + np.count_nonzero(row > score)))
        gaps.append(float(row.max() - score))
    core_passed = (
        all(metric["finite"] for metric in tensors.values())
        and logits["cosine_similarity_min"] >= APPROVED_GATE["min_cosine"]
        and max(ranks) <= APPROVED_GATE["max_fp32_rank"]
        and max(gaps) <= APPROVED_GATE["max_fp32_gap"]
    )
    return {
        "core_passed": core_passed,
        "passed": core_passed
        and logits["top5_overlap_mean"] >= APPROVED_GATE["min_mean_top5_overlap"],
        "production_fp32_ranks": ranks,
        "production_fp32_gaps": gaps,
        "tensors": tensors,
    }


def as_named_outputs(session: ort.InferenceSession, outputs: list[np.ndarray]) -> dict[str, np.ndarray]:
    return {item.name: value for item, value in zip(session.get_outputs(), outputs, strict=True)}


def hf_prefill(model, ids: np.ndarray) -> dict[str, np.ndarray]:
    sequence = ids.shape[1]
    ids_t = torch.from_numpy(ids.astype(np.int64)).cuda()
    with torch.inference_mode():
        result = model(
            input_ids=ids_t,
            attention_mask=torch.ones_like(ids_t),
            position_ids=torch.arange(sequence, device="cuda").unsqueeze(0),
            use_cache=True,
        )
    outputs = {"logits": result.logits[:, -1].float().cpu().numpy()}
    for layer, (key, value) in enumerate(legacy_cache(result.past_key_values)):
        outputs[f"present_key_{layer}"] = key.cpu().numpy()
        outputs[f"present_value_{layer}"] = value.cpu().numpy()
    return outputs


def hf_decode(model, feed: dict[str, np.ndarray], layers: int) -> dict[str, np.ndarray]:
    model_dtype = next(model.parameters()).dtype
    pairs = [
        (
            torch.from_numpy(feed[f"past_key_{layer}"]).to(device="cuda", dtype=model_dtype),
            torch.from_numpy(feed[f"past_value_{layer}"]).to(device="cuda", dtype=model_dtype),
        )
        for layer in range(layers)
    ]
    with torch.inference_mode():
        result = model(
            input_ids=torch.from_numpy(feed["input_ids"].astype(np.int64)).cuda(),
            attention_mask=torch.from_numpy(feed["attention_mask"].astype(np.int64)).cuda(),
            position_ids=torch.from_numpy(feed["position_ids"].astype(np.int64)).cuda(),
            past_key_values=dynamic_cache(pairs, model.config),
            use_cache=True,
        )
    outputs = {"logits": result.logits[:, -1].float().cpu().numpy()}
    for layer, (key, value) in enumerate(legacy_cache(result.past_key_values)):
        outputs[f"present_key_{layer}"] = key[..., -1:, :].cpu().numpy()
        outputs[f"present_value_{layer}"] = value[..., -1:, :].cpu().numpy()
    return outputs


def make_prefill_feed(batch: int, sequence: int, vocab_size: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260807 + sequence)
    ids = rng.integers(100, vocab_size - 256, size=(batch, sequence), dtype=np.int32)
    return {
        "input_ids": ids,
        "attention_mask": np.ones((batch, sequence), dtype=np.int32),
        "position_ids": np.arange(sequence, dtype=np.int32)[None, :],
    }


def make_decode_feed(
    batch: int, history: int, layers: int, kv_heads: int, head_dim: int, vocab_size: int
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260807 + batch * 10000 + history)
    feed = {
        "input_ids": rng.integers(100, vocab_size - 256, size=(batch, 1), dtype=np.int32),
        "attention_mask": np.ones((batch, history + 1), dtype=np.int32),
        "position_ids": np.full((batch, 1), history, dtype=np.int32),
    }
    shape = (batch, kv_heads, history, head_dim)
    for layer in range(layers):
        # Sparse deterministic patterns exercise addressing without spending minutes
        # generating several GiB of random endpoint KV data.
        key = np.zeros(shape, dtype=np.float16)
        value = np.zeros(shape, dtype=np.float16)
        key[..., 0, 0] = np.float16((layer + 1) / layers)
        value[..., -1, -1] = np.float16(-(layer + 1) / layers)
        feed[f"past_key_{layer}"] = key
        feed[f"past_value_{layer}"] = value
    return feed


def make_decode_feed_from_prefill(
    runner: EngineRunner,
    batch: int,
    history: int,
    layers: int,
    vocab_size: int,
) -> dict[str, np.ndarray]:
    """Build a decode endpoint from production Prefill Engine KV tensors.

    One deterministic B1 history is replicated across the batch.  The decode
    tokens remain different for every row, so this exercises the requested
    B/H endpoint without using a degenerate, almost-all-zero synthetic cache.
    """
    rng = np.random.default_rng(20260807 + batch * 10000 + history)
    history_ids = rng.integers(
        100, vocab_size - 256, size=(1, history), dtype=np.int32
    )
    prefill_feed = {
        "input_ids": history_ids,
        "attention_mask": np.ones((1, history), dtype=np.int32),
        "position_ids": np.arange(history, dtype=np.int32)[None, :],
    }
    prefill_outputs = runner.run(prefill_feed)
    feed = {
        "input_ids": rng.integers(
            100, vocab_size - 256, size=(batch, 1), dtype=np.int32
        ),
        "attention_mask": np.ones((batch, history + 1), dtype=np.int32),
        "position_ids": np.full((batch, 1), history, dtype=np.int32),
    }
    for layer in range(layers):
        feed[f"past_key_{layer}"] = np.repeat(
            prefill_outputs[f"present_key_{layer}"], batch, axis=0
        )
        feed[f"past_value_{layer}"] = np.repeat(
            prefill_outputs[f"present_value_{layer}"], batch, axis=0
        )
    return feed


def validate_case(
    name: str,
    feed: dict[str, np.ndarray],
    hf_outputs: dict[str, np.ndarray],
    onnx_path: Path,
    engine_path: Path,
) -> dict:
    started = time.perf_counter()
    session = ort_session(onnx_path)
    ort_outputs = as_named_outputs(session, session.run(None, feed))
    hf_to_ort = output_metrics(hf_outputs, ort_outputs)
    del session
    gc.collect()
    torch.cuda.empty_cache()
    runner = EngineRunner(engine_path)
    trt_outputs = runner.run(feed)
    hf_to_trt = output_metrics(hf_outputs, trt_outputs)
    ort_to_trt = output_metrics(ort_outputs, trt_outputs)
    del runner
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "name": name,
        # Top-5 is an aggregate gate over the whole validation population;
        # cosine/rank/gap and finiteness remain per-case hard gates.
        "passed": hf_to_ort["core_passed"] and hf_to_trt["core_passed"],
        "hf_to_ort": hf_to_ort,
        "hf_to_trt": hf_to_trt,
        "ort_to_trt": ort_to_trt,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("artifacts/model"))
    parser.add_argument("--onnx-dir", type=Path, default=Path("artifacts/onnx/fp16"))
    parser.add_argument(
        "--engine-dir", type=Path, default=Path("artifacts/engines/rtx5090/fp16")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/validation/rtx5090_engine.json")
    )
    parser.add_argument("--min-free-gb", type=float, default=28.0)
    parser.add_argument("--kind", choices=["prefill", "decode", "both"], default="both")
    parser.add_argument(
        "--decode-kv-source",
        choices=["prefill", "sparse"],
        default="prefill",
        help="Use production Prefill Engine KV by default; sparse is a diagnostic stress case.",
    )
    args = parser.parse_args()
    kinds = ("prefill", "decode") if args.kind == "both" else (args.kind,)
    required_artifacts = list(
        path
        for kind in kinds
        for path in (args.onnx_dir / f"{kind}.onnx", args.engine_dir / f"{kind}.plan")
    )
    if "decode" in kinds and args.decode_kv_source == "prefill":
        required_artifacts.append(args.engine_dir / "prefill.plan")
    missing = [str(path) for path in required_artifacts if not path.is_file()]
    if missing:
        raise SystemExit(f"missing validation artifacts: {missing}")
    gpu_gate(args.min_free_gb)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="eager",
    ).eval().cuda()
    config = model.config
    cases = []
    prefill_runner = None
    try:
        if "prefill" in kinds:
            for batch, sequence in PREFILL_CASES:
                feed = make_prefill_feed(batch, sequence, config.vocab_size)
                hf_outputs = hf_prefill(model, feed["input_ids"])
                cases.append(
                    validate_case(
                        f"prefill_b{batch}_s{sequence}",
                        feed,
                        hf_outputs,
                        args.onnx_dir / "prefill.onnx",
                        args.engine_dir / "prefill.plan",
                    )
                )
        if "decode" in kinds:
            if args.decode_kv_source == "prefill":
                prefill_runner = EngineRunner(args.engine_dir / "prefill.plan")
            for batch, history in DECODE_CASES:
                if prefill_runner is not None:
                    feed = make_decode_feed_from_prefill(
                        prefill_runner,
                        batch,
                        history,
                        config.num_hidden_layers,
                        config.vocab_size,
                    )
                else:
                    feed = make_decode_feed(
                        batch,
                        history,
                        config.num_hidden_layers,
                        config.num_key_value_heads,
                        config.hidden_size // config.num_attention_heads,
                        config.vocab_size,
                    )
                hf_outputs = hf_decode(model, feed, config.num_hidden_layers)
                cases.append(
                    validate_case(
                        f"decode_b{batch}_h{history}",
                        feed,
                        hf_outputs,
                        args.onnx_dir / "decode.onnx",
                        args.engine_dir / "decode.plan",
                    )
                )
                del feed, hf_outputs
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        del prefill_runner
        del model
        gc.collect()
        torch.cuda.empty_cache()
    top5_aggregate = {}
    for comparison in ("hf_to_ort", "hf_to_trt"):
        weighted_sum = 0.0
        rows = 0
        for case in cases:
            logits = case[comparison]["tensors"]["logits"]
            batch_rows = int(logits["shape"][0])
            weighted_sum += logits["top5_overlap_mean"] * batch_rows
            rows += batch_rows
        top5_aggregate[comparison] = weighted_sum / rows
    report = {
        "passed": all(case["passed"] for case in cases)
        and all(
            value >= APPROVED_GATE["min_mean_top5_overlap"]
            for value in top5_aggregate.values()
        ),
        "gate": APPROVED_GATE,
        "top5_aggregate": top5_aggregate,
        "decode_kv_source": args.decode_kv_source,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({"passed": report["passed"], "cases": len(cases)}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
