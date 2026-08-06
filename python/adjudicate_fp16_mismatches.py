#!/usr/bin/env python3
"""Adjudicate FP16 greedy disagreements with the strict FP32 reference path."""

import argparse
import gc
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from validate_onnx_matrix import legacy_cache, session_for, tensor_metrics


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


def append_cache(previous: list[tuple[np.ndarray, np.ndarray]], outputs: list[np.ndarray]):
    return [
        (
            np.concatenate((key, outputs[1 + 2 * layer]), axis=2),
            np.concatenate((value, outputs[2 + 2 * layer]), axis=2),
        )
        for layer, (key, value) in enumerate(previous)
    ]


def topk(logits: np.ndarray, count: int = 5) -> list[dict]:
    row = logits.astype(np.float32).reshape(-1)
    indices = np.argsort(row)[-count:][::-1]
    return [{"token_id": int(index), "logit": float(row[index])} for index in indices]


def adjudicate(
    record: dict,
    fp16_result: dict,
    model,
    prefill_session,
    decode_session,
    max_fp32_rank: int,
    max_fp32_gap: float,
) -> dict:
    started = time.perf_counter()
    ids_np = np.asarray([record["input_ids"]], dtype=np.int32)
    sequence = ids_np.shape[1]
    ids = torch.from_numpy(ids_np.astype(np.int64)).cuda()
    with torch.inference_mode():
        hf = model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            position_ids=torch.arange(sequence, dtype=torch.long, device="cuda").unsqueeze(0),
            use_cache=True,
        )
    ort_outputs = prefill_session.run(
        None,
        {
            "input_ids": ids_np,
            "attention_mask": np.ones_like(ids_np, dtype=np.int32),
            "position_ids": np.arange(sequence, dtype=np.int32)[None, :],
        },
    )
    ort_cache = [
        (ort_outputs[1 + 2 * layer], ort_outputs[2 + 2 * layer])
        for layer in range((len(ort_outputs) - 1) // 2)
    ]
    hf_logits = hf.logits[:, -1].cpu().numpy()
    ort_logits = ort_outputs[0]
    hf_fp16_tokens = fp16_result["hf_output_ids"]
    ort_fp16_tokens = fp16_result["ort_output_ids"]
    if len(hf_fp16_tokens) != len(ort_fp16_tokens):
        raise RuntimeError("stored FP16 sequences have different lengths")
    divergence = next(
        index
        for index, (left, right) in enumerate(zip(hf_fp16_tokens, ort_fp16_tokens, strict=True))
        if left != right
    )
    prefix_checks = []
    decision = None
    for step in range(divergence + 1):
        metric = tensor_metrics(hf_logits, ort_logits, 1e-3, 1e-3, True)
        hf_fp32_token = int(hf_logits.argmax(axis=-1)[0])
        ort_fp32_token = int(ort_logits.argmax(axis=-1)[0])
        fp32_consistent = metric["allclose"] and hf_fp32_token == ort_fp32_token
        if step == divergence:
            production_token = int(ort_fp16_tokens[step])
            reference_fp16_token = int(hf_fp16_tokens[step])
            hf_fp32_top5 = topk(hf_logits)
            production_ranks = [
                index + 1
                for index, item in enumerate(hf_fp32_top5)
                if item["token_id"] == production_token
            ]
            production_rank = production_ranks[0] if production_ranks else None
            production_scores = [
                item["logit"]
                for item in hf_fp32_top5
                if item["token_id"] == production_token
            ]
            production_gap = (
                float(hf_fp32_top5[0]["logit"] - production_scores[0])
                if production_scores
                else None
            )
            decision = {
                "step": step,
                "hf_fp16_token": reference_fp16_token,
                "ort_fp16_production_token": production_token,
                "hf_fp32_token": hf_fp32_token,
                "ort_fp32_token": ort_fp32_token,
                "fp32_consistent": fp32_consistent,
                "production_matches_fp32": production_token == hf_fp32_token,
                "hf_fp16_matches_fp32": reference_fp16_token == hf_fp32_token,
                "production_fp32_rank": production_rank,
                "production_fp32_gap": production_gap,
                "hf_fp32_top5": hf_fp32_top5,
                "ort_fp32_top5": topk(ort_logits),
                "fp32_metric": metric,
            }
            break
        common_token = int(hf_fp16_tokens[step])
        if common_token != int(ort_fp16_tokens[step]):
            raise RuntimeError("unexpected FP16 divergence before stored mismatch")
        prefix_checks.append(
            {
                "step": step,
                "token": common_token,
                "hf_fp32_token": hf_fp32_token,
                "ort_fp32_token": ort_fp32_token,
                "fp32_consistent": fp32_consistent,
                "common_token_matches_fp32": common_token == hf_fp32_token,
            }
        )
        past_length = sequence + step
        token_tensor = torch.tensor([[common_token]], dtype=torch.long, device="cuda")
        with torch.inference_mode():
            hf = model(
                input_ids=token_tensor,
                attention_mask=torch.ones((1, past_length + 1), dtype=torch.long, device="cuda"),
                position_ids=torch.tensor([[past_length]], dtype=torch.long, device="cuda"),
                past_key_values=hf.past_key_values,
                use_cache=True,
            )
        hf_logits = hf.logits[:, -1].cpu().numpy()
        decode_feed = {
            "input_ids": np.asarray([[common_token]], dtype=np.int32),
            "attention_mask": np.ones((1, past_length + 1), dtype=np.int32),
            "position_ids": np.asarray([[past_length]], dtype=np.int32),
        }
        for layer, (key, value) in enumerate(ort_cache):
            decode_feed[f"past_key_{layer}"] = key
            decode_feed[f"past_value_{layer}"] = value
        ort_outputs = decode_session.run(None, decode_feed)
        ort_logits = ort_outputs[0]
        ort_cache = append_cache(ort_cache, ort_outputs)
    if decision is None:
        raise RuntimeError("failed to adjudicate stored mismatch")
    passed = (
        decision["fp32_consistent"]
        and decision["production_fp32_rank"] is not None
        and decision["production_fp32_rank"] <= max_fp32_rank
        and decision["production_fp32_gap"] <= max_fp32_gap
    )
    return {
        "request_id": record["request_id"],
        "task_id": int(record["task_id"]),
        "passed": passed,
        "prefix_checks": prefix_checks,
        "decision": decision,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("artifacts/model"))
    parser.add_argument("--fp32-onnx", type=Path, default=Path("artifacts/onnx/fp32"))
    parser.add_argument("--requests", type=Path, default=Path("artifacts/validation/requests.jsonl"))
    parser.add_argument("--fp16-results", type=Path, default=Path("results/validation/fp16_corpus.jsonl"))
    parser.add_argument("--fp16-analysis", type=Path, default=Path("results/validation/fp16_corpus_analysis.json"))
    parser.add_argument("--output", type=Path, default=Path("results/validation/fp16_fp32_adjudication.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("results/validation/fp16_fp32_adjudication_summary.json"))
    parser.add_argument("--min-free-gb", type=float, default=28.0)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--min-top5-overlap", type=float, default=0.95)
    parser.add_argument("--max-fp32-rank", type=int, default=3)
    parser.add_argument("--max-fp32-gap", type=float, default=0.125)
    args = parser.parse_args()
    if args.max_fp32_rank < 1 or args.max_fp32_rank > 5:
        parser.error("--max-fp32-rank must be in [1, 5]")
    if args.max_fp32_gap < 0:
        parser.error("--max-fp32-gap must be non-negative")
    gpu_gate(args.min_free_gb)
    analysis = json.loads(args.fp16_analysis.read_text())
    if analysis["cosine_min"] < args.min_cosine:
        raise SystemExit("FP16 cosine gate failed")
    if analysis["mean_top5_overlap"] < args.min_top5_overlap:
        raise SystemExit("FP16 top-5 gate failed")
    requests = {
        item["request_id"]: item
        for item in (json.loads(line) for line in args.requests.read_text().splitlines() if line)
    }
    fp16_failures = [
        item
        for item in (
            json.loads(line) for line in args.fp16_results.read_text().splitlines() if line
        )
        if not item["token_match"]
    ]
    if len(fp16_failures) != analysis["token_mismatch_count"]:
        raise SystemExit("FP16 mismatch count does not match the persisted analysis")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="eager",
    ).eval().cuda()
    prefill_session = session_for("cuda_noopt", args.fp32_onnx / "prefill.onnx")
    decode_session = session_for("cuda_noopt", args.fp32_onnx / "decode.onnx")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with args.output.open("w", encoding="utf-8") as stream:
        for index, fp16_result in enumerate(fp16_failures, start=1):
            result = adjudicate(
                requests[fp16_result["request_id"]],
                fp16_result,
                model,
                prefill_session,
                decode_session,
                args.max_fp32_rank,
                args.max_fp32_gap,
            )
            results.append(result)
            stream.write(json.dumps(result, separators=(",", ":")) + "\n")
            stream.flush()
            print(
                json.dumps(
                    {
                        "progress": f"{index}/{len(fp16_failures)}",
                        "request_id": result["request_id"],
                        "passed": result["passed"],
                    }
                ),
                flush=True,
            )
    summary = {
        "schema_version": 1,
        "adjudicated_requests": len(results),
        "production_matches_fp32": sum(
            result["decision"]["production_matches_fp32"] for result in results
        ),
        "hf_fp16_matches_fp32": sum(
            result["decision"]["hf_fp16_matches_fp32"] for result in results
        ),
        "fp32_consistent": sum(result["decision"]["fp32_consistent"] for result in results),
        "prefix_token_disagreements_with_fp32": sum(
            not check["common_token_matches_fp32"]
            for result in results
            for check in result["prefix_checks"]
        ),
        "fp16_cosine_min": analysis["cosine_min"],
        "fp16_mean_top5_overlap": analysis["mean_top5_overlap"],
        "fp16_nrmse_max_diagnostic": analysis["nrmse_max"],
        "max_observed_fp32_rank": max(
            result["decision"]["production_fp32_rank"] for result in results
        ),
        "max_observed_fp32_gap": max(
            result["decision"]["production_fp32_gap"] for result in results
        ),
        "gate": {
            "max_fp32_rank": args.max_fp32_rank,
            "max_fp32_gap": args.max_fp32_gap,
            "min_cosine": args.min_cosine,
            "min_mean_top5_overlap": args.min_top5_overlap,
            "nrmse": "diagnostic_only",
        },
    }
    summary["passed"] = (
        len(results) == analysis["token_mismatch_count"]
        and all(result["passed"] for result in results)
        and summary["fp16_cosine_min"] >= args.min_cosine
        and summary["fp16_mean_top5_overlap"] >= args.min_top5_overlap
    )
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    del model, prefill_session, decode_session
    gc.collect()
    torch.cuda.empty_cache()
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
