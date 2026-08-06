#!/usr/bin/env python3
"""Adjudicate TensorRT/C++ greedy divergences against the FP32 reference."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from adjudicate_fp16_mismatches import adjudicate, gpu_gate
from validate_onnx_matrix import session_for


def common_prefix(left: list[int], right: list[int]) -> int:
    return next(
        (index for index, (a, b) in enumerate(zip(left, right)) if a != b),
        min(len(left), len(right)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("artifacts/model"))
    parser.add_argument("--fp32-onnx", type=Path, default=Path("artifacts/onnx/fp32"))
    parser.add_argument("--requests", type=Path, default=Path("artifacts/validation/requests.jsonl"))
    parser.add_argument(
        "--fp16-reference", type=Path, default=Path("results/validation/fp16_corpus.jsonl")
    )
    parser.add_argument(
        "--production", type=Path, default=Path("results/validation/cpp_runtime_corpus.jsonl")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/validation/runtime_fp32_adjudication.jsonl")
    )
    parser.add_argument(
        "--summary", type=Path, default=Path("results/validation/runtime_fp32_adjudication_summary.json")
    )
    parser.add_argument("--min-free-gb", type=float, default=28.0)
    parser.add_argument("--max-fp32-rank", type=int, default=3)
    parser.add_argument("--max-fp32-gap", type=float, default=0.125)
    args = parser.parse_args()
    gpu_gate(args.min_free_gb)

    requests = {
        item["request_id"]: item
        for item in (json.loads(line) for line in args.requests.read_text().splitlines() if line)
    }
    references = {
        item["request_id"]: item
        for item in (
            json.loads(line) for line in args.fp16_reference.read_text().splitlines() if line
        )
    }
    production = {
        item["request_id"]: item["output_ids"]
        for item in (json.loads(line) for line in args.production.read_text().splitlines() if line)
    }

    cases = []
    for request_id, reference in references.items():
        candidate = production[request_id]
        hf_tokens = reference["hf_output_ids"]
        ort_tokens = reference["ort_output_ids"]
        if hf_tokens == ort_tokens:
            baseline = hf_tokens
            baseline_name = "hf_fp16_equals_ort_fp16"
            if candidate == baseline:
                continue
        elif candidate[: len(ort_tokens)] == ort_tokens:
            baseline = hf_tokens
            baseline_name = "hf_fp16"
        elif candidate[: len(hf_tokens)] == hf_tokens:
            baseline = ort_tokens
            baseline_name = "ort_fp16"
        elif common_prefix(candidate, hf_tokens) >= common_prefix(candidate, ort_tokens):
            baseline = hf_tokens
            baseline_name = "hf_fp16"
        else:
            baseline = ort_tokens
            baseline_name = "ort_fp16"
        divergence = common_prefix(candidate, baseline)
        if divergence >= min(len(candidate), len(baseline)):
            raise RuntimeError(f"cannot identify a token divergence for {request_id}")
        # The shared prefix and first divergent token are sufficient for the
        # FP32 rank/gap decision.  Later tokens follow a different greedy path.
        cases.append(
            (
                request_id,
                baseline_name,
                {
                    "hf_output_ids": baseline[: divergence + 1],
                    "ort_output_ids": candidate[: divergence + 1],
                },
            )
        )

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

    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for index, (request_id, baseline_name, synthetic) in enumerate(cases, start=1):
            result = adjudicate(
                requests[request_id],
                synthetic,
                model,
                prefill_session,
                decode_session,
                args.max_fp32_rank,
                args.max_fp32_gap,
            )
            result["baseline"] = baseline_name
            result["production"] = "tensorrt_cpp"
            results.append(result)
            stream.write(json.dumps(result, separators=(",", ":")) + "\n")
            stream.flush()
            if index % 10 == 0 or not result["passed"]:
                print(
                    json.dumps(
                        {
                            "progress": f"{index}/{len(cases)}",
                            "request_id": request_id,
                            "passed": result["passed"],
                        }
                    ),
                    flush=True,
                )

    ranks = [
        result["decision"]["production_fp32_rank"]
        for result in results
        if result["decision"]["production_fp32_rank"] is not None
    ]
    gaps = [
        result["decision"]["production_fp32_gap"]
        for result in results
        if result["decision"]["production_fp32_gap"] is not None
    ]
    summary = {
        "schema_version": 1,
        "production": "tensorrt_cpp",
        "requests_total": len(references),
        "exact_without_adjudication": len(references) - len(results),
        "adjudicated_requests": len(results),
        "fp32_consistent": sum(result["decision"]["fp32_consistent"] for result in results),
        "max_observed_fp32_rank": max(ranks, default=1),
        "max_observed_fp32_gap": max(gaps, default=0.0),
        "gate": {
            "max_fp32_rank": args.max_fp32_rank,
            "max_fp32_gap": args.max_fp32_gap,
        },
        "passed": all(result["passed"] for result in results),
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    del model, prefill_session, decode_session
    gc.collect()
    torch.cuda.empty_cache()
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
