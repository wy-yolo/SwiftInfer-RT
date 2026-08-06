#!/usr/bin/env python3
"""Capture detailed top-k evidence for a validation-corpus token mismatch."""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from validate_onnx_matrix import session_for, tensor_metrics


def topk(logits: np.ndarray, count: int) -> list[dict]:
    row = logits.astype(np.float32).reshape(-1)
    indices = np.argsort(row)[-count:][::-1]
    return [
        {"token_id": int(index), "logit": float(row[index])}
        for index in indices
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--model", type=Path, default=Path("artifacts/model"))
    parser.add_argument("--requests", type=Path, default=Path("artifacts/validation/requests.jsonl"))
    parser.add_argument("--onnx-dir", type=Path, default=Path("artifacts/onnx/fp16"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--include-fp32", action="store_true")
    args = parser.parse_args()
    records = [json.loads(line) for line in args.requests.read_text().splitlines() if line]
    matches = [record for record in records if record["request_id"] == args.request_id]
    if len(matches) != 1:
        raise SystemExit(f"expected one request named {args.request_id}, found {len(matches)}")
    record = matches[0]
    ids_np = np.asarray([record["input_ids"]], dtype=np.int32)
    sequence = ids_np.shape[1]

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.float16,
        attn_implementation="eager",
    ).eval().cuda()
    ids = torch.from_numpy(ids_np.astype(np.int64)).cuda()
    with torch.inference_mode():
        hf = model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            position_ids=torch.arange(sequence, device="cuda").unsqueeze(0),
            use_cache=True,
        )
    hf_logits = hf.logits[:, -1].float().cpu().numpy()
    feeds = {
        "input_ids": ids_np,
        "attention_mask": np.ones_like(ids_np, dtype=np.int32),
        "position_ids": np.arange(sequence, dtype=np.int32)[None, :],
    }
    ort_results = {}
    for mode in ("cpu", "cuda_noopt", "cuda"):
        session = session_for(mode, args.onnx_dir / "prefill.onnx")
        ort_results[mode] = session.run(["logits"], feeds)[0]

    comparisons = {
        mode: tensor_metrics(hf_logits, logits, 5e-2, 5e-2, True)
        for mode, logits in ort_results.items()
    }
    candidates = set(int(item["token_id"]) for item in topk(hf_logits, args.top_k))
    for logits in ort_results.values():
        candidates.update(int(item["token_id"]) for item in topk(logits, args.top_k))
    candidate_scores = []
    for token_id in sorted(candidates):
        candidate_scores.append(
            {
                "token_id": token_id,
                "hf": float(hf_logits[0, token_id]),
                **{
                    mode: float(logits[0, token_id])
                    for mode, logits in ort_results.items()
                },
            }
        )
    report = {
        "request": record,
        "hf_topk": topk(hf_logits, args.top_k),
        "ort_topk": {mode: topk(logits, args.top_k) for mode, logits in ort_results.items()},
        "comparisons": comparisons,
        "candidate_scores": candidate_scores,
    }
    if args.include_fp32:
        del model, hf, ids
        gc.collect()
        torch.cuda.empty_cache()
        fp32_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            local_files_only=True,
            dtype=torch.float32,
            attn_implementation="eager",
        ).eval().cuda()
        fp32_ids = torch.from_numpy(ids_np.astype(np.int64)).cuda()
        with torch.inference_mode():
            fp32_hf = fp32_model(
                input_ids=fp32_ids,
                attention_mask=torch.ones_like(fp32_ids),
                position_ids=torch.arange(sequence, device="cuda").unsqueeze(0),
                use_cache=True,
            )
        fp32_hf_logits = fp32_hf.logits[:, -1].cpu().numpy()
        fp32_session = session_for("cuda_noopt", Path("artifacts/onnx/fp32/prefill.onnx"))
        fp32_ort_logits = fp32_session.run(["logits"], feeds)[0]
        report["fp32"] = {
            "hf_topk": topk(fp32_hf_logits, args.top_k),
            "ort_topk": topk(fp32_ort_logits, args.top_k),
            "comparison": tensor_metrics(
                fp32_hf_logits, fp32_ort_logits, 1e-3, 1e-3, True
            ),
        }
    output = args.output or Path("results/diagnostics") / f"{args.request_id}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
