#!/usr/bin/env python3
"""Resume-safe full greedy-sequence comparison between HF and ORT."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForCausalLM

from validate_onnx_matrix import legacy_cache, session_for, tensor_metrics


def gpu_state() -> tuple[int, list[dict]]:
    free = int(
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
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    processes = []
    for line in output.splitlines():
        if not line:
            continue
        pid, name, used = line.split(", ", 2)
        if int(used) >= 64:
            processes.append({"pid": int(pid), "name": name, "used_memory_mib": int(used)})
    return free, processes


def append_ort_cache(
    previous: list[tuple[np.ndarray, np.ndarray]], outputs: list[np.ndarray]
) -> list[tuple[np.ndarray, np.ndarray]]:
    result = []
    for layer, (key, value) in enumerate(previous):
        result.append(
            (
                np.concatenate((key, outputs[1 + 2 * layer]), axis=2),
                np.concatenate((value, outputs[2 + 2 * layer]), axis=2),
            )
        )
    return result


def topk(logits: np.ndarray, count: int = 5) -> list[dict]:
    row = logits.astype(np.float32).reshape(-1)
    indices = np.argsort(row)[-count:][::-1]
    return [{"token_id": int(index), "logit": float(row[index])} for index in indices]


def compare_request(
    record: dict,
    model,
    prefill_session: ort.InferenceSession,
    decode_session: ort.InferenceSession,
    representative: bool,
    thresholds: dict,
) -> dict:
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
    hf_logits = hf.logits[:, -1].float().cpu().numpy()
    ort_logits = ort_outputs[0]
    hf_tokens: list[int] = []
    ort_tokens: list[int] = []
    distribution = []
    mismatch = None
    eos = model.config.eos_token_id
    eos_ids = {int(item) for item in eos} if isinstance(eos, (list, tuple)) else {int(eos)}
    started = time.perf_counter()
    for step in range(int(record["max_new_tokens"])):
        hf_token = int(hf_logits.argmax(axis=-1)[0])
        ort_token = int(ort_logits.argmax(axis=-1)[0])
        hf_tokens.append(hf_token)
        ort_tokens.append(ort_token)
        if representative or hf_token != ort_token:
            distribution.append(tensor_metrics(hf_logits, ort_logits, 5e-2, 5e-2, True))
        if hf_token != ort_token:
            hf_row = hf_logits.astype(np.float32).reshape(-1)
            ort_row = ort_logits.astype(np.float32).reshape(-1)
            mismatch = {
                "step": step,
                "hf_token": hf_token,
                "ort_token": ort_token,
                "hf_top5": topk(hf_logits),
                "ort_top5": topk(ort_logits),
                "hf_margin": float(hf_row.max() - hf_row[ort_token]),
                "ort_margin": float(ort_row.max() - ort_row[hf_token]),
                "hf_score_for_ort_token": float(hf_row[ort_token]),
                "ort_score_for_hf_token": float(ort_row[hf_token]),
            }
        if hf_token != ort_token or hf_token in eos_ids or step + 1 == int(record["max_new_tokens"]):
            break
        past_length = sequence + step
        token_tensor = torch.tensor([[hf_token]], dtype=torch.long, device="cuda")
        with torch.inference_mode():
            hf = model(
                input_ids=token_tensor,
                attention_mask=torch.ones((1, past_length + 1), dtype=torch.long, device="cuda"),
                position_ids=torch.tensor([[past_length]], dtype=torch.long, device="cuda"),
                past_key_values=hf.past_key_values,
                use_cache=True,
            )
        hf_logits = hf.logits[:, -1].float().cpu().numpy()
        decode_feed = {
            "input_ids": np.asarray([[ort_token]], dtype=np.int32),
            "attention_mask": np.ones((1, past_length + 1), dtype=np.int32),
            "position_ids": np.asarray([[past_length]], dtype=np.int32),
        }
        for layer, (key, value) in enumerate(ort_cache):
            decode_feed[f"past_key_{layer}"] = key
            decode_feed[f"past_value_{layer}"] = value
        ort_outputs = decode_session.run(None, decode_feed)
        ort_logits = ort_outputs[0]
        ort_cache = append_ort_cache(ort_cache, ort_outputs)

    token_match = hf_tokens == ort_tokens
    distribution_passed = all(
        metric["finite"]
        and metric["cosine_similarity_min"] >= thresholds["min_cosine"]
        and metric["nrmse_max"] <= thresholds["max_nrmse"]
        for metric in distribution
    )
    return {
        "task_id": int(record["task_id"]),
        "request_id": record["request_id"],
        "prompt_length": len(record["input_ids"]),
        "max_new_tokens": int(record["max_new_tokens"]),
        "hf_output_ids": hf_tokens,
        "ort_output_ids": ort_tokens,
        "token_match": token_match,
        "representative": representative,
        "distribution_passed": distribution_passed,
        "distribution": distribution,
        "mismatch": mismatch,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("artifacts/model"))
    parser.add_argument("--onnx", type=Path, default=Path("artifacts/onnx/fp16"))
    parser.add_argument("--requests", type=Path, default=Path("artifacts/validation/requests.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/validation/manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("results/validation/fp16_corpus.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("results/validation/fp16_corpus_summary.json"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--continue-on-mismatch",
        action="store_true",
        help="finish the diagnostic corpus while preserving a failed acceptance status",
    )
    parser.add_argument("--min-free-gb", type=float, default=16.0)
    parser.add_argument("--require-exclusive", action="store_true")
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--max-nrmse", type=float, default=0.02)
    parser.add_argument("--min-top5-overlap", type=float, default=0.95)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--rerun-mismatches-from",
        type=Path,
        help="restrict records to request IDs that failed in an earlier JSONL result",
    )
    args = parser.parse_args()
    if args.progress_every < 1:
        parser.error("--progress-every must be positive")

    free_mib, processes = gpu_state()
    if free_mib < int(args.min_free_gb * 1024):
        raise SystemExit(f"GPU gate: {free_mib} MiB free")
    if args.require_exclusive and processes:
        raise SystemExit(f"GPU gate: active compute processes: {processes}")

    records = [json.loads(line) for line in args.requests.read_text().splitlines() if line]
    if args.rerun_mismatches_from:
        failed_ids = {
            item["request_id"]
            for item in (
                json.loads(line)
                for line in args.rerun_mismatches_from.read_text().splitlines()
                if line
            )
            if not item["token_match"]
        }
        records = [record for record in records if record["request_id"] in failed_ids]
    if args.limit is not None:
        records = records[: args.limit]
    first_request_by_task = {}
    for record in records:
        first_request_by_task.setdefault(int(record["task_id"]), record["request_id"])
    completed = set()
    if args.resume and args.output.exists():
        completed = {
            json.loads(line)["request_id"]
            for line in args.output.read_text().splitlines()
            if line
        }
    elif args.output.exists():
        args.output.unlink()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        attn_implementation="eager",
        local_files_only=True,
    ).eval().cuda()
    prefill_session = session_for("cuda_noopt", args.onnx / "prefill.onnx")
    decode_session = session_for("cuda_noopt", args.onnx / "decode.onnx")
    thresholds = {"min_cosine": args.min_cosine, "max_nrmse": args.max_nrmse}
    with args.output.open("a", encoding="utf-8") as stream:
        for index, record in enumerate(records, start=1):
            if record["request_id"] in completed:
                continue
            result = compare_request(
                record,
                model,
                prefill_session,
                decode_session,
                record["request_id"] == first_request_by_task[int(record["task_id"])],
                thresholds,
            )
            stream.write(json.dumps(result, separators=(",", ":")) + "\n")
            stream.flush()
            if index % args.progress_every == 0 or not result["token_match"]:
                print(
                    json.dumps(
                        {
                            "progress": f"{index}/{len(records)}",
                            "request_id": result["request_id"],
                            "token_match": result["token_match"],
                        }
                    ),
                    flush=True,
                )
            if not result["token_match"] and not args.continue_on_mismatch:
                break

    results = [json.loads(line) for line in args.output.read_text().splitlines() if line]
    result_ids = {item["request_id"] for item in results}
    relevant = [item for item in results if item["request_id"] in {r["request_id"] for r in records}]
    representative_metrics = [
        metric
        for item in relevant
        if item["representative"]
        for metric in item["distribution"]
    ]
    mean_top5 = (
        float(np.mean([metric["top5_overlap_mean"] for metric in representative_metrics]))
        if representative_metrics
        else 0.0
    )
    summary = {
        "schema_version": 1,
        "expected_requests": len(records),
        "completed_requests": sum(record["request_id"] in result_ids for record in records),
        "representative_tasks": sum(item["representative"] for item in relevant),
        "representative_steps": len(representative_metrics),
        "token_match": all(item["token_match"] for item in relevant),
        "token_mismatch_count": sum(not item["token_match"] for item in relevant),
        "numeric_failure_count": sum(
            not metric["finite"]
            or metric["cosine_similarity_min"] < args.min_cosine
            or metric["nrmse_max"] > args.max_nrmse
            for metric in representative_metrics
        ),
        "distribution_passed": all(item["distribution_passed"] for item in relevant if item["representative"])
        and mean_top5 >= args.min_top5_overlap,
        "mean_top5_overlap": mean_top5,
        "thresholds": {
            **thresholds,
            "min_mean_top5_overlap": args.min_top5_overlap,
        },
    }
    summary["passed"] = (
        summary["completed_requests"] == summary["expected_requests"]
        and summary["token_match"]
        and summary["distribution_passed"]
    )
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
