#!/usr/bin/env python3
"""Compare the C++ KV runtime with an independent Python TensorRT scheduler."""

from __future__ import annotations

import argparse
import gc
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from validate_engine import EngineRunner, gpu_gate


@dataclass
class State:
    record: dict
    output_ids: list[int]
    cache: list[tuple[np.ndarray, np.ndarray]] | None = None
    staged_token: int | None = None

    @property
    def request_id(self) -> str:
        return str(self.record["request_id"])

    @property
    def sequence_length(self) -> int:
        return len(self.record["input_ids"]) + len(self.output_ids)


def prefill_one(state: State, runner: EngineRunner, layers: int) -> None:
    ids = np.asarray([state.record["input_ids"]], dtype=np.int32)
    sequence = ids.shape[1]
    outputs = runner.run(
        {
            "input_ids": ids,
            "attention_mask": np.ones_like(ids, dtype=np.int32),
            "position_ids": np.arange(sequence, dtype=np.int32)[None, :],
        }
    )
    state.staged_token = int(outputs["logits"].argmax(axis=1)[0])
    state.cache = [
        (outputs[f"present_key_{layer}"], outputs[f"present_value_{layer}"])
        for layer in range(layers)
    ]


def decode_bucket(states: list[State], runner: EngineRunner, layers: int) -> list[int]:
    if not states:
        return []
    history = states[0].sequence_length - 1
    if any(state.sequence_length - 1 != history for state in states):
        raise RuntimeError("decode bucket contains mixed histories")
    feed = {
        "input_ids": np.asarray([[state.output_ids[-1]] for state in states], dtype=np.int32),
        "attention_mask": np.ones((len(states), history + 1), dtype=np.int32),
        "position_ids": np.full((len(states), 1), history, dtype=np.int32),
    }
    for layer in range(layers):
        feed[f"past_key_{layer}"] = np.concatenate(
            [state.cache[layer][0] for state in states], axis=0
        )
        feed[f"past_value_{layer}"] = np.concatenate(
            [state.cache[layer][1] for state in states], axis=0
        )
    outputs = runner.run(feed)
    for row, state in enumerate(states):
        state.cache = [
            (
                np.concatenate(
                    (state.cache[layer][0], outputs[f"present_key_{layer}"][row : row + 1]),
                    axis=2,
                ),
                np.concatenate(
                    (
                        state.cache[layer][1],
                        outputs[f"present_value_{layer}"][row : row + 1],
                    ),
                    axis=2,
                ),
            )
            for layer in range(layers)
        ]
    return outputs["logits"].argmax(axis=1).astype(int).tolist()


def run_chunk(
    records: list[dict],
    prefill: EngineRunner,
    decode: EngineRunner,
    layers: int,
    eos_token_id: int,
    max_active: int,
) -> dict[str, list[int]]:
    waiting = deque(State(record, []) for record in records)
    active: list[State] = []
    completed: dict[str, list[int]] = {}

    def admit() -> None:
        while waiting and len(active) < max_active:
            active.append(waiting.popleft())

    admit()
    while active:
        for state in active:
            if state.cache is None:
                prefill_one(state, prefill, layers)

        tokens: dict[str, int] = {}
        groups: dict[int, list[State]] = defaultdict(list)
        for state in active:
            if state.staged_token is not None:
                tokens[state.request_id] = state.staged_token
                state.staged_token = None
            else:
                groups[state.sequence_length - 1].append(state)
        for bucket in groups.values():
            for state, token in zip(
                bucket, decode_bucket(bucket, decode, layers), strict=True
            ):
                tokens[state.request_id] = token

        survivors = []
        for state in active:
            token = tokens[state.request_id]
            state.output_ids.append(token)
            finished = token == eos_token_id or len(state.output_ids) >= int(
                state.record["max_new_tokens"]
            )
            if finished:
                completed[state.request_id] = state.output_ids
            else:
                survivors.append(state)
        active = survivors
        admit()
    return completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, default=Path("artifacts/validation/requests.jsonl"))
    parser.add_argument(
        "--cpp-results", type=Path, default=Path("results/validation/cpp_runtime_corpus.jsonl")
    )
    parser.add_argument(
        "--prefill-engine",
        type=Path,
        default=Path("artifacts/engines/rtx5090/fp16_rope_lookup/prefill.plan"),
    )
    parser.add_argument(
        "--decode-engine",
        type=Path,
        default=Path("artifacts/engines/rtx5090/fp16_rope_lookup/decode.plan"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/validation/cpp_vs_python_trt.json")
    )
    parser.add_argument("--layers", type=int, default=24)
    parser.add_argument("--eos-token-id", type=int, default=151645)
    parser.add_argument("--max-active", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--min-free-gb", type=float, default=28.0)
    args = parser.parse_args()
    if args.max_active < 1 or args.chunk_size < args.max_active:
        parser.error("chunk size must be at least max active")

    gpu_gate(args.min_free_gb)
    records = [json.loads(line) for line in args.requests.read_text().splitlines() if line]
    if args.limit is not None:
        records = records[: args.limit]
    expected_cpp = {
        item["request_id"]: item["output_ids"]
        for item in (
            json.loads(line) for line in args.cpp_results.read_text().splitlines() if line
        )
    }
    prefill = EngineRunner(args.prefill_engine)
    decode = EngineRunner(args.decode_engine)
    python_outputs: dict[str, list[int]] = {}
    for start in range(0, len(records), args.chunk_size):
        chunk = records[start : start + args.chunk_size]
        python_outputs.update(
            run_chunk(
                chunk,
                prefill,
                decode,
                args.layers,
                args.eos_token_id,
                args.max_active,
            )
        )
        print(json.dumps({"progress": f"{min(start + len(chunk), len(records))}/{len(records)}"}), flush=True)

    mismatches = [
        {
            "request_id": record["request_id"],
            "cpp": expected_cpp.get(record["request_id"]),
            "python_trt": python_outputs.get(record["request_id"]),
        }
        for record in records
        if expected_cpp.get(record["request_id"]) != python_outputs.get(record["request_id"])
    ]
    report = {
        "passed": not mismatches,
        "requests": len(records),
        "exact_matches": len(records) - len(mismatches),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "mismatches"}, indent=2))
    del prefill, decode
    gc.collect()
    torch.cuda.empty_cache()
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
