#!/usr/bin/env python3
"""Run the reproducible RTX 5090 MiniLLM-RT benchmark matrix.

The production variants use one persistent C++ process per active-batch limit,
so TensorRT deserialization is outside the measurements.  The no-KV baseline
uses the same prefill engine and recomputes the complete sequence for every
generated token.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from validate_engine import EngineRunner, gpu_gate


TARGET_MATRIX = {
    "rtx5090": {
        "prompts": (256, 1024, 2048, 3968),
        "active": (1, 8, 16, 32),
        "totals": (16, 32, 64),
    },
    "rtx5060": {
        "prompts": (256, 1024, 2016),
        "active": (1, 4, 8),
        "totals": (8, 16),
    },
}


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return float(ordered[index])


def checkpoint(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row) + "\n" for row in rows))


class Telemetry:
    def __init__(self, interval: float = 0.1) -> None:
        self.interval = interval
        self.samples: list[tuple[float, float, float, float]] = []
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def __enter__(self) -> "Telemetry":
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()
        return self

    def _poll(self) -> None:
        command = [
            "nvidia-smi",
            "--query-gpu=memory.used,utilization.gpu,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
            "--id=0",
        ]
        while not self.stop_event.is_set():
            try:
                fields = subprocess.check_output(command, text=True).strip().split(", ")
                self.samples.append(tuple(float(value) for value in fields))
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            self.stop_event.wait(self.interval)

    def __exit__(self, *_: object) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2)

    def summary(self) -> dict[str, float | int | None]:
        if not self.samples:
            return {
                "telemetry_samples": 0,
                "gpu_memory_peak_mib": None,
                "gpu_utilization_mean_pct": None,
                "power_mean_w": None,
                "temperature_peak_c": None,
            }
        return {
            "telemetry_samples": len(self.samples),
            "gpu_memory_peak_mib": max(item[0] for item in self.samples),
            "gpu_utilization_mean_pct": statistics.fmean(item[1] for item in self.samples),
            "power_mean_w": statistics.fmean(item[2] for item in self.samples),
            "temperature_peak_c": max(item[3] for item in self.samples),
        }


@dataclass
class Template:
    input_ids: list[int]
    expected_output_ids: list[int]


def load_templates(requests_path: Path, results_path: Path) -> dict[int, Template]:
    outputs = {
        row["request_id"]: row["output_ids"]
        for row in map(json.loads, results_path.read_text().splitlines())
    }
    templates: dict[int, Template] = {}
    for record in map(json.loads, requests_path.read_text().splitlines()):
        prompt = len(record["input_ids"])
        output = outputs.get(record["request_id"], [])
        if record["max_new_tokens"] == 32 and len(output) == 32 and prompt not in templates:
            templates[prompt] = Template(record["input_ids"], output)
    return templates


class PersistentRuntime:
    def __init__(self, args: argparse.Namespace, max_active: int) -> None:
        environment = os.environ.copy()
        sdk_lib = str(args.tensorrt_root / "lib")
        environment["LD_LIBRARY_PATH"] = sdk_lib + ":" + environment.get(
            "LD_LIBRARY_PATH", ""
        )
        command = [
            str(args.binary),
            "--prefill-engine",
            str(args.prefill_engine),
            "--decode-engine",
            str(args.decode_engine),
            "--model-spec",
            str(args.model_spec),
            "--generate-jsonl",
            "--max-active",
            str(max_active),
            "--max-total",
            "64",
            "--flush-on-empty-line",
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env=environment,
        )
        self.sequence = 0

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        return_code = self.process.wait(timeout=30)
        if return_code != 0:
            raise RuntimeError(f"runtime exited with {return_code}")

    def run(self, template: Template, total_requests: int) -> tuple[list[dict], float, dict]:
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("runtime pipes are unavailable")
        records = []
        for request_index in range(total_requests):
            request_id = f"bench-{self.sequence:08d}-{request_index:03d}"
            records.append(
                {
                    "request_id": request_id,
                    "input_ids": template.input_ids,
                    "max_new_tokens": 32,
                }
            )
        self.sequence += 1
        with Telemetry() as telemetry:
            begin = time.perf_counter()
            for record in records:
                self.process.stdin.write(json.dumps(record, separators=(",", ":")) + "\n")
            self.process.stdin.write("\n")
            self.process.stdin.flush()
            responses = [json.loads(self.process.stdout.readline()) for _ in records]
            elapsed = time.perf_counter() - begin
        for response in responses:
            if response.get("finish_reason") == "error":
                raise RuntimeError(response.get("error", "runtime error"))
            if response["output_ids"] != template.expected_output_ids:
                raise RuntimeError(f"token mismatch for {response['request_id']}")
        return responses, elapsed, telemetry.summary()


def summarize_run(
    experiment: str,
    prompt: int,
    active: int,
    total: int,
    repeat: int,
    responses: list[dict],
    elapsed: float,
    telemetry: dict,
) -> dict:
    ttft = [float(row["ttft_ms"]) for row in responses]
    tpot = [float(row["tpot_ms"]) for row in responses]
    tokens = sum(len(row["output_ids"]) for row in responses)
    per_request = [
        len(row["output_ids"])
        / max(1e-9, (row["ttft_ms"] + max(0, len(row["output_ids"]) - 1) * row["tpot_ms"]) / 1000)
        for row in responses
    ]
    return {
        "schema_version": 1,
        "experiment": experiment,
        "prompt_tokens": prompt,
        "output_tokens": 32,
        "active_batch": active,
        "total_requests": total,
        "repeat": repeat,
        "ttft_p50_ms": statistics.median(ttft),
        "ttft_p95_ms": percentile(ttft, 0.95),
        "tpot_p50_ms": statistics.median(tpot),
        "tpot_p95_ms": percentile(tpot, 0.95),
        "throughput_tok_s": tokens / elapsed,
        "per_request_tok_s_p50": statistics.median(per_request),
        "per_request_tok_s_p95": percentile(per_request, 0.95),
        "wall_time_s": elapsed,
        **telemetry,
    }


def run_cpp_group(
    runtime: PersistentRuntime,
    template: Template,
    experiment: str,
    prompt: int,
    active: int,
    total: int,
    warmup: int,
    repeats: int,
) -> list[dict]:
    measured = []
    for iteration in range(warmup + repeats):
        responses, elapsed, telemetry = runtime.run(template, total)
        if iteration >= warmup:
            measured.append(
                summarize_run(
                    experiment,
                    prompt,
                    active,
                    total,
                    iteration - warmup,
                    responses,
                    elapsed,
                    telemetry,
                )
            )
    return measured


def no_kv_request(
    runner: EngineRunner, template: Template, eos_token_id: int
) -> tuple[list[int], float, float]:
    sequence = list(template.input_ids)
    output: list[int] = []
    begin = time.perf_counter()
    first = 0.0
    for _ in range(32):
        ids = np.asarray([sequence], dtype=np.int32)
        length = ids.shape[1]
        result = runner.run(
            {
                "input_ids": ids,
                "attention_mask": np.ones_like(ids, dtype=np.int32),
                "position_ids": np.arange(length, dtype=np.int32)[None, :],
            }
        )
        token = int(result["logits"].argmax(axis=1)[0])
        output.append(token)
        sequence.append(token)
        if len(output) == 1:
            first = time.perf_counter()
        if token == eos_token_id:
            break
    end = time.perf_counter()
    ttft = (first - begin) * 1000
    tpot = 0.0 if len(output) <= 1 else (end - first) * 1000 / (len(output) - 1)
    return output, ttft, tpot


def run_no_kv_group(
    runner: EngineRunner,
    template: Template,
    warmup: int,
    repeats: int,
    total: int = 16,
) -> list[dict]:
    rows = []
    stream = torch.cuda.Stream()
    for iteration in range(warmup + repeats):
        responses = []
        with Telemetry() as telemetry:
            begin = time.perf_counter()
            with torch.cuda.stream(stream):
                for request_index in range(total):
                    output, ttft, tpot = no_kv_request(runner, template, 151645)
                    if output != template.expected_output_ids:
                        raise RuntimeError(f"no-KV token mismatch for request {request_index}")
                    responses.append(
                        {"output_ids": output, "ttft_ms": ttft, "tpot_ms": tpot}
                    )
            elapsed = time.perf_counter() - begin
        if iteration >= warmup:
            rows.append(
                summarize_run(
                    "no_kv_b1",
                    256,
                    1,
                    total,
                    iteration - warmup,
                    responses,
                    elapsed,
                    telemetry.summary(),
                )
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path("build-gpu-release2/minillm_cli"))
    parser.add_argument("--prefill-engine", type=Path, default=Path("artifacts/engines/rtx5090/fp16/prefill.plan"))
    parser.add_argument("--decode-engine", type=Path, default=Path("artifacts/engines/rtx5090/fp16/decode.plan"))
    parser.add_argument("--model-spec", type=Path, default=Path("artifacts/onnx/fp16_rope_lookup/model_spec.json"))
    parser.add_argument("--tensorrt-root", type=Path, default=Path("artifacts/tensorrt-sdk"))
    parser.add_argument("--requests", type=Path, default=Path("artifacts/validation/requests.jsonl"))
    parser.add_argument("--reference", type=Path, default=Path("results/validation/cpp_runtime_corpus.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/benchmarks/rtx5090/raw.jsonl"))
    parser.add_argument("--mode", choices=("screenshot", "matrix", "all"), default="all")
    parser.add_argument("--target", choices=sorted(TARGET_MATRIX), default="rtx5090")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--min-free-gb", type=float, default=28.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        parser.error("warmup must be non-negative and repeats must be positive")
    gpu_gate(args.min_free_gb)
    templates = load_templates(args.requests, args.reference)
    if args.target == "rtx5060" and args.mode != "matrix":
        parser.error("rtx5060 supports the reduced matrix mode only")
    matrix = TARGET_MATRIX[args.target]
    required = set(matrix["prompts"])
    if missing := required - templates.keys():
        raise SystemExit(f"missing stable 32-token templates: {sorted(missing)}")

    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    rows: list[dict] = []
    if args.resume and temporary.exists():
        rows = [json.loads(line) for line in temporary.read_text().splitlines() if line]

    def complete(experiment: str, prompt: int, active: int, total: int) -> bool:
        return sum(
            row["experiment"] == experiment
            and row["prompt_tokens"] == prompt
            and row["active_batch"] == active
            and row["total_requests"] == total
            for row in rows
        ) >= args.repeats

    groups_done = 0
    if args.mode in ("screenshot", "all"):
        if not complete("no_kv_b1", 256, 1, 16):
            prefill = EngineRunner(args.prefill_engine)
            rows.extend(run_no_kv_group(prefill, templates[256], args.warmup, args.repeats))
            del prefill
            gc.collect()
            torch.cuda.empty_cache()
            checkpoint(rows, args.output)
        for active, name in ((1, "kv_b1"), (8, "kv_dynamic_b8")):
            if complete(name, 256, active, 16):
                continue
            runtime = PersistentRuntime(args, active)
            rows.extend(
                run_cpp_group(runtime, templates[256], name, 256, active, 16, args.warmup, args.repeats)
            )
            runtime.close()
            checkpoint(rows, args.output)

    if args.mode in ("matrix", "all"):
        for active in matrix["active"]:
            runtime = PersistentRuntime(args, active)
            totals = [total for total in matrix["totals"] if total >= active]
            for prompt in matrix["prompts"]:
                for total in totals:
                    experiment = f"kv_dynamic_b{active}"
                    if complete(experiment, prompt, active, total):
                        groups_done += 1
                        continue
                    rows.extend(
                        run_cpp_group(
                            runtime,
                            templates[prompt],
                            experiment,
                            prompt,
                            active,
                            total,
                            args.warmup,
                            args.repeats,
                        )
                    )
                    groups_done += 1
                    checkpoint(rows, args.output)
                    print(
                        json.dumps(
                            {
                                "progress_groups": groups_done,
                                "experiment": experiment,
                                "prompt_tokens": prompt,
                                "total_requests": total,
                            }
                        ),
                        flush=True,
                    )
            runtime.close()

    checkpoint(rows, args.output)
    temporary.replace(args.output)
    print(json.dumps({"passed": True, "measured_rows": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
