#!/usr/bin/env python3
import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path


def percentile(values, fraction):
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def bootstrap_delta(baseline, optimized, samples=5000, seed=202606):
    rng = random.Random(seed)
    deltas = []
    for _ in range(samples):
        left = [rng.choice(baseline) for _ in baseline]
        right = [rng.choice(optimized) for _ in optimized]
        deltas.append(statistics.median(left) - statistics.median(right))
    return percentile(deltas, 0.025), percentile(deltas, 0.975)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/summary.json"))
    args = parser.parse_args()
    groups = defaultdict(list)
    for line in args.input.read_text().splitlines():
        row = json.loads(line)
        key = (
            row["experiment"],
            int(row.get("prompt_tokens", 0)),
            int(row.get("active_batch", 0)),
            int(row.get("total_requests", 0)),
        )
        groups[key].append(row)
    summary = {}
    for key, rows in groups.items():
        name, prompt, active, total = key
        label = f"{name}:p{prompt}:a{active}:n{total}"
        tpot = [float(row["tpot_p50_ms"]) for row in rows]
        ttft = [float(row["ttft_p50_ms"]) for row in rows]
        throughput = [float(row["throughput_tok_s"]) for row in rows]
        summary[label] = {
            "runs": len(rows),
            "ttft_p50_ms": statistics.median(ttft),
            "ttft_p95_ms": percentile([float(row["ttft_p95_ms"]) for row in rows], 0.95),
            "tpot_p50_ms": statistics.median(tpot),
            "tpot_p95_ms": percentile([float(row["tpot_p95_ms"]) for row in rows], 0.95),
            "throughput_p50_tok_s": statistics.median(throughput),
            "throughput_p95_tok_s": percentile(throughput, 0.95),
        }
    no_kv_key = ("no_kv_b1", 256, 1, 16)
    kv_key = ("kv_b1", 256, 1, 16)
    dynamic_key = ("kv_dynamic_b8", 256, 8, 16)
    comparisons = {}
    if no_kv_key in groups and kv_key in groups:
        no_kv_tpot = [float(row["tpot_p50_ms"]) for row in groups[no_kv_key]]
        kv_tpot = [float(row["tpot_p50_ms"]) for row in groups[kv_key]]
        comparisons["kv_vs_no_kv_tpot_delta_95ci_ms"] = bootstrap_delta(
            no_kv_tpot,
            kv_tpot,
        )
        comparisons["kv_tpot_reduction_pct"] = 100 * (
            statistics.median(no_kv_tpot) - statistics.median(kv_tpot)
        ) / statistics.median(no_kv_tpot)
    if kv_key in groups and dynamic_key in groups:
        dynamic_throughput = [float(row["throughput_tok_s"]) for row in groups[dynamic_key]]
        b1_throughput = [float(row["throughput_tok_s"]) for row in groups[kv_key]]
        comparisons["dynamic_b8_vs_b1_throughput_delta_95ci_tok_s"] = bootstrap_delta(
            dynamic_throughput,
            b1_throughput,
        )
        comparisons["dynamic_b8_throughput_improvement_pct"] = 100 * (
            statistics.median(dynamic_throughput) - statistics.median(b1_throughput)
        ) / statistics.median(b1_throughput)

    matrix_comparisons = []
    candidate_keys = sorted(
        key
        for key in groups
        if key[0].startswith("kv_dynamic_b") and key[2] > 1
    )
    for candidate_key in candidate_keys:
        _, prompt, active, total = candidate_key
        baseline_key = ("kv_dynamic_b1", prompt, 1, total)
        if baseline_key not in groups:
            continue
        ci = bootstrap_delta(
            [float(row["throughput_tok_s"]) for row in groups[candidate_key]],
            [float(row["throughput_tok_s"]) for row in groups[baseline_key]],
        )
        matrix_comparisons.append(
            {
                "prompt_tokens": prompt,
                "active_batch": active,
                "total_requests": total,
                "throughput_delta_95ci_tok_s": ci,
                "passed": ci[0] > 0,
            }
        )
    if matrix_comparisons:
        comparisons["matrix_throughput_comparisons"] = matrix_comparisons
        comparisons["matrix_throughput_gate_passed"] = all(
            item["passed"] for item in matrix_comparisons
        )
    if comparisons:
        summary["comparisons"] = comparisons
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
