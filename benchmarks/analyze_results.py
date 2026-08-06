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
        groups[row["experiment"]].append(row)
    summary = {}
    for name, rows in groups.items():
        tpot = [float(row["tpot_ms"]) for row in rows]
        throughput = [float(row["throughput_tok_s"]) for row in rows]
        summary[name] = {
            "runs": len(rows),
            "tpot_p50_ms": statistics.median(tpot),
            "tpot_p95_ms": percentile(tpot, 0.95),
            "throughput_p50_tok_s": statistics.median(throughput),
            "throughput_p95_tok_s": percentile(throughput, 0.95),
        }
    if "no_kv_b1" in groups and "kv_b1" in groups:
        ci = bootstrap_delta(
            [row["tpot_ms"] for row in groups["no_kv_b1"]],
            [row["tpot_ms"] for row in groups["kv_b1"]],
        )
        summary["kv_tpot_delta_95ci_ms"] = ci
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

