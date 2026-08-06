#!/usr/bin/env python3
"""Summarize persisted validation JSONL without rerunning GPU inference."""

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-requests", type=int)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--max-nrmse", type=float, default=0.02)
    parser.add_argument("--min-top5-overlap", type=float, default=0.95)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line]
    expected = args.expected_requests if args.expected_requests is not None else len(rows)
    metrics = [
        metric
        for row in rows
        if row.get("representative")
        for metric in row.get("distribution", [])
    ]
    numeric_failures = [
        {
            "request_id": row["request_id"],
            "step": step,
            "nrmse": metric["nrmse_max"],
            "cosine": metric["cosine_similarity_min"],
            "top5_overlap": metric["top5_overlap_mean"],
        }
        for row in rows
        if row.get("representative")
        for step, metric in enumerate(row.get("distribution", []))
        if (
            not metric["finite"]
            or metric["cosine_similarity_min"] < args.min_cosine
            or metric["nrmse_max"] > args.max_nrmse
        )
    ]
    mismatch_rows = [row for row in rows if not row["token_match"]]
    mean_top5 = float(np.mean([metric["top5_overlap_mean"] for metric in metrics])) if metrics else 0.0
    report = {
        "schema_version": 1,
        "input": str(args.input),
        "expected_requests": expected,
        "completed_requests": len(rows),
        "token_mismatch_count": len(mismatch_rows),
        "representative_steps": len(metrics),
        "numeric_failure_count": len(numeric_failures),
        "nrmse_max": max((metric["nrmse_max"] for metric in metrics), default=None),
        "cosine_min": min((metric["cosine_similarity_min"] for metric in metrics), default=None),
        "mean_top5_overlap": mean_top5,
        "numeric_failures": numeric_failures,
        "thresholds": {
            "min_cosine": args.min_cosine,
            "max_nrmse": args.max_nrmse,
            "min_mean_top5_overlap": args.min_top5_overlap,
        },
    }
    report["passed"] = (
        len(rows) == expected
        and not mismatch_rows
        and not numeric_failures
        and mean_top5 >= args.min_top5_overlap
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "numeric_failures"}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
