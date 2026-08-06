#!/usr/bin/env python3
"""Generate the deterministic reduced-profile RTX 5060 request corpus."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path, default=Path("artifacts/validation/requests.jsonl")
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("results/validation/cpp_runtime_corpus.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/validation/rtx5060_requests.jsonl"),
    )
    parser.add_argument("--requests-per-prompt", type=int, default=16)
    args = parser.parse_args()
    if args.requests_per_prompt < 1:
        parser.error("requests-per-prompt must be positive")

    reference = {
        row["request_id"]: row["output_ids"]
        for row in map(json.loads, args.reference.read_text().splitlines())
    }
    templates = {}
    for record in map(json.loads, args.source.read_text().splitlines()):
        prompt = len(record["input_ids"])
        if (
            prompt in (256, 1024, 2048)
            and record["max_new_tokens"] == 32
            and len(reference.get(record["request_id"], [])) == 32
            and prompt not in templates
        ):
            templates[prompt] = record["input_ids"]
    if templates.keys() != {256, 1024, 2048}:
        raise SystemExit(f"missing source templates: {sorted(templates)}")
    templates[2016] = templates.pop(2048)[:2016]

    records = []
    for prompt in (256, 1024, 2016):
        for index in range(args.requests_per_prompt):
            records.append(
                {
                    "request_id": f"rtx5060-p{prompt}-{index:02d}",
                    "input_ids": templates[prompt],
                    "max_new_tokens": 32,
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row) + "\n" for row in records))
    temporary.replace(args.output)
    print(json.dumps({"requests": len(records), "output": str(args.output)}))


if __name__ == "__main__":
    main()
