#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/validation"))
    parser.add_argument("--seed", type=int, default=202606)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    prompt_lengths = [1, 15, 16, 17, 32, 128, 255, 256, 512, 1024, 2048, 3968]
    output_lengths = [1, 8, 32]
    batches = [1, 2, 4, 8, 16, 32]
    configs = []
    total_requests = 0
    request_path = args.output / "requests.jsonl"
    with request_path.open("w", encoding="utf-8") as handle:
        for task in range(182):
            request_count = 11 if task < 98 else 10
            prompt_length = prompt_lengths[task % len(prompt_lengths)]
            output_length = output_lengths[(task // len(prompt_lengths)) % len(output_lengths)]
            batch = batches[(task // (len(prompt_lengths) * len(output_lengths))) % len(batches)]
            config = {
                "task_id": task,
                "prompt_length": prompt_length,
                "output_length": output_length,
                "batch_limit": batch,
                "request_count": request_count,
            }
            configs.append(config)
            for index in range(request_count):
                token_ids = [rng.randrange(100, 50000) for _ in range(prompt_length)]
                record = {
                    "task_id": task,
                    "request_id": f"task-{task:03d}-{index:02d}",
                    "input_ids": token_ids,
                    "max_new_tokens": output_length,
                    "batch_limit": batch,
                }
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                total_requests += 1
    manifest = {
        "seed": args.seed,
        "task_count": len(configs),
        "request_count": total_requests,
        "configs": configs,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if len(configs) != 182 or total_requests != 1918:
        raise RuntimeError("validation corpus cardinality mismatch")
    print(json.dumps({"tasks": len(configs), "requests": total_requests}, indent=2))


if __name__ == "__main__":
    main()

