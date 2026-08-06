#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path

from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("artifacts/model"))
    parser.add_argument("--runtime", type=Path, default=Path("build-release/minillm_cli"))
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    ids = tokenizer(args.prompt, add_special_tokens=True).input_ids
    request = {
        "request_id": "cli-0",
        "input_ids": ids,
        "max_new_tokens": args.max_new_tokens,
    }
    process = subprocess.run(
        [str(args.runtime), "--validate-jsonl"],
        input=json.dumps(request) + "\n",
        text=True,
        capture_output=True,
        check=True,
    )
    print(process.stdout.strip())


if __name__ == "__main__":
    main()

