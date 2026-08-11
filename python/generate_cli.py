#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path

from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("artifacts/model"))
    parser.add_argument("--runtime", type=Path, default=Path("build-release/swiftinfer_cli"))
    parser.add_argument(
        "--prefill-engine",
        type=Path,
        default=Path("artifacts/engines/rtx5090/fp16/prefill.plan"),
    )
    parser.add_argument(
        "--decode-engine",
        type=Path,
        default=Path("artifacts/engines/rtx5090/fp16/decode.plan"),
    )
    parser.add_argument(
        "--model-spec",
        type=Path,
        default=Path("artifacts/onnx/fp16/model_spec.json"),
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    messages = [{"role": "user", "content": args.prompt}]
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    # Transformers 5 returns a BatchEncoding here, while Transformers 4
    # returned the token list directly.  Normalize both APIs to the JSONL
    # runtime's one-dimensional integer array.
    ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise RuntimeError("chat template unexpectedly returned a token batch")
        ids = ids[0]
    ids = [int(token) for token in ids]
    request = {
        "request_id": "cli-0",
        "input_ids": ids,
        "max_new_tokens": args.max_new_tokens,
    }
    process = subprocess.run(
        [
            str(args.runtime),
            "--prefill-engine",
            str(args.prefill_engine),
            "--decode-engine",
            str(args.decode_engine),
            "--model-spec",
            str(args.model_spec),
            "--generate-jsonl",
        ],
        input=json.dumps(request, ensure_ascii=False) + "\n",
        text=True,
        capture_output=True,
        check=False,
    )
    responses = [json.loads(line) for line in process.stdout.splitlines() if line.strip()]
    if not responses:
        raise RuntimeError(process.stderr.strip() or "runtime produced no JSONL response")
    response = responses[0]
    if response.get("finish_reason") == "error":
        raise RuntimeError(response.get("error", "runtime generation failed"))
    response["text"] = tokenizer.decode(response["output_ids"], skip_special_tokens=True)
    print(json.dumps(response, ensure_ascii=False, indent=2))
    if process.returncode != 0:
        raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
