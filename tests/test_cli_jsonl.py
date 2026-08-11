import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLI = Path(os.environ.get("SWIFTINFER_CLI", ROOT / "build-gpu-release2" / "swiftinfer_cli"))
SPEC = ROOT / "artifacts" / "onnx" / "fp16_rope_lookup" / "model_spec.json"


def run_jsonl(records: list[dict]) -> list[dict]:
    process = subprocess.run(
        [str(CLI), "--validate-jsonl", "--model-spec", str(SPEC)],
        input="".join(json.dumps(record) + "\n" for record in records),
        text=True,
        capture_output=True,
        check=True,
    )
    return [json.loads(line) for line in process.stdout.splitlines()]


def test_valid_and_invalid_requests_remain_jsonl_parseable() -> None:
    responses = run_jsonl(
        [
            {"request_id": "ok", "input_ids": [1, 2, 3], "max_new_tokens": 2},
            {"request_id": "too-long", "input_ids": [1], "max_new_tokens": 4096},
            {"request_id": "bad-token", "input_ids": [151936], "max_new_tokens": 1},
        ]
    )
    assert responses[0] == {
        "request_id": "ok",
        "accepted": True,
        "input_length": 3,
        "max_new_tokens": 2,
    }
    assert responses[1]["request_id"] == "too-long"
    assert responses[1]["finish_reason"] == "error"
    assert responses[2]["request_id"] == "bad-token"
    assert responses[2]["finish_reason"] == "error"


def test_generation_requires_all_artifacts() -> None:
    process = subprocess.run(
        [str(CLI), "--generate-jsonl"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert process.returncode == 2
    assert "generation requires both engines" in process.stderr


def test_batch_limits_are_validated_before_generation() -> None:
    process = subprocess.run(
        [str(CLI), "--validate-jsonl", "--max-active", "33"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert process.returncode == 2
    assert "1 <= max-active <= 32" in process.stderr


def test_long_context_array_is_parsed_without_regex_stack_overflow() -> None:
    response = run_jsonl(
        [
            {
                "request_id": "long-context",
                "input_ids": [100 + index % 1000 for index in range(3968)],
                "max_new_tokens": 1,
            }
        ]
    )[0]
    assert response == {
        "request_id": "long-context",
        "accepted": True,
        "input_length": 3968,
        "max_new_tokens": 1,
    }
