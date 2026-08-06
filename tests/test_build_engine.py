import os
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import build_engine  # noqa: E402


def test_prefill_profile():
    assert build_engine.shape_triplet("input_ids", (-1, -1), "prefill") == (
        (1, 1),
        (1, 256),
        (1, 4096),
    )


def test_decode_kv_profile():
    assert build_engine.shape_triplet("past_key_0", (-1, 2, -1, 64), "decode") == (
        (1, 2, 1, 64),
        (8, 2, 256, 64),
        (32, 2, 4095, 64),
    )


def test_gpu_gate_rejects_foreign_process(monkeypatch):
    monkeypatch.setattr(build_engine, "gpu_snapshot", lambda: {"memory.free": "31300"})
    monkeypatch.setattr(
        build_engine,
        "compute_processes",
        lambda: [{"pid": os.getpid() + 1, "process_name": "python", "used_memory_mib": 1024}],
    )
    with pytest.raises(RuntimeError, match="foreign compute"):
        build_engine.enforce_gpu_gate(28)


def test_gpu_gate_rejects_low_free_memory(monkeypatch):
    monkeypatch.setattr(build_engine, "gpu_snapshot", lambda: {"memory.free": "20000"})
    monkeypatch.setattr(build_engine, "compute_processes", lambda: [])
    with pytest.raises(RuntimeError, match="MiB free"):
        build_engine.enforce_gpu_gate(28)


def test_atomic_write_removes_temporary_file(tmp_path):
    destination = tmp_path / "prefill.plan"
    build_engine.atomic_write(destination, b"engine")
    assert destination.read_bytes() == b"engine"
    assert not destination.with_name("prefill.plan.tmp").exists()
