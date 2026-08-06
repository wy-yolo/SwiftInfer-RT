#!/usr/bin/env python3
"""Build GPU-specific TensorRT engines with resource gates and atomic output."""

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import onnx
import tensorrt as trt


LOGGER = trt.Logger(trt.Logger.WARNING)


def shape_triplet(name: str, shape: tuple[int, ...], kind: str):
    if kind == "prefill":
        if name in {"input_ids", "attention_mask", "position_ids"}:
            return (1, 1), (1, 256), (1, 4096)
    else:
        if name in {"input_ids", "position_ids"}:
            return (1, 1), (8, 1), (32, 1)
        if name == "attention_mask":
            return (1, 2), (8, 257), (32, 4096)
        if name.startswith("past_") and len(shape) == 4:
            return (
                (1, shape[1], 1, shape[3]),
                (8, shape[1], 256, shape[3]),
                (32, shape[1], 4095, shape[3]),
            )
    raise ValueError(f"No profile rule for {kind} input {name} with shape {shape}")


def run_text(command: list[str]) -> str:
    return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()


def gpu_snapshot() -> dict:
    fields = "index,name,uuid,driver_version,memory.total,memory.used,memory.free,utilization.gpu"
    values = run_text(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits", "--id=0"]
    ).split(", ")
    return dict(zip(fields.split(","), values, strict=True))


def compute_processes() -> list[dict]:
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = run_text(command)
    except subprocess.CalledProcessError:
        return []
    processes = []
    for line in output.splitlines():
        if not line.strip():
            continue
        pid, name, used = line.split(", ", 2)
        processes.append({"pid": int(pid), "process_name": name, "used_memory_mib": int(used)})
    return processes


def enforce_gpu_gate(min_free_gb: float) -> dict:
    snapshot = gpu_snapshot()
    free_mib = int(snapshot["memory.free"])
    required_mib = int(min_free_gb * 1024)
    foreign = [
        process
        for process in compute_processes()
        if process["pid"] != os.getpid() and process["used_memory_mib"] >= 64
    ]
    if foreign:
        raise RuntimeError(f"GPU gate: foreign compute processes are active: {foreign}")
    if free_mib < required_mib:
        raise RuntimeError(f"GPU gate: {free_mib} MiB free, {required_mib} MiB required")
    return snapshot


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def onnx_artifacts(path: Path) -> list[dict]:
    model = onnx.load_model(path, load_external_data=False)
    paths = {path.resolve()}
    for initializer in model.graph.initializer:
        if initializer.data_location != onnx.TensorProto.EXTERNAL:
            continue
        entries = {entry.key: entry.value for entry in initializer.external_data}
        if "location" in entries:
            paths.add((path.parent / entries["location"]).resolve())
    return [
        {"path": str(artifact), "size": artifact.stat().st_size, "sha256": sha256(artifact)}
        for artifact in sorted(paths)
    ]


def source_commit() -> str | None:
    try:
        return run_text(["git", "rev-parse", "HEAD"])
    except subprocess.CalledProcessError:
        return None


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_bytes(payload)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build(
    onnx_path: Path,
    engine_path: Path,
    kind: str,
    workspace_gb: int,
    min_free_gb: float,
) -> None:
    gpu = enforce_gpu_gate(min_free_gb)
    artifacts = onnx_artifacts(onnx_path)
    builder = trt.Builder(LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, LOGGER)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT ONNX parse failed:\n{errors}")
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    profile = builder.create_optimization_profile()
    profile_metadata = {}
    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        static_shape = tuple(int(value) for value in tensor.shape)
        minimum, optimum, maximum = shape_triplet(tensor.name, static_shape, kind)
        profile_result = profile.set_shape(tensor.name, minimum, optimum, maximum)
        if profile_result is False:
            raise RuntimeError(f"failed to set profile for {tensor.name}")
        profile_metadata[tensor.name] = {
            "min": list(minimum),
            "opt": list(optimum),
            "max": list(maximum),
        }
    config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build returned null")
    payload = bytes(serialized)
    runtime = trt.Runtime(LOGGER)
    check_engine = runtime.deserialize_cuda_engine(payload)
    if check_engine is None:
        raise RuntimeError("new TensorRT engine failed immediate deserialization")
    if check_engine.num_optimization_profiles != 1:
        raise RuntimeError("unexpected optimization profile count")

    atomic_write(engine_path, payload)
    metadata = {
        "schema_version": 2,
        "kind": kind,
        "engine": str(engine_path.resolve()),
        "engine_size": engine_path.stat().st_size,
        "engine_sha256": sha256(engine_path),
        "onnx_artifacts": artifacts,
        "tensorrt": trt.__version__,
        "workspace_gb": workspace_gb,
        "precision": "FP16",
        "profile": profile_metadata,
        "gpu": gpu,
        "source_commit": source_commit(),
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write(
        engine_path.with_suffix(engine_path.suffix + ".json"),
        json.dumps(metadata, indent=2).encode("utf-8"),
    )
    print(json.dumps(metadata, indent=2))
    del check_engine, runtime, serialized, config, parser, network, builder
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-dir", type=Path, default=Path("artifacts/onnx"))
    parser.add_argument("--engine-dir", type=Path, default=Path("artifacts/engines/rtx5090"))
    parser.add_argument("--workspace-gb", type=int, default=8)
    parser.add_argument("--min-free-gb", type=float, default=28.0)
    parser.add_argument("--kind", choices=["prefill", "decode", "both"], default="both")
    args = parser.parse_args()

    if args.kind == "both":
        # Separate processes guarantee the first builder and CUDA context are
        # released before the second resource gate is evaluated.
        for kind in ("prefill", "decode"):
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--onnx-dir",
                    str(args.onnx_dir),
                    "--engine-dir",
                    str(args.engine_dir),
                    "--workspace-gb",
                    str(args.workspace_gb),
                    "--min-free-gb",
                    str(args.min_free_gb),
                    "--kind",
                    kind,
                ],
                check=True,
            )
        return
    build(
        args.onnx_dir / f"{args.kind}.onnx",
        args.engine_dir / f"{args.kind}.plan",
        args.kind,
        args.workspace_gb,
        args.min_free_gb,
    )


if __name__ == "__main__":
    main()
