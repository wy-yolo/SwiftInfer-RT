# MiniLLM-RT

A compact C++17/TensorRT runtime for Qwen2.5-0.5B with separate prefill/decode
engines, request-scoped blocked KV cache, CUDA gather/scatter kernels and FCFS
continuous decode batching.

## Fixed scope

- Qwen2.5-0.5B-Instruct, FP16, greedy decoding, one GPU.
- Context up to 4096 tokens; block size 16; 8192 physical blocks.
- One-at-a-time prefill; decode batch 1-32; queue up to 64 requests.
- Blocked KV storage is gathered into dense TensorRT inputs. This is not a
  PagedAttention plugin.

## Environment

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate minillm-rt
python scripts/verify_environment.py
```

CUDA compilation must resolve to the environment's CUDA 12.9 compiler, never
`/usr/bin/nvcc` (CUDA 12.2). TensorRT C++ builds require `TENSORRT_ROOT`.

## Build and test the runtime core

```bash
cmake -S . -B build-core -G Ninja \
  -DMINILLM_WITH_TENSORRT=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-core -j
ctest --test-dir build-core --output-on-failure
```

Run Compute Sanitizer only on an otherwise idle GPU:

```bash
scripts/run_sanitizer.sh build-core
```

## Model pipeline

```bash
HTTP_PROXY=http://127.0.0.1:7897 HTTPS_PROXY=http://127.0.0.1:7897 \
  python python/download_model.py
python python/export_qwen.py
python python/validate_onnx.py
python python/validate_onnx_matrix.py
# Only after strict ONNX validation passes:
python python/build_engine.py --kind both --min-free-gb 28
```

TensorRT plans are GPU-specific and are intentionally excluded from Git.
The builder refuses to run when another compute process uses at least 64 MiB
of VRAM or when less than 28 GiB is free. Prefill and decode are built in
separate processes, and plans are published atomically with SHA256 metadata.

The strict numerical gate currently blocks engine construction. See
[`docs/numerical_gate.md`](docs/numerical_gate.md) for the diagnosis and exact
reproduction commands.

## Validation and experiments

```bash
python python/generate_validation_corpus.py
build-core/benchmark_kv_cache_cuda > results/kv_gather.csv
```

Formal benchmarks require no foreign compute process on the GPU and at least
28 GiB free VRAM. Each case uses 20 warm-up iterations and at least five
measured repetitions. Results must include TTFT, TPOT p50/p95, aggregate
tokens/s, per-request tokens/s, peak memory, utilization, power and temperature.
