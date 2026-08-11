# SwiftInfer-RT

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

The `minillm-rt` and `minillm-rt-5060` Conda environment names predate the
SwiftInfer-RT rename. They are intentionally retained so the committed
environment snapshots continue to describe the environments used for the
recorded experiments.

CUDA compilation must resolve to the environment's CUDA 12.9 compiler, never
`/usr/bin/nvcc` (CUDA 12.2). TensorRT C++ builds require `TENSORRT_ROOT`.

## Build and test the runtime core

```bash
cmake -S . -B build-gpu-release2 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_C_COMPILER="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc" \
  -DCMAKE_CXX_COMPILER="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++" \
  -DCMAKE_CUDA_COMPILER="$CONDA_PREFIX/bin/nvcc" \
  -DCMAKE_CUDA_HOST_COMPILER="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++" \
  -DTENSORRT_ROOT="$PWD/artifacts/tensorrt-sdk"
cmake --build build-gpu-release2 -j
ctest --test-dir build-gpu-release2 --output-on-failure
PYTHONPATH=python python -m pytest -q
```

The explicit CUDA host compiler is required. Allowing CUDA 12.9 to select the
system GCC 14 mixes incompatible system and Conda headers on this host.

Run Compute Sanitizer only on an otherwise idle GPU. The first command checks
the standalone CUDA kernels; the second checks the complete TensorRT runtime:

```bash
scripts/run_sanitizer.sh build-gpu-debug2
scripts/run_runtime_sanitizer.sh build-gpu-debug2
```

## Model pipeline

```bash
HTTP_PROXY=http://127.0.0.1:7897 HTTPS_PROXY=http://127.0.0.1:7897 \
  python python/download_model.py
python python/export_qwen.py --precision fp16 --device cuda:0
python python/rewrite_rope_lookup.py
python python/export_qwen.py --precision fp32 --device cuda:0
python python/validate_onnx.py
python python/validate_onnx_matrix.py
# Full production gate; requires an otherwise idle GPU.
python python/validate_corpus.py --resume --require-exclusive --min-free-gb 28
# Build in separate processes after the numerical gate passes.
python python/build_engine.py --kind prefill --min-free-gb 28 \
  --fp32-gemm-accumulation
python python/build_engine.py --kind decode --min-free-gb 28
```

TensorRT plans are GPU-specific and are intentionally excluded from Git.
The builder refuses to run when another compute process uses at least 64 MiB
of VRAM or when less than 28 GiB is free. Prefill and decode are built in
separate processes, and plans are published atomically with SHA256 metadata.

The FP32 reference gate, FP16 adjudication gate, six TensorRT profile endpoints
and the 1918-request C++ regression all pass. See
[`docs/numerical_gate.md`](docs/numerical_gate.md) for the exact policy and
recorded hashes.

## Validation and experiments

```bash
python python/generate_validation_corpus.py
PYTHONPATH=python python python/validate_runtime_corpus.py \
  --prefill-engine artifacts/engines/rtx5090/fp16/prefill.plan \
  --decode-engine artifacts/engines/rtx5090/fp16/decode.plan
PYTHONPATH=python python python/adjudicate_runtime_tokens.py

# Screenshot baseline plus the full 4K/active-batch/request-count matrix.
PYTHONPATH=python python python/benchmark_runtime.py --mode all
python benchmarks/analyze_results.py \
  results/benchmarks/rtx5090/raw.jsonl \
  --output results/benchmarks/rtx5090/summary.json
```

Formal benchmarks require no foreign compute process on the GPU and at least
28 GiB free VRAM. Each case uses 20 warm-up iterations and at least five
measured repetitions. Results must include TTFT, TPOT p50/p95, aggregate
tokens/s, per-request tokens/s, peak memory, utilization, power and temperature.
The benchmark driver keeps each C++ Engine loaded across runs, uses an explicit
flush marker between cases, verifies every generated sequence against the
validated corpus, and writes the measured JSONL atomically.

## Recorded RTX 5090 result

The formal screenshot case (prompt 256, output 32, 16 requests) passed both
project targets over five measured runs after 20 warm-ups:

- KV B1 reduced median TPOT from 5.375 ms (no-KV recomputation) to 3.035 ms,
  a 43.54% reduction. The bootstrap 95% TPOT-delta interval was
  [2.336, 2.345] ms.
- Dynamic B8 increased median aggregate throughput from 330.6 tok/s (KV B1)
  to 2034.9 tok/s, a 515.55% improvement. The bootstrap 95% throughput-delta
  interval was [1698.6, 1706.5] tok/s.

The extension matrix completed all 44 configurations and 220 measured runs.
All 32 comparable active-B8/B16/B32 throughput intervals were strictly above
their B1 baselines. Peak measured throughput was 4464.3 tok/s at prompt 256,
active B32, 32 total requests. Peak observed GPU memory was 7875 MiB and the
highest sampled temperature was 78 C. Machine-readable outputs are under
`results/benchmarks/rtx5090/`.

The complete runtime also passed memcheck, initcheck, racecheck, and synccheck
for short B1/B8/B32 plus a 3968-token B1 case. Every summary reported zero
errors or hazards; memcheck reported zero leaked bytes. Logs are under
`results/sanitizer/logs/`.

## Recorded RTX 5060 Laptop result

The same source commit and FP16 ONNX files were copied to WSL; the RTX 5090
plans were not reused. TensorRT 10.10 rebuilt reduced-profile plans with a
2 GiB workspace:

- Prefill B1 S=1/256/2048, SHA256
  `c38189ed78623d07c8941c72ced5b1947de4305a4e199184dbdaecdd6582c71b`;
- Decode B1/H1, B4/H256, B8/H2047, SHA256
  `627a0d20aadcaf6cc68193fbf0c26d4d8c810da9a0947dc5bfef2af7f04290c8`.

All six HF/ORT/TRT profile endpoints passed. A 48-request portability corpus
covering prompt 256/1024/2016 completed in C++, and the independent Python TRT
scheduler matched all 48 sequences token-for-token. The reduced benchmark
completed 18 configurations and 54 measured runs (five warm-ups, three
repetitions). All 12 active-B4/B8 throughput intervals were above their B1
baselines. Peak measured throughput was 707.7 tok/s at prompt 256, active B8,
16 total requests; peak GPU memory was 5455 MiB. Laptop temperature briefly
reached 86 C, so longer sustained experiments should use adequate cooling.

The RTX 5060 environment locks are in `configs/rtx5060/`. Machine-readable
engine metadata, validation reports, and benchmark results remain local under
the ignored `artifacts/` and `results/` directories.

## License

Licensed under the Apache License 2.0. See [`LICENSE`](LICENSE) and
[`NOTICE`](NOTICE).
