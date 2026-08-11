# Numerical correctness gate

Status: **FP32, adjudicated FP16, TensorRT endpoints, and C++ corpus passed**.

## Acceptance policy

The FP32 path is the semantic reference. Every HF FP32 versus ORT CPU/CUDA
logits tensor must pass `atol=rtol=1e-3`; all logits and K/V tensors must be
finite and every greedy token must match.

FP16 is the production TensorRT path. It records allclose and NRMSE but uses
distribution and FP32-adjudication gates appropriate to near-tied FP16 logits:

- cosine similarity >= 0.999;
- aggregate top-5 overlap >= 95%;
- exact token agreement unless the production token is in the FP32 top-3 and
  no more than 0.125 logit below the FP32 top-1 token.

This policy was approved before the final engines and runtime were accepted;
failed requests are never discarded.

## FP32 reference result

HF FP32 was compared with ORT CPU and ORT CUDA (TF32 disabled) for Prefill B1
lengths 1/16/17/128/256 and Decode B1/B2/B4/B8 with history length 32. All 11
cases passed. The largest observed logits error was approximately `1.27e-4`,
and greedy tokens were 100% identical.

## FP16 ONNX result and RoPE rewrite

The initial 1918-request run had 66 HF FP16 versus ORT FP16 token differences.
They were near-tie numerical choices, and all passed the FP32 top-3/0.125
adjudication. Runtime `Sin`/`Cos` was nevertheless unstable at the 4096-token
profile endpoint. `python/rewrite_rope_lookup.py` generates the 4096 x 64 FP16
RoPE table on GPU (`rope_theta=1e6`), replaces runtime trigonometry with ONNX
`Gather`, and removes 34 dead nodes. The production models are in
`artifacts/onnx/fp16_rope_lookup/`; the FP32 references remain in
`artifacts/onnx/fp32/`.

## RTX 5090 TensorRT result

The production plans are:

- Prefill: `artifacts/engines/rtx5090/fp16/prefill.plan`, SHA256
  `053dd706a0db65f3331a8d97da258a88315019b65db52ba163049f81f65c2eaa`;
- Decode: `artifacts/engines/rtx5090/fp16/decode.plan`, SHA256
  `50b674cbbda37b9cfd5d59a8495c2786b69c56eee25ece9d81f73eb0efad585e`.

Prefill uses FP16 weights and outputs with FP32 GEMM accumulation. Decode uses
the normal FP16 path while keeping RMSNorm Pow/Reduce in FP32. Prefill
S=1/256/4096 and Decode B1/H1, B8/H256, B32/H4095 all deserialize, set dynamic
shapes, allocate buffers, enqueue successfully, and pass the production token
gate. Decode endpoints use real Prefill Engine KV rather than synthetic zeros.
The machine-readable result is
`results/validation/rtx5090_engine_final_gate.json`.

## C++ runtime result

The Release runtime is compiled with CUDA 12.9.86, Conda GCC 13.4 and
`compute_120,sm_120`. It completed 1918/1918 corpus requests with no errors.
An independent Python TensorRT scheduler matched the C++ output token-for-token
on all 1918 requests. Against the FP32 reference, 1781 requests were exact
without adjudication and 137/137 differences passed; maximum observed FP32 rank
was 3 and maximum gap was `0.11930465698242188`.

Relevant results:

- `results/validation/cpp_runtime_corpus.jsonl`;
- `results/validation/cpp_vs_python_trt_release2.json`;
- `results/validation/runtime_fp32_adjudication.jsonl`.

## Reproduction

```bash
conda activate minillm-rt

python python/validate_onnx_matrix.py --precision fp32 \
  --modes cpu,cuda_noopt,cuda \
  --output results/validation/fp32_matrix.json

python python/rewrite_rope_lookup.py

python python/validate_engine.py \
  --prefill-engine artifacts/engines/rtx5090/fp16/prefill.plan \
  --decode-engine artifacts/engines/rtx5090/fp16/decode.plan

LD_LIBRARY_PATH=artifacts/tensorrt-sdk/lib:$CONDA_PREFIX/lib \
  build-gpu-release2/swiftinfer_cli \
  --prefill-engine artifacts/engines/rtx5090/fp16/prefill.plan \
  --decode-engine artifacts/engines/rtx5090/fp16/decode.plan \
  --model-spec artifacts/onnx/fp16_rope_lookup/model_spec.json \
  --generate-jsonl < artifacts/validation/requests.jsonl \
  > results/validation/cpp_runtime_corpus.jsonl

PYTHONPATH=python python python/validate_runtime_corpus.py \
  --prefill-engine artifacts/engines/rtx5090/fp16/prefill.plan \
  --decode-engine artifacts/engines/rtx5090/fp16/decode.plan
PYTHONPATH=python python python/adjudicate_runtime_tokens.py
```

TensorRT plans remain GPU-specific. The RTX 5060 must rebuild both plans from
the same FP16 ONNX models using its reduced profiles; a 5090 plan must never be
copied as a portable artifact.
