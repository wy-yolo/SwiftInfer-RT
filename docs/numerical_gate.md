# ONNX strict numerical gate

Status: **blocked before TensorRT engine construction**.

The required gate is `numpy.allclose(atol=0.05, rtol=0.05)` for every logits
tensor plus 100% greedy-token equality. No tolerance was relaxed. All observed
KV tensors were finite and the tested greedy tokens matched, but at least one
logits tensor failed strict allclose in every provider configuration.

## Diagnosis

- CUDA, CUDA with graph optimization disabled, and CPU execution all reproduce
  the failure, so it is not isolated to a CUDA graph fusion or TF32. TF32 is
  disabled explicitly in both PyTorch and ORT.
- Decode was tested twice: once with identical HF-produced KV inputs and once
  end-to-end with ORT-produced KV. Isolated Decode can fail independently of
  Prefill error propagation.
- A diagnostic ONNX graph exposed all 25 hidden states. Representative first
  divergent outputs were hidden 24 at sequence 1, hidden 1 at sequence 2,
  hidden 10 at sequence 8, and hidden 12 at sequence 17. The exact layer varies
  with the input, consistent with accumulated FP16 rounding rather than one
  fixed semantic break.
- With the same final hidden tensor, ONNX and PyTorch LM-head outputs differed
  by at most 0.0039-0.0078. The larger logits difference therefore originates
  in accumulated hidden-state drift, not the LM head.
- Layer-0 probes show small Q/K/V, attention, and MLP differences. For sequence
  17, the attention output max absolute difference was about 0.0019; later
  normalization and projection operations amplify accumulated drift.
- Promoting all 217 FP16 MatMul nodes in the diagnostic graph to FP32 inputs and
  casting outputs back to FP16 did not reliably satisfy the gate. It was not
  applied to the production ONNX files.

## Reproduction

```bash
conda activate minillm-rt

# Formal matrix: B1 prefill lengths 1/16/17/128/256, decode B1/B2/B4/B8,
# two natural-language prompts, and three ORT execution modes.
python python/validate_onnx_matrix.py \
  --modes cuda,cuda_noopt,cpu \
  --output results/diagnostics/onnx_matrix_strict.json

# Analysis-only hidden-state and layer probes.
python python/export_diagnostics.py
python python/validate_diagnostics.py
python python/export_diagnostics.py --detail-layer 0
python python/validate_layer_detail.py
```

Generated ONNX diagnostics stay under `artifacts/diagnostics/`, and JSON results
stay under `results/diagnostics/`; both directories are intentionally ignored
by Git. The production Prefill and Decode ONNX files were not overwritten.

## Engine gate

`python/build_engine.py` checks the GPU before each engine, rejects foreign
compute processes using at least 64 MiB, requires at least 28 GiB free VRAM,
builds Prefill and Decode in separate processes, validates immediate engine
deserialization, and writes the plan and metadata atomically. It must remain
unused until the strict numerical gate is resolved or the acceptance policy is
explicitly changed.
