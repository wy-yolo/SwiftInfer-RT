#!/usr/bin/env python3
"""Compare PyTorch and ONNX tensors inside one Qwen decoder layer."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForCausalLM

from export_diagnostics import LayerDetailPrefill


def metrics(reference: np.ndarray, candidate: np.ndarray, atol: float, rtol: float) -> dict:
    reference = reference.astype(np.float32)
    candidate = candidate.astype(np.float32)
    delta = np.abs(reference - candidate)
    within = delta <= atol + rtol * np.abs(candidate)
    return {
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "allclose": bool(np.allclose(reference, candidate, atol=atol, rtol=rtol)),
        "within_tolerance_fraction": float(within.mean()),
        "finite": bool(np.isfinite(reference).all() and np.isfinite(candidate).all()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("artifacts/model"))
    parser.add_argument("--onnx", type=Path, default=Path("artifacts/diagnostics/prefill_layer_0.onnx"))
    parser.add_argument("--output", type=Path, default=Path("results/diagnostics/layer_0.json"))
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--sequences", default="1,2,8,17")
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=5e-2)
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        attn_implementation="eager",
        local_files_only=True,
    ).eval().cuda()
    capture = LayerDetailPrefill(model, args.layer)
    session = ort.InferenceSession(
        str(args.onnx),
        providers=[("CUDAExecutionProvider", {"use_tf32": "0"}), "CPUExecutionProvider"],
    )
    cases = []
    for case_index, sequence in enumerate(int(item) for item in args.sequences.split(",")):
        generator = np.random.default_rng(4000 + case_index)
        ids_np = generator.integers(100, int(model.config.vocab_size) - 256, (1, sequence), dtype=np.int32)
        with torch.inference_mode():
            hf_logits = capture(
                torch.from_numpy(ids_np.astype(np.int64)).cuda(),
                torch.ones((1, sequence), dtype=torch.long, device="cuda"),
                torch.arange(sequence, dtype=torch.long, device="cuda").unsqueeze(0),
            )[0]
        outputs = session.run(
            None,
            {
                "input_ids": ids_np,
                "attention_mask": np.ones((1, sequence), dtype=np.int32),
                "position_ids": np.arange(sequence, dtype=np.int32)[None, :],
            },
        )
        comparisons = {
            "logits": metrics(hf_logits.float().cpu().numpy(), outputs[0], args.atol, args.rtol)
        }
        for index, name in enumerate(LayerDetailPrefill.CAPTURE_NAMES, start=1):
            comparisons[name] = metrics(
                capture.captured[name].float().cpu().numpy(), outputs[index], args.atol, args.rtol
            )
        order = [*LayerDetailPrefill.CAPTURE_NAMES, "logits"]
        first_divergent = next((name for name in order if not comparisons[name]["allclose"]), None)
        cases.append({"sequence": sequence, "first_divergent": first_divergent, "tensors": comparisons})
        print(
            json.dumps(
                {
                    "sequence": sequence,
                    "first_divergent": first_divergent,
                    "max_abs": {name: comparisons[name]["max_abs"] for name in order},
                }
            ),
            flush=True,
        )
    result = {"layer": args.layer, "tolerances": {"atol": args.atol, "rtol": args.rtol}, "cases": cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
