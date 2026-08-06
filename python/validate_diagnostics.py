#!/usr/bin/env python3
"""Locate the first hidden-state divergence in the analysis-only ONNX graph."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForCausalLM


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
    parser.add_argument("--onnx", type=Path, default=Path("artifacts/diagnostics/prefill_hidden.onnx"))
    parser.add_argument("--output", type=Path, default=Path("results/diagnostics/hidden_states.json"))
    parser.add_argument("--sequences", default="1,2,17")
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--disable-fp16-reduced-precision", action="store_true")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
        not args.disable_fp16_reduced_precision
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        attn_implementation="eager",
        local_files_only=True,
    ).eval().cuda()
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(args.onnx),
        sess_options=options,
        providers=[("CUDAExecutionProvider", {"use_tf32": "0"}), "CPUExecutionProvider"],
    )
    cases = []
    for case_index, sequence in enumerate(int(item) for item in args.sequences.split(",")):
        generator = np.random.default_rng(3000 + case_index)
        ids_np = generator.integers(100, int(model.config.vocab_size) - 256, (1, sequence), dtype=np.int32)
        ids = torch.from_numpy(ids_np.astype(np.int64)).cuda()
        mask = torch.ones((1, sequence), dtype=torch.long, device="cuda")
        positions = torch.arange(sequence, dtype=torch.long, device="cuda").unsqueeze(0)
        with torch.inference_mode():
            hf = model(
                input_ids=ids,
                attention_mask=mask,
                position_ids=positions,
                use_cache=False,
                output_hidden_states=True,
            )
        outputs = session.run(
            None,
            {
                "input_ids": ids_np,
                "attention_mask": np.ones((1, sequence), dtype=np.int32),
                "position_ids": np.arange(sequence, dtype=np.int32)[None, :],
            },
        )
        hidden = [
            metrics(reference.float().cpu().numpy(), candidate, args.atol, args.rtol)
            for reference, candidate in zip(hf.hidden_states, outputs[1:], strict=True)
        ]
        logits = metrics(hf.logits[:, -1].float().cpu().numpy(), outputs[0], args.atol, args.rtol)
        first_divergent = next((index for index, item in enumerate(hidden) if not item["allclose"]), None)

        # Recompute the head from each backend's final hidden state using the
        # exact PyTorch weight. This separates hidden-state drift from ONNX MatMul drift.
        weight = model.lm_head.weight
        ort_hidden = torch.from_numpy(outputs[-1][:, -1]).cuda()
        with torch.inference_mode():
            hf_head = torch.nn.functional.linear(hf.hidden_states[-1][:, -1], weight).float().cpu().numpy()
            ort_hidden_torch_head = torch.nn.functional.linear(ort_hidden, weight).float().cpu().numpy()
        head_from_ort_hidden = metrics(hf_head, ort_hidden_torch_head, args.atol, args.rtol)
        onnx_head_only = metrics(ort_hidden_torch_head, outputs[0], args.atol, args.rtol)
        cases.append(
            {
                "sequence": sequence,
                "logits": logits,
                "hidden_states": hidden,
                "first_divergent_hidden": first_divergent,
                "head_from_ort_hidden_vs_hf": head_from_ort_hidden,
                "onnx_head_vs_torch_head_same_hidden": onnx_head_only,
            }
        )
        print(
            json.dumps(
                {
                    "sequence": sequence,
                    "logits_max_abs": logits["max_abs"],
                    "first_divergent_hidden": first_divergent,
                    "hidden_drift_max_abs": head_from_ort_hidden["max_abs"],
                    "onnx_head_max_abs": onnx_head_only["max_abs"],
                }
            ),
            flush=True,
        )

    result = {
        "providers": session.get_providers(),
        "tolerances": {"atol": args.atol, "rtol": args.rtol},
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
