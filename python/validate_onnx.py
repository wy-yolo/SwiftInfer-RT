#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict:
    delta = np.abs(reference.astype(np.float32) - candidate.astype(np.float32))
    return {
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "reference_argmax": int(reference.argmax(axis=-1).reshape(-1)[0]),
        "candidate_argmax": int(candidate.argmax(axis=-1).reshape(-1)[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("artifacts/model"))
    parser.add_argument("--onnx", type=Path, default=Path("artifacts/onnx"))
    parser.add_argument("--prompt", default="Explain why KV cache helps autoregressive decoding.")
    parser.add_argument("--output", type=Path, default=Path("results/onnx_validation.json"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--provider", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, attn_implementation="eager", local_files_only=True
    ).eval().to(args.device)
    encoded = tokenizer(args.prompt, return_tensors="pt")
    input_ids = encoded.input_ids.to(dtype=torch.int32, device=args.device)
    attention_mask = encoded.attention_mask.to(dtype=torch.int32, device=args.device)
    position_ids = torch.arange(input_ids.shape[1], dtype=torch.int32, device=args.device).unsqueeze(0)
    with torch.no_grad():
        hf = model(input_ids=input_ids.long(), attention_mask=attention_mask.long(), use_cache=True)
    hf_logits = hf.logits[:, -1].float().cpu().numpy()

    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if args.provider == "cuda" else ["CPUExecutionProvider"])
    session = ort.InferenceSession(str(args.onnx / "prefill.onnx"), providers=providers)
    actual_providers = session.get_providers()
    feed = {
        "input_ids": input_ids.cpu().numpy(),
        "attention_mask": attention_mask.cpu().numpy(),
        "position_ids": position_ids.cpu().numpy(),
    }
    outputs = session.run(None, feed)
    next_token = int(hf_logits.argmax(axis=-1)[0])
    token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=args.device)
    next_mask = torch.ones((1, input_ids.shape[1] + 1), dtype=torch.long, device=args.device)
    next_position = torch.tensor([[input_ids.shape[1]]], dtype=torch.long, device=args.device)
    with torch.no_grad():
        hf_decode = model(
            input_ids=token_tensor,
            attention_mask=next_mask,
            position_ids=next_position,
            past_key_values=hf.past_key_values,
            use_cache=True,
        )
    decode_session = ort.InferenceSession(str(args.onnx / "decode.onnx"), providers=providers)
    decode_feed = {
        "input_ids": np.asarray([[next_token]], dtype=np.int32),
        "attention_mask": np.ones((1, input_ids.shape[1] + 1), dtype=np.int32),
        "position_ids": np.asarray([[input_ids.shape[1]]], dtype=np.int32),
    }
    for layer in range((len(outputs) - 1) // 2):
        decode_feed[f"past_key_{layer}"] = outputs[1 + 2 * layer]
        decode_feed[f"past_value_{layer}"] = outputs[2 + 2 * layer]
    decode_outputs = decode_session.run(None, decode_feed)
    result = {
        "providers": actual_providers,
        "prefill": metrics(hf_logits, outputs[0]),
        "decode": metrics(hf_decode.logits[:, -1].float().cpu().numpy(), decode_outputs[0]),
        "prompt_tokens": int(input_ids.shape[1]),
    }
    result["prefill"]["token_match"] = (
        result["prefill"]["reference_argmax"] == result["prefill"]["candidate_argmax"]
    )
    result["decode"]["token_match"] = (
        result["decode"]["reference_argmax"] == result["decode"]["candidate_argmax"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    provider_ok = args.provider == "cpu" or "CUDAExecutionProvider" in actual_providers
    if not provider_ok or not result["prefill"]["token_match"] or not result["decode"]["token_match"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
