#!/usr/bin/env python3
"""Strict HF -> ORT validation for MiniLLM-RT prefill and decode graphs."""

import argparse
import gc
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


DEFAULT_PROMPTS = (
    "Explain why KV cache helps autoregressive decoding.",
    "用一句话说明连续批处理为什么能提高推理吞吐量。",
)


def parse_csv_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return values


def parse_modes(value: str) -> list[str]:
    modes = [item.strip() for item in value.split(",") if item.strip()]
    valid = {"cpu", "cuda_noopt", "cuda"}
    if not modes or any(mode not in valid for mode in modes):
        raise argparse.ArgumentTypeError("modes must be cpu,cuda_noopt,cuda")
    return modes


def gpu_free_mib() -> int:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
            "--id=0",
        ],
        text=True,
    )
    return int(output.strip().splitlines()[0])


def legacy_cache(cache) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    if hasattr(cache, "to_legacy_cache"):
        cache = cache.to_legacy_cache()
    return tuple((layer[0], layer[1]) for layer in cache)


def dynamic_cache(pairs: Sequence[tuple[torch.Tensor, torch.Tensor]], config):
    try:
        from transformers.cache_utils import DynamicCache

        return DynamicCache(ddp_cache_data=tuple(pairs), config=config)
    except Exception:
        return tuple(pairs)


def tensor_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    atol: float,
    rtol: float,
    include_tokens: bool = False,
) -> dict:
    reference_fp32 = reference.astype(np.float32)
    candidate_fp32 = candidate.astype(np.float32)
    finite = bool(np.isfinite(reference_fp32).all() and np.isfinite(candidate_fp32).all())
    delta = np.abs(reference_fp32 - candidate_fp32)
    tolerance = atol + rtol * np.abs(candidate_fp32)
    within = delta <= tolerance
    result = {
        "shape": list(reference.shape),
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "allclose": bool(finite and np.allclose(reference_fp32, candidate_fp32, atol=atol, rtol=rtol)),
        "within_tolerance_fraction": float(within.mean()),
        "finite": finite,
    }
    if include_tokens:
        reference_rows = reference_fp32.reshape(reference_fp32.shape[0], -1)
        candidate_rows = candidate_fp32.reshape(candidate_fp32.shape[0], -1)
        dot = np.sum(reference_rows * candidate_rows, axis=1)
        denominator = np.linalg.norm(reference_rows, axis=1) * np.linalg.norm(candidate_rows, axis=1)
        cosine = np.divide(dot, denominator, out=np.ones_like(dot), where=denominator != 0)
        rmse = np.sqrt(np.mean(np.square(reference_rows - candidate_rows), axis=1))
        reference_rms = np.sqrt(np.mean(np.square(reference_rows), axis=1))
        nrmse = np.divide(rmse, reference_rms, out=np.zeros_like(rmse), where=reference_rms != 0)
        reference_tokens = reference_fp32.argmax(axis=-1).reshape(-1)
        candidate_tokens = candidate_fp32.argmax(axis=-1).reshape(-1)
        reference_top5 = np.argpartition(reference_rows, -5, axis=1)[:, -5:]
        candidate_top5 = np.argpartition(candidate_rows, -5, axis=1)[:, -5:]
        top5_overlap = np.asarray(
            [len(set(left.tolist()) & set(right.tolist())) / 5.0
             for left, right in zip(reference_top5, candidate_top5, strict=True)],
            dtype=np.float32,
        )
        result.update(
            {
                "token_match": bool(np.array_equal(reference_tokens, candidate_tokens)),
                "reference_tokens": reference_tokens.astype(int).tolist(),
                "candidate_tokens": candidate_tokens.astype(int).tolist(),
                "cosine_similarity_min": float(cosine.min()),
                "cosine_similarity_mean": float(cosine.mean()),
                "nrmse_max": float(nrmse.max()),
                "nrmse_mean": float(nrmse.mean()),
                "top5_overlap_min": float(top5_overlap.min()),
                "top5_overlap_mean": float(top5_overlap.mean()),
            }
        )
    return result


def kv_metrics(
    reference: Sequence[tuple[np.ndarray, np.ndarray]],
    candidate: Sequence[tuple[np.ndarray, np.ndarray]],
    atol: float,
    rtol: float,
) -> dict:
    layers = []
    first_divergent = None
    finite = True
    for index, ((ref_key, ref_value), (cand_key, cand_value)) in enumerate(
        zip(reference, candidate, strict=True)
    ):
        key = tensor_metrics(ref_key, cand_key, atol, rtol)
        value = tensor_metrics(ref_value, cand_value, atol, rtol)
        layer_finite = key["finite"] and value["finite"]
        finite = finite and layer_finite
        if first_divergent is None and (not key["allclose"] or not value["allclose"]):
            first_divergent = index
        layers.append({"layer": index, "key": key, "value": value})
    return {
        "finite": finite,
        "allclose": all(
            layer[part]["allclose"] for layer in layers for part in ("key", "value")
        ),
        "first_divergent_layer": first_divergent,
        "layers": layers,
    }


def make_random_tokens(batch: int, sequence: int, vocab_size: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return generator.integers(100, vocab_size - 256, size=(batch, sequence), dtype=np.int32)


def session_for(mode: str, model_path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if mode == "cuda_noopt"
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    providers = (
        ["CPUExecutionProvider"]
        if mode == "cpu"
        else [("CUDAExecutionProvider", {"use_tf32": "0"}), "CPUExecutionProvider"]
    )
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=providers)
    if mode != "cpu" and "CUDAExecutionProvider" not in session.get_providers():
        raise RuntimeError("CUDAExecutionProvider is unavailable")
    return session


def ort_cache(outputs: Sequence[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(outputs[1 + 2 * layer], outputs[2 + 2 * layer]) for layer in range((len(outputs) - 1) // 2)]


def run_case(
    *,
    case_name: str,
    ids_np: np.ndarray,
    model,
    config,
    prefill_session: ort.InferenceSession,
    decode_session: ort.InferenceSession,
    atol: float,
    rtol: float,
) -> dict:
    batch, sequence = ids_np.shape
    hf_logits_rows: list[np.ndarray] = []
    hf_cache_rows: list[tuple[tuple[torch.Tensor, torch.Tensor], ...]] = []
    ort_prefill_rows: list[list[np.ndarray]] = []

    # Prefill is a B=1 graph. Keep the reference B=1 too, even when preparing a
    # multi-request decode batch, so backend batch algorithms cannot confound it.
    for row in range(batch):
        ids = torch.from_numpy(ids_np[row : row + 1].astype(np.int64)).cuda()
        mask = torch.ones((1, sequence), dtype=torch.long, device="cuda")
        positions = torch.arange(sequence, dtype=torch.long, device="cuda").unsqueeze(0)
        with torch.inference_mode():
            hf = model(
                input_ids=ids,
                attention_mask=mask,
                position_ids=positions,
                use_cache=True,
            )
        hf_logits_rows.append(hf.logits[:, -1].float().cpu().numpy())
        hf_cache_rows.append(tuple((key.detach(), value.detach()) for key, value in legacy_cache(hf.past_key_values)))
        ort_prefill_rows.append(
            prefill_session.run(
                None,
                {
                    "input_ids": ids_np[row : row + 1].astype(np.int32),
                    "attention_mask": np.ones((1, sequence), dtype=np.int32),
                    "position_ids": np.arange(sequence, dtype=np.int32)[None, :],
                },
            )
        )
        del ids, mask, positions, hf

    hf_prefill_logits = np.concatenate(hf_logits_rows, axis=0)
    ort_prefill_logits = np.concatenate([outputs[0] for outputs in ort_prefill_rows], axis=0)
    hf_pairs = [
        (
            torch.cat([cache[layer][0] for cache in hf_cache_rows], dim=0),
            torch.cat([cache[layer][1] for cache in hf_cache_rows], dim=0),
        )
        for layer in range(len(hf_cache_rows[0]))
    ]
    hf_pairs_np = [(key.cpu().numpy(), value.cpu().numpy()) for key, value in hf_pairs]
    ort_pairs_np = [
        (
            np.concatenate([outputs[1 + 2 * layer] for outputs in ort_prefill_rows], axis=0),
            np.concatenate([outputs[2 + 2 * layer] for outputs in ort_prefill_rows], axis=0),
        )
        for layer in range(len(hf_pairs))
    ]
    next_tokens = hf_prefill_logits.argmax(axis=-1).astype(np.int32)

    decode_common = {
        "input_ids": next_tokens[:, None],
        "attention_mask": np.ones((batch, sequence + 1), dtype=np.int32),
        "position_ids": np.full((batch, 1), sequence, dtype=np.int32),
    }
    isolated_feed = dict(decode_common)
    end_to_end_feed = dict(decode_common)
    for layer, ((hf_key, hf_value), (ort_key, ort_value)) in enumerate(
        zip(hf_pairs_np, ort_pairs_np, strict=True)
    ):
        isolated_feed[f"past_key_{layer}"] = hf_key
        isolated_feed[f"past_value_{layer}"] = hf_value
        end_to_end_feed[f"past_key_{layer}"] = ort_key
        end_to_end_feed[f"past_value_{layer}"] = ort_value

    isolated_outputs = decode_session.run(None, isolated_feed)
    end_to_end_outputs = decode_session.run(None, end_to_end_feed)
    hf_decode_cache = dynamic_cache([(key.clone(), value.clone()) for key, value in hf_pairs], config)
    with torch.inference_mode():
        hf_decode = model(
            input_ids=torch.from_numpy(next_tokens[:, None].astype(np.int64)).cuda(),
            attention_mask=torch.ones((batch, sequence + 1), dtype=torch.long, device="cuda"),
            position_ids=torch.full((batch, 1), sequence, dtype=torch.long, device="cuda"),
            past_key_values=hf_decode_cache,
            use_cache=True,
        )
    hf_decode_logits = hf_decode.logits[:, -1].float().cpu().numpy()
    hf_new_pairs = [
        (key[..., -1:, :].float().cpu().numpy(), value[..., -1:, :].float().cpu().numpy())
        for key, value in legacy_cache(hf_decode.past_key_values)
    ]

    prefill_logits = tensor_metrics(hf_prefill_logits, ort_prefill_logits, atol, rtol, True)
    isolated_logits = tensor_metrics(hf_decode_logits, isolated_outputs[0], atol, rtol, True)
    end_to_end_logits = tensor_metrics(hf_decode_logits, end_to_end_outputs[0], atol, rtol, True)
    prefill_kv = kv_metrics(hf_pairs_np, ort_pairs_np, atol, rtol)
    isolated_kv = kv_metrics(hf_new_pairs, ort_cache(isolated_outputs), atol, rtol)
    end_to_end_kv = kv_metrics(hf_new_pairs, ort_cache(end_to_end_outputs), atol, rtol)
    logits = (prefill_logits, isolated_logits, end_to_end_logits)
    kv = (prefill_kv, isolated_kv, end_to_end_kv)
    result = {
        "name": case_name,
        "batch": batch,
        "sequence": sequence,
        "prefill": {"logits": prefill_logits, "kv": prefill_kv},
        "decode_isolated": {"logits": isolated_logits, "kv": isolated_kv},
        "decode_end_to_end": {"logits": end_to_end_logits, "kv": end_to_end_kv},
        "semantic_passed": all(metric["token_match"] for metric in logits),
        "numeric_passed": all(metric["allclose"] for metric in logits),
        "kv_finite": all(metric["finite"] for metric in kv),
    }
    result["passed"] = result["semantic_passed"] and result["numeric_passed"] and result["kv_finite"]
    del hf_pairs, hf_decode_cache, hf_decode, isolated_outputs, end_to_end_outputs
    torch.cuda.empty_cache()
    result["gpu_free_mib_post_case"] = gpu_free_mib()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("artifacts/model"))
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--lengths", type=parse_csv_ints, default=parse_csv_ints("1,16,17,128,256"))
    parser.add_argument("--decode-batches", type=parse_csv_ints, default=parse_csv_ints("1,2,4,8"))
    parser.add_argument("--batch-sequence", type=int, default=32)
    parser.add_argument("--modes", type=parse_modes, default=parse_modes("cuda"))
    parser.add_argument("--min-free-gb", type=float, default=16.0)
    parser.add_argument("--atol", type=float)
    parser.add_argument("--rtol", type=float)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--max-nrmse", type=float, default=0.02)
    parser.add_argument("--min-top5-overlap", type=float, default=0.95)
    parser.add_argument("--skip-prompts", action="store_true")
    args = parser.parse_args()

    if args.onnx is None:
        args.onnx = Path("artifacts/onnx") / args.precision
    if args.output is None:
        args.output = Path("results/validation") / f"{args.precision}_matrix.json"
    if args.atol is None:
        args.atol = 1e-3 if args.precision == "fp32" else 5e-2
    if args.rtol is None:
        args.rtol = 1e-3 if args.precision == "fp32" else 5e-2

    if max(args.decode_batches) > 8:
        raise SystemExit("validation is capped at decode batch 8")
    if max(args.lengths + [args.batch_sequence]) > 256:
        raise SystemExit("validation is capped at sequence length 256")
    free_before = gpu_free_mib()
    required_mib = int(args.min_free_gb * 1024)
    if free_before < required_mib:
        raise SystemExit(f"GPU safety gate: {free_before} MiB free, {required_mib} MiB required")

    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model_dtype = torch.float16 if args.precision == "fp16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=model_dtype,
        attn_implementation="eager",
        local_files_only=True,
    ).eval().cuda()

    case_specs: list[tuple[str, np.ndarray]] = []
    for index, sequence in enumerate(args.lengths):
        case_specs.append(
            (f"random_b1_s{sequence}", make_random_tokens(1, sequence, int(config.vocab_size), 1000 + index))
        )
    for index, batch in enumerate(args.decode_batches):
        case_specs.append(
            (
                f"random_b{batch}_s{args.batch_sequence}",
                make_random_tokens(batch, args.batch_sequence, int(config.vocab_size), 2000 + index),
            )
        )
    if not args.skip_prompts:
        for index, prompt in enumerate(DEFAULT_PROMPTS):
            encoded = tokenizer(prompt, return_tensors="np", add_special_tokens=True)["input_ids"].astype(np.int32)
            case_specs.append((f"prompt_{index}", encoded))

    started = time.perf_counter()
    mode_results = []
    for mode in args.modes:
        mode_started = time.perf_counter()
        prefill_session = session_for(mode, args.onnx / "prefill.onnx")
        decode_session = session_for(mode, args.onnx / "decode.onnx")
        cases = []
        for name, ids_np in case_specs:
            case = run_case(
                case_name=name,
                ids_np=ids_np,
                model=model,
                config=config,
                prefill_session=prefill_session,
                decode_session=decode_session,
                atol=args.atol,
                rtol=args.rtol,
            )
            cases.append(case)
            print(
                json.dumps(
                    {
                        "mode": mode,
                        "case": name,
                        "passed": case["passed"],
                        "prefill_max_abs": case["prefill"]["logits"]["max_abs"],
                        "isolated_decode_max_abs": case["decode_isolated"]["logits"]["max_abs"],
                        "end_to_end_decode_max_abs": case["decode_end_to_end"]["logits"]["max_abs"],
                    }
                ),
                flush=True,
            )
        mode_result = {
            "mode": mode,
            "providers": prefill_session.get_providers(),
            "provider_options": prefill_session.get_provider_options(),
            "semantic_passed": all(case["semantic_passed"] for case in cases),
            "numeric_passed": all(case["numeric_passed"] for case in cases),
            "kv_finite": all(case["kv_finite"] for case in cases),
            "elapsed_seconds": time.perf_counter() - mode_started,
            "cases": cases,
        }
        logits_metrics = [
            case[stage]["logits"]
            for case in cases
            for stage in ("prefill", "decode_isolated", "decode_end_to_end")
        ]
        mode_result["distribution_passed"] = all(
            metric["cosine_similarity_min"] >= args.min_cosine
            and metric["nrmse_max"] <= args.max_nrmse
            for metric in logits_metrics
        ) and statistics.fmean(metric["top5_overlap_mean"] for metric in logits_metrics) >= args.min_top5_overlap
        mode_result["kv_allclose"] = all(
            case[stage]["kv"]["allclose"]
            for case in cases
            for stage in ("prefill", "decode_isolated", "decode_end_to_end")
        )
        strict_passed = mode_result["numeric_passed"] and mode_result["kv_allclose"]
        production_passed = mode_result["distribution_passed"]
        mode_result["passed"] = (
            mode_result["semantic_passed"]
            and mode_result["kv_finite"]
            and (strict_passed if args.precision == "fp32" else production_passed)
        )
        mode_results.append(mode_result)
        del prefill_session, decode_session
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(0.5)

    result = {
        "schema_version": 2,
        "precision": args.precision,
        "tolerances": {"atol": args.atol, "rtol": args.rtol},
        "distribution_thresholds": {
            "min_cosine": args.min_cosine,
            "max_nrmse": args.max_nrmse,
            "min_mean_top5_overlap": args.min_top5_overlap,
        },
        "gpu_free_mib_before": free_before,
        "gpu_free_mib_after": gpu_free_mib(),
        "semantic_passed": all(mode["semantic_passed"] for mode in mode_results),
        "numeric_passed": all(mode["numeric_passed"] for mode in mode_results),
        "distribution_passed": all(mode["distribution_passed"] for mode in mode_results),
        "kv_finite": all(mode["kv_finite"] for mode in mode_results),
        "elapsed_seconds": time.perf_counter() - started,
        "modes": mode_results,
    }
    result["passed"] = all(mode["passed"] for mode in mode_results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "modes"}, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
