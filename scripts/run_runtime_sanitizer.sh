#!/usr/bin/env bash
set -euo pipefail

build_dir="${1:-build-gpu-debug2}"
tools_csv="${2:-memcheck,initcheck,racecheck,synccheck}"
samples_csv="${3:-short_b1,short_b8,short_b32,long_b1}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root}"

sanitizer="${COMPUTE_SANITIZER:-${CONDA_PREFIX:-}/bin/compute-sanitizer}"
python_bin="${PYTHON:-${CONDA_PREFIX:-}/bin/python}"
binary="${build_dir}/swiftinfer_cli"
if [[ ! -x "${sanitizer}" || ! -x "${binary}" || ! -x "${python_bin}" ]]; then
  echo "activate minillm-rt and provide a RelWithDebInfo TensorRT build" >&2
  exit 2
fi

PYTHONPATH=python "${python_bin}" -c 'from validate_engine import gpu_gate; gpu_gate(28.0)'
mkdir -p results/sanitizer/runtime_inputs results/sanitizer/logs

jq -s -c 'map(select(.task_id==12) | .max_new_tokens=2)[:1][]' artifacts/validation/requests.jsonl \
  > results/sanitizer/runtime_inputs/short_b1.jsonl
jq -s -c 'map(select((.input_ids|length)==16 and .max_new_tokens==8) | .max_new_tokens=2)[:8][]' \
  artifacts/validation/requests.jsonl \
  > results/sanitizer/runtime_inputs/short_b8.jsonl
jq -s -c 'map(select((.input_ids|length)==16 and .max_new_tokens==8) | .max_new_tokens=2)[:32][]' \
  artifacts/validation/requests.jsonl \
  > results/sanitizer/runtime_inputs/short_b32.jsonl
jq -s -c 'map(select(.task_id==23) | .max_new_tokens=2)[:1][]' artifacts/validation/requests.jsonl \
  > results/sanitizer/runtime_inputs/long_b1.jsonl

IFS=',' read -r -a tools <<< "${tools_csv}"
IFS=',' read -r -a samples <<< "${samples_csv}"
for tool_name in "${tools[@]}"; do
  for sample_name in "${samples[@]}"; do
    input="results/sanitizer/runtime_inputs/${sample_name}.jsonl"
    log="results/sanitizer/logs/runtime_${tool_name}_${sample_name}.log"
    extra=()
    if [[ "${tool_name}" == "memcheck" ]]; then extra+=(--leak-check full); fi
    echo "== ${tool_name}: ${sample_name} =="
    LD_LIBRARY_PATH="artifacts/tensorrt-sdk/lib:${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}" \
      "${sanitizer}" --tool "${tool_name}" --error-exitcode=99 \
      "${extra[@]}" --log-file "${log}.tmp" "${binary}" \
      --prefill-engine artifacts/engines/rtx5090/fp16/prefill.plan \
      --decode-engine artifacts/engines/rtx5090/fp16/decode.plan \
      --model-spec artifacts/onnx/fp16_rope_lookup/model_spec.json \
      --generate-jsonl < "${input}" > /dev/null
    mv "${log}.tmp" "${log}"
    grep -E 'LEAK SUMMARY|ERROR SUMMARY|RACECHECK SUMMARY' "${log}"
  done
done
