#!/usr/bin/env bash
set -euo pipefail

build_dir="${1:-build-sanitizer}"
binaries=(
  "${build_dir}/test_kv_cache_cuda"
  "${build_dir}/test_greedy_sampler_cuda"
)
sanitizer="${COMPUTE_SANITIZER:-}"

if [[ -z "${sanitizer}" && -n "${CONDA_PREFIX:-}" &&
      -x "${CONDA_PREFIX}/bin/compute-sanitizer" ]]; then
  sanitizer="${CONDA_PREFIX}/bin/compute-sanitizer"
fi
if [[ -z "${sanitizer}" ]]; then
  sanitizer="$(command -v compute-sanitizer || true)"
fi

for binary in "${binaries[@]}"; do
  if [[ ! -x "${binary}" ]]; then
    echo "Missing ${binary}; configure with RelWithDebInfo and build first" >&2
    exit 2
  fi
done
if [[ -z "${sanitizer}" || ! -x "${sanitizer}" ]]; then
  echo "Compute Sanitizer was not found; activate the minillm-rt Conda environment" >&2
  exit 2
fi

echo "Using ${sanitizer}"
"${sanitizer}" --version

for tool in memcheck initcheck racecheck synccheck; do
  for binary in "${binaries[@]}"; do
    echo "== compute-sanitizer ${tool}: ${binary} =="
    "${sanitizer}" --tool "${tool}" --error-exitcode=99 "${binary}"
  done
done
