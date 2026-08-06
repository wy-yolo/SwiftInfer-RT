#!/usr/bin/env bash
set -euo pipefail

build_dir="${1:-build-sanitizer}"
binary="${build_dir}/test_kv_cache_cuda"

if [[ ! -x "${binary}" ]]; then
  echo "Missing ${binary}; configure with RelWithDebInfo and build first" >&2
  exit 2
fi

for tool in memcheck initcheck racecheck synccheck; do
  echo "== compute-sanitizer ${tool} =="
  compute-sanitizer --tool "${tool}" --error-exitcode=99 "${binary}"
done

