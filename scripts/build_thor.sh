#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
command -v g++ >/dev/null || { echo 'g++ is required on the target Thor.' >&2; exit 1; }
mkdir -p build
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -shared -fPIC -pthread cpp/FUSE.cpp -o build/libfuse_dvfs.so
echo "Built $(pwd)/build/libfuse_dvfs.so"
if [[ ${1:-} == --cuda ]]; then
  cuda_root=${CUDA_HOME:-/usr/local/cuda}
  [[ -f "$cuda_root/include/cuda_runtime_api.h" ]] || { echo 'CUDA headers unavailable; set CUDA_HOME to the existing matching toolkit.' >&2; exit 1; }
  g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -shared -fPIC -pthread \
    -I"$cuda_root/include" cpp/CUDAFUSE.cpp -Lbuild -lfuse_dvfs \
    -L"$cuda_root/lib64" -lcudart -Wl,-rpath,'$ORIGIN' -o build/libfuse_cuda.so
  echo "Built $(pwd)/build/libfuse_cuda.so"
elif [[ $# -gt 0 ]]; then
  echo 'Usage: bash scripts/build_thor.sh [--cuda]' >&2
  exit 2
fi
