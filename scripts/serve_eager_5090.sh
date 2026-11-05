#!/usr/bin/env bash
# A separate user-operated configuration; never stops another process.
set -euo pipefail
cd "$(dirname "$0")/.."
CHECK=0
if [[ "${1:-}" == --check-only ]]; then CHECK=1; shift; fi
if [[ $# -ne 1 || "$1" != /* ]]; then
  echo 'Usage: bash scripts/serve_eager_5090.sh [--check-only] /absolute/new/run-directory' >&2
  exit 2
fi
RUN="$1"
export PYTHONPATH="/mnt/d/s00958243/triton-360:$PWD:${PYTHONPATH:-}"
export SPARSEDVFS_TRITON_PATH=/mnt/d/s00958243/triton-360
export TRITON_CACHE_DIR=/tmp/triton360-vllm-cache
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
unset NCCL_P2P_DISABLE NCCL_SHM_DISABLE NCCL_IB_DISABLE NCCL_DEBUG NCCL_DEBUG_SUBSYS
export NCCL_CUMEM_HOST_ENABLE=0
if [[ "$CHECK" == 1 ]]; then
  exec python -B -m rtx5090.eager preflight --run "$RUN"
fi
if [[ ! -f "$RUN/preflight.json" || -e "$RUN/startup.log" || -e "$RUN/launch.json" ]]; then
  echo 'Run --check-only before stopping your service. Never reuse a launch directory.' >&2
  exit 2
fi
exec > >(tee "$RUN/startup.log") 2>&1
exec python -B -m rtx5090.eager serve --run "$RUN"
