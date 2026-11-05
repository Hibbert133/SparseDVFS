#!/usr/bin/env bash
# User-operated one-restart launcher. Never stops a running service itself.
set -euo pipefail
cd "$(dirname "$0")/.."

CHECK_ONLY=0
if [[ "${1:-}" == "--check-only" ]]; then
  CHECK_ONLY=1
  shift
fi
if [[ $# -ne 1 || "$1" != /* ]]; then
  echo 'Usage: bash scripts/serve_profiled_5090.sh [--check-only] /absolute/new/profile-directory' >&2
  exit 2
fi
PROFILE_DIR="$1"
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH="/mnt/d/s00958243/triton-360:${PYTHONPATH:-}"
export TRITON_CACHE_DIR=/tmp/triton360-vllm-cache
# This script runs in its own shell; the invoking terminal is not changed.
unset NCCL_P2P_DISABLE NCCL_SHM_DISABLE NCCL_IB_DISABLE NCCL_DEBUG NCCL_DEBUG_SUBSYS
export NCCL_CUMEM_HOST_ENABLE=0

if [[ "$CHECK_ONLY" == 1 ]]; then
  python -B -m rtx5090.service_model preflight --out "$PROFILE_DIR"
  exit
fi
if [[ ! -f "$PROFILE_DIR/preflight.json" ]]; then
  echo 'Run --check-only BEFORE stopping the current service.' >&2
  exit 2
fi
if [[ -e "$PROFILE_DIR/startup.log" || -e "$PROFILE_DIR/launch.json" ]]; then
  echo 'Launch evidence already exists; preserve it. Do not restart automatically.' >&2
  exit 2
fi
python -B -m rtx5090.service_model preflight --launch --out "$PROFILE_DIR"
PROFILE_CONFIG="$(python -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["profiler_config"]))' "$PROFILE_DIR/preflight.json")"
exec > >(tee "$PROFILE_DIR/startup.log") 2>&1
exec vllm serve /mnt/d/models/Qwen3.5-27B \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.90 \
  --reasoning-parser qwen3 \
  --language-model-only \
  --trust-remote-code \
  --profiler-config "$PROFILE_CONFIG"
