#!/usr/bin/env bash
# WSL only. No automatic service restart, no retry, never overwrites evidence.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# != 3 || "$3" != --apply ]]; then
  echo 'Usage: bash scripts/resource_model_5090.sh /absolute/service/run /absolute/new/output --apply' >&2
  exit 2
fi
RUN="$1"
OUT="$2"
PY=/home/s00958243/miniforge3/envs/vllm-qwen/bin/python
[[ "$RUN" == /* && "$OUT" == /* && ! -e "$OUT" ]]
[[ -n "${SPARSEDVFS_BRIDGE_TOKEN:-}" ]] || { echo 'Missing SPARSEDVFS_BRIDGE_TOKEN; no writes' >&2; exit 2; }
export PYTHONPATH="/mnt/d/s00958243/triton-360:$PWD:${PYTHONPATH:-}"
export SPARSEDVFS_TRITON_PATH=/mnt/d/s00958243/triton-360
export TRITON_CACHE_DIR=/tmp/triton360-vllm-cache
mkdir "$OUT"
"$PY" -B -m rtx5090.resource_calibration check --run "$RUN" --telemetry configs/service_telemetry.template.json --out "$OUT/check"
"$PY" -B -m rtx5090.resource_calibration train --run "$RUN" --telemetry configs/service_telemetry.template.json --out "$OUT/train" --apply --ack-default-auto
"$PY" -B -m rtx5090.resource_model fit --input "$OUT/train/collection.json" --out "$OUT/frozen-model.json"
"$PY" -B -m rtx5090.resource_calibration holdout --run "$RUN" --telemetry configs/service_telemetry.template.json --out "$OUT/holdout" --model "$OUT/frozen-model.json" --apply --ack-default-auto
"$PY" -B -m rtx5090.resource_model validate --input "$OUT/holdout/collection.json" --model "$OUT/frozen-model.json" --out "$OUT/validation.json"
echo "Saved: $OUT/validation.json (conditional primitive model, not Qwen deployment certificate)"
