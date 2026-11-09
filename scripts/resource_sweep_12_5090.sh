#!/usr/bin/env bash
# WSL; use only after the earlier collector has exited. No restart or refit.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# != 3 || "$3" != --apply ]]; then
  echo 'Usage: bash scripts/resource_sweep_12_5090.sh /service/run /new/output --apply' >&2
  exit 2
fi
RUN="$1"
OUT="$2"
PY=/home/s00958243/miniforge3/envs/vllm-qwen/bin/python
[[ "$RUN" == /* && "$OUT" == /* && ! -e "$OUT" ]]
[[ -n "${SPARSEDVFS_BRIDGE_TOKEN:-}" ]] || { echo 'Missing token; no writes'; exit 2; }
export PYTHONPATH="/mnt/d/s00958243/triton-360:$PWD:${PYTHONPATH:-}"
export SPARSEDVFS_TRITON_PATH=/mnt/d/s00958243/triton-360
export TRITON_CACHE_DIR=/tmp/triton360-vllm-cache
mkdir "$OUT"
"$PY" -B -m rtx5090.resource_calibration check --run "$RUN" --telemetry configs/service_telemetry.template.json --out "$OUT/check" --rates 300 600 900 1200 1500 1800 2025 2250 2527 2700 2850 3000
set +e
"$PY" -u -B -m rtx5090.resource_calibration sweep --run "$RUN" --telemetry configs/service_telemetry.template.json --out "$OUT/collection" --window-seconds 60 --sweep-repeats 2 --apply --ack-default-auto 2>&1 | tee "$OUT/progress.log"
code=${PIPESTATUS[0]}
set -e
if [[ -f "$OUT/collection/collection.json" ]]; then
  "$PY" -B -m rtx5090.resource_sweep_report --input "$OUT/collection/collection.json" --out "$OUT/report"
fi
echo "Finished with exit=$code; evidence: $OUT"
exit "$code"
