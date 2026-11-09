#!/usr/bin/env bash
# WSL, after the original collection process has fully exited. No restart.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# != 4 || "$4" != --apply ]]; then
  echo 'Usage: bash scripts/resource_wide_frequency_5090.sh /service/run /old/train/collection.json /new/output --apply' >&2
  exit 2
fi
RUN="$1"
OLD="$2"
OUT="$3"
PY=/home/s00958243/miniforge3/envs/vllm-qwen/bin/python
[[ "$RUN" == /* && "$OLD" == /* && "$OUT" == /* && -f "$OLD" && ! -e "$OUT" ]]
[[ -n "${SPARSEDVFS_BRIDGE_TOKEN:-}" ]] || { echo 'Missing token; no writes'; exit 2; }
export PYTHONPATH="/mnt/d/s00958243/triton-360:$PWD:${PYTHONPATH:-}"
export SPARSEDVFS_TRITON_PATH=/mnt/d/s00958243/triton-360
export TRITON_CACHE_DIR=/tmp/triton360-vllm-cache
# Refuse to spend new GPU time if the old source is incomplete/unrestored.
"$PY" -B - "$OLD" <<'PY'
import json,sys
c=json.load(open(sys.argv[1]))
assert c['status']=='resource_collection_complete' and c['collection_role']=='train'
assert c['restoration']['verified'] and c['requested_window_s']==30
assert {r['requested_mhz'] for r in c['rows']}=={2250,2527}
print('Existing high-frequency training preserved:',len(c['rows']),'rank rows')
PY
mkdir "$OUT"
"$PY" -B -m rtx5090.resource_calibration check --run "$RUN" --telemetry configs/service_telemetry.template.json --out "$OUT/check" --rates 1800 1950 2025 2250 2385 2527
"$PY" -B - "$OLD" "$OUT/check/collection.json" <<'PY'
import json,sys
old,current=[json.load(open(p)) for p in sys.argv[1:]]
assert old['identity_fingerprint']==current['identity_fingerprint'], 'Service identity changed; no writes'
print('Same resident identity verified; starting supplemental pilot')
PY
"$PY" -B -m rtx5090.resource_calibration pilot --run "$RUN" --telemetry configs/service_telemetry.template.json --out "$OUT/low-pilot" --rates 1800 --window-seconds 30 --apply --ack-default-auto
"$PY" -B -m rtx5090.resource_calibration train --run "$RUN" --telemetry configs/service_telemetry.template.json --out "$OUT/low-train" --rates 1800 2025 --window-seconds 30 --apply --ack-default-auto
"$PY" -B -m rtx5090.resource_model merge --input "$OLD" --extra "$OUT/low-train/collection.json" --out "$OUT/merged-training.json"
"$PY" -B -m rtx5090.resource_model fit --input "$OUT/merged-training.json" --out "$OUT/frozen-model.json"
"$PY" -B -m rtx5090.resource_calibration holdout --run "$RUN" --telemetry configs/service_telemetry.template.json --out "$OUT/holdout" --rates 1950 2385 --seed 711100 --window-seconds 30 --model "$OUT/frozen-model.json" --apply --ack-default-auto
"$PY" -B -m rtx5090.resource_model validate --input "$OUT/holdout/collection.json" --model "$OUT/frozen-model.json" --out "$OUT/validation.json"
echo "Saved: $OUT/validation.json (scores_by_frequency; conditional, not deployable)"
