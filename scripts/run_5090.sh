#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
[[ $(uname -m) == x86_64 ]] || { echo 'Requires Linux/WSL2 x86_64'; exit 1; }
mkdir -p models runs
if [[ ${1:-} == build ]]; then
  docker build --platform linux/amd64 -f container/Dockerfile.5090 -t sparsedvfs:5090 .
  exit
fi
docker run --rm --gpus all --network host --ipc host \
  -e SPARSEDVFS_BRIDGE_TOKEN -e SPARSEDVFS_APPLY=0 \
  -v "$PWD/models:/model" -v "$PWD/runs:/results" \
  -v "$PWD/configs:/configs:ro" sparsedvfs:5090 "$@"
