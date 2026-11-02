#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
bash scripts/build_thor.sh --cuda
python3 -B -m unittest discover -s tests -v
python3 -B paper_demo.py --scenario dense --output /results/synthetic_dense.json > /results/synthetic_dense.stdout.json
python3 -B tools/verify_package.py
# Native clock tests use temporary mock nodes only. No real frequency mounts.
