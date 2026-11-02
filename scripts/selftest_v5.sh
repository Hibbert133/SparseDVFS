#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
bash scripts/build_thor.sh --cuda
python3 -B -m unittest discover -s tests -v
python3 -B paper_demo.py --scenario dense --output /results/synthetic_dense.json > /results/synthetic_dense.stdout.json
python3 -B v5_demo.py --output /results/synthetic_v5.json > /results/synthetic_v5.stdout.json
python3 -B tools/verify_package.py
# No real control mounts. Native tests use only temporary mock sysfs nodes.
