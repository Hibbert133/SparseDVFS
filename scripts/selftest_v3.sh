#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
bash scripts/build_thor.sh --cuda
python3 -B -m unittest discover -s tests -v
python3 -B tools/verify_package.py
# Tests use TemporaryDirectory and mock_root, never /host-dvfs or real sysfs.
