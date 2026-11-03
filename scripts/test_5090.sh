#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
RESULTS="${1:-validation}"
mkdir -p build "$RESULTS"
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -fPIC -shared -pthread cpp/FUSE.cpp -o build/libfuse_dvfs.so
python3 -B run_virtual_validation.py --output "$RESULTS"
python3 -B -m rtx5090.simulate > "$RESULTS/effect_simulation.json"
python3 -B - "$RESULTS" <<'PY'
import json
import sys
from pathlib import Path
s=json.loads((Path(sys.argv[1])/'summary.json').read_text())
assert s['successful']
assert s['native_core_library_present']
assert all('test_cuda_fuse.' in x['test'] or 'test_v4_native.NativeCudaABI.' in x['test'] for x in s['skipped']), s['skipped']
print('All CPU/native mock requirements passed; CUDA tests explicitly remain unverified.')
PY
