#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
mkdir -p build
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -fPIC -shared -pthread cpp/FUSE.cpp -o build/libfuse_dvfs.so
python3 -B run_virtual_validation.py --output validation
python3 -B -m rtx5090.simulate > validation/effect_simulation.json
python3 -B - <<'PY'
import json
from pathlib import Path
s=json.loads(Path('validation/summary.json').read_text())
assert s['successful']
assert s['native_core_library_present']
assert all('test_cuda_fuse.' in x['test'] or 'test_v4_native.NativeCudaABI.' in x['test'] for x in s['skipped']), s['skipped']
print('All CPU/native mock requirements passed; CUDA tests explicitly remain unverified.')
PY
