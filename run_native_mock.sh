#!/usr/bin/env bash
# Requires an existing Linux C++17 compiler, CMake and Python. No installation.
set -euo pipefail
cd -- "$(dirname -- "$0")"
cmake -S . -B build -DFUSE_WITH_CUDA=OFF
cmake --build build --parallel 2
python3 -B run_virtual_validation.py --output virtual_results_linux
# Require the actual native mock suites to pass without skips.
python3 -B - <<'PY'
import sys
import unittest
sys.path.insert(0, 'tests')
import test_fuse
import test_v4_native
import test_native_schedule_mock
suite = unittest.TestSuite([
    unittest.defaultTestLoader.loadTestsFromTestCase(test_fuse.CppTests),
    unittest.defaultTestLoader.loadTestsFromTestCase(test_v4_native.NativeCore),
    unittest.defaultTestLoader.loadTestsFromTestCase(test_native_schedule_mock.NativeScheduleTests),
])
result = unittest.TextTestRunner(verbosity=2).run(suite)
if not result.wasSuccessful() or result.skipped:
    raise SystemExit('Native mock acceptance failed or skipped')
PY
