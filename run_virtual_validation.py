"""Run from any directory; standard library only. Never modifies real clocks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import unittest

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'virtual_results')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.environ['VIRTUAL_TRACE_DIR'] = str(out)
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT/'tests'))
    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    with (out/'tests.txt').open('w', encoding='utf-8') as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    summary = dict(date='2026-11-02', platform=platform.platform(), python=sys.version,
                   total=result.testsRun, passed=result.testsRun-len(result.skipped)-len(result.failures)-len(result.errors),
                   skipped=[dict(test=str(t), reason=r) for t,r in result.skipped],
                   failures=[dict(test=str(t), trace=r) for t,r in result.failures],
                   errors=[dict(test=str(t), trace=r) for t,r in result.errors],
                   successful=result.wasSuccessful(), synthetic=True, deployable=False,
                   real_vllm_executed=False, real_cuda_executed=False,
                   native_core_library_present=(ROOT/'build/libfuse_dvfs.so').exists(),
                   sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                            for folder in ('cpp','sparsedvfs','sparsedvfs_vllm','rtx5090','tests')
                            for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.cpp','.h')})
    (out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ('total','passed','successful')}) )
    print('Skipped:', len(result.skipped), 'Output:', out)
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    raise SystemExit(main())
