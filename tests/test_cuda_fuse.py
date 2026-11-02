"""Thor-side CUDA integration with temporary mock clocks, never real DVFS."""
import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tests'))
from test_core import fake_hardware
from sparsedvfs.fuse import FUSE
from sparsedvfs.hardware import read, snapshot


@unittest.skipUnless((ROOT / 'build/libfuse_cuda.so').exists() and importlib.util.find_spec('torch'),
                     'Build --cuda on Thor with its existing PyTorch first')
class CudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        if not torch.cuda.is_available():
            raise unittest.SkipTest('No CUDA device')
        cls.torch = torch

    def test_two_asynchronous_scopes_complete_and_restore(self):
        torch = self.torch
        stream = torch.cuda.Stream()
        with tempfile.TemporaryDirectory() as d:
            hw = fake_hardware(d); hw['mock_root'] = d
            original = snapshot(hw)
            a = torch.ones((256, 256), device='cuda', dtype=torch.bfloat16)
            stream.wait_stream(torch.cuda.current_stream())
            block0 = {'block_id': 0, 'triplet': [800000, 500000000, 1000000000], 'estimated_ms': 0.5}
            block1 = {'block_id': 1, 'triplet': [400000, 200000000, 2000000000], 'estimated_ms': 0.5}
            with FUSE(hw, Path(d) / 'logs', mock=True) as engine:
                engine.enable_cuda(torch.cuda.current_device())
                with torch.cuda.stream(stream):
                    t0 = engine.cuda_begin(stream.cuda_stream, block0, block1, 50, True)
                    for _ in range(8): _ = a @ a
                    engine.cuda_end(t0)
                    # Intentionally submit the next scope before synchronizing GPU.
                    t1 = engine.cuda_begin(stream.cuda_stream, block1, None, 50, False)
                    for _ in range(8): _ = a @ a
                    engine.cuda_end(t1)
                    engine.cuda_drain()
                self.assertEqual(read(Path(hw['gpu']['path']) / 'min_freq'), '200000000')
            after = snapshot(hw)
            self.assertEqual(original['domains'], after['domains'])
            self.assertEqual(original['bpmp'], after['bpmp'])
            with open(Path(d) / 'logs/cuda_timing.csv', newline='') as f:
                rows = list(csv.DictReader(f))
            completed = [r for r in rows if r['kind'] == 'gpu_end_observed']
            self.assertEqual(len(completed), 2)
            self.assertTrue(all(float(r['gpu_elapsed_ms']) >= 0 for r in completed))
            # Do not assert timing/savings: scheduler/hardware determine overlap.

    def test_python_failure_with_unclosed_scope_still_restores(self):
        torch = self.torch
        stream = torch.cuda.Stream()
        with tempfile.TemporaryDirectory() as d:
            hw = fake_hardware(d); hw['mock_root'] = d
            original = snapshot(hw)
            with self.assertRaisesRegex(RuntimeError, 'forced failure'):
                with FUSE(hw, Path(d) / 'logs', mock=True) as engine:
                    engine.enable_cuda(torch.cuda.current_device())
                    block = {'block_id': 0, 'triplet': [800000, 500000000, 2000000000], 'estimated_ms': 1}
                    engine.cuda_begin(stream.cuda_stream, block, None, 100, False)
                    raise RuntimeError('forced failure')
            after = snapshot(hw)
            self.assertEqual(original['domains'], after['domains'])
            self.assertEqual(original['bpmp'], after['bpmp'])


if __name__ == '__main__':
    unittest.main()
