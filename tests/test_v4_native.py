"""Thor-side native ABI/no-op tests on temporary mock files, not real clocks."""
import csv
import ctypes
import tempfile
import unittest
from pathlib import Path
from test_core import fake_hardware
from sparsedvfs.fuse import FUSE

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless((ROOT / 'build/libfuse_dvfs.so').exists(), 'v5 native core not built in this environment')
class NativeCore(unittest.TestCase):
    def test_native_reference_abi_is_five(self):
        with tempfile.TemporaryDirectory() as directory:
            hw = fake_hardware(directory)
            hw['mock_root'] = directory
            with FUSE(hw, Path(directory) / 'out', mock=True) as engine:
                self.assertEqual(engine.lib.fuse_reference_abi_version(), 5)

    def test_same_gpu_emc_target_returns_noop_and_is_not_logged_as_a_write(self):
        with tempfile.TemporaryDirectory() as directory:
            hw = fake_hardware(directory)
            hw['mock_root'] = directory
            out = Path(directory) / 'out'
            with FUSE(hw, out, mock=True) as engine:
                ptr, u = ctypes.c_void_p, ctypes.c_uint64
                lib = engine.lib
                lib.fuse_submission_begin.argtypes = [ptr, u, u]
                lib.fuse_submission_begin.restype = u
                for name in ('fuse_gpu_boundary', 'fuse_gpu_lookahead'):
                    getattr(lib, name).argtypes = [ptr, u, u, u]
                    getattr(lib, name).restype = ctypes.c_int
                ticket = lib.fuse_submission_begin(engine.handle, 1000000, 400000)
                self.assertEqual(lib.fuse_gpu_boundary(engine.handle, ticket, 500000000, 1000000000), 0)
                nodes = [Path(hw['gpu']['path']) / n for n in ('min_freq', 'max_freq')]
                nodes += [Path(hw['emc']['path']) / n for n in ('rate', 'mrq_rate_locked')]
                before = [(p.read_text(), p.stat().st_mtime_ns) for p in nodes]
                self.assertEqual(lib.fuse_gpu_lookahead(engine.handle, ticket, 500000000, 1000000000), 2)
                self.assertEqual(before, [(p.read_text(), p.stat().st_mtime_ns) for p in nodes])
            with (out / 'fuse_trace.csv').open(newline='') as source:
                kinds = [r['kind'] for r in csv.DictReader(source)]
            self.assertIn('gpu_event_lookahead_unchanged', kinds)
            self.assertNotIn('gpu_event_lookahead', kinds)


@unittest.skipUnless((ROOT / 'build/libfuse_cuda.so').exists(), 'v5 CUDA adapter not built in this environment')
class NativeCudaABI(unittest.TestCase):
    def test_cuda_reference_abi_is_five(self):
        library = ctypes.CDLL(str(ROOT / 'build/libfuse_cuda.so'))
        library.fuse_cuda_reference_abi_version.argtypes = []
        library.fuse_cuda_reference_abi_version.restype = ctypes.c_int
        self.assertEqual(library.fuse_cuda_reference_abi_version(), 5)


if __name__ == '__main__':
    unittest.main()
