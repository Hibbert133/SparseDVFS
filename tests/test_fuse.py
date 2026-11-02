"""Target-side C++ integration tests: temporary fake sysfs only, no real clocks."""
import sys
import tempfile
import time
import unittest
import ctypes
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))
from test_core import fake_hardware
from sparsedvfs.fuse import FUSE
from sparsedvfs.hardware import read, snapshot

LIBRARY = ROOT / 'build/libfuse_dvfs.so'


@unittest.skipUnless(LIBRARY.exists(), 'Build the Linux C++ library on Thor first')
class CppTests(unittest.TestCase):
    def test_submission_ownership_and_stale_gpu_lookahead(self):
        with tempfile.TemporaryDirectory() as d:
            hw = fake_hardware(d); hw['mock_root'] = d
            with FUSE(hw, Path(d) / 'logs', mock=True) as engine:
                lib = engine.lib; ptr = ctypes.c_void_p; u = ctypes.c_uint64
                lib.fuse_submission_begin.argtypes = [ptr, u, u]; lib.fuse_submission_begin.restype = u
                lib.fuse_submission_end.argtypes = [ptr, u]; lib.fuse_submission_end.restype = ctypes.c_int
                for name in ('fuse_gpu_boundary', 'fuse_gpu_lookahead'):
                    getattr(lib, name).argtypes = [ptr, u, u, u]; getattr(lib, name).restype = ctypes.c_int
                first = lib.fuse_submission_begin(engine.handle, 800000, 400000)
                second = lib.fuse_submission_begin(engine.handle, 1000000, 800000)
                self.assertEqual(lib.fuse_submission_end(engine.handle, first), 1)
                cpu = Path(hw['cpu'][0]['path']) / 'scaling_min_freq'
                self.assertEqual(read(cpu), '1000000')
                self.assertEqual(lib.fuse_submission_end(engine.handle, second), 0)
                self.assertEqual(lib.fuse_gpu_boundary(engine.handle, first, 500000000, 1000000000), 0)
                self.assertEqual(lib.fuse_gpu_boundary(engine.handle, second, 1000000000, 2000000000), 0)
                self.assertEqual(lib.fuse_gpu_lookahead(engine.handle, first, 200000000, 1000000000), 1)
                self.assertEqual(read(Path(hw['gpu']['path']) / 'min_freq'), '1000000000')
                self.assertEqual(lib.fuse_gpu_lookahead(engine.handle, second, 200000000, 1000000000), 0)
                self.assertEqual(read(cpu), '800000')  # GPU lookahead never changes CPU ownership.

    def test_retained_callback_handle_survives_owner_close(self):
        with tempfile.TemporaryDirectory() as d:
            hw = fake_hardware(d); hw['mock_root'] = d
            engine = FUSE(hw, Path(d) / 'logs', mock=True)
            lib = engine.lib; ptr = ctypes.c_void_p
            lib.fuse_retain.argtypes = [ptr]; lib.fuse_release.argtypes = [ptr]
            lib.fuse_retain.restype = None; lib.fuse_release.restype = None
            lib.fuse_status.argtypes = [ptr]; lib.fuse_status.restype = ctypes.c_int
            handle = engine.handle
            lib.fuse_retain(handle)
            engine.set([800000, 500000000, 2000000000])
            engine.close()
            self.assertEqual(lib.fuse_status(handle), -1)
            lib.fuse_release(handle)

    def test_set_queue_cancel_and_restore(self):
        with tempfile.TemporaryDirectory() as d:
            hw = fake_hardware(d)
            hw['mock_root'] = d
            original = snapshot(hw)
            with FUSE(hw, Path(d) / 'logs', mock=True) as engine:
                engine.set([800000, 500000000, 2000000000])
                self.assertEqual(read(Path(hw['gpu']['path']) / 'min_freq'), '500000000')
                self.assertEqual(read(Path(hw['emc']['path']) / 'rate'), '2000000000')
                engine.enqueue([400000, 200000000, 1000000000], 1000000)
                engine.cancel()
                engine.enqueue([1000000, 1000000000, 1000000000], 0)
                limit = time.monotonic() + 1
                while read(Path(hw['gpu']['path']) / 'min_freq') != '1000000000' and time.monotonic() < limit:
                    time.sleep(0.001)
                self.assertEqual(read(Path(hw['gpu']['path']) / 'min_freq'), '1000000000')
                engine.enqueue([400000, 200000000, 2000000000], 1000000)
            after = snapshot(hw)
            self.assertEqual(original['domains'], after['domains'])
            self.assertEqual(original['bpmp'], after['bpmp'])

    def test_unsupported_rate_and_python_exception_restore(self):
        with tempfile.TemporaryDirectory() as d:
            hw = fake_hardware(d); hw['mock_root'] = d
            original = snapshot(hw)
            with self.assertRaisesRegex(RuntimeError, 'outside'):
                with FUSE(hw, Path(d) / 'logs', mock=True) as engine:
                    engine.set([800000, 500000000, 2000000000])
                    engine.set([123, 500000000, 2000000000])
            after = snapshot(hw)
            self.assertEqual(original['domains'], after['domains'])
            self.assertEqual(original['bpmp'], after['bpmp'])


if __name__ == '__main__':
    unittest.main()
