"""Reuse the native FUSE scheduler and CUDA ABI with GPU-only backend callbacks."""
import ctypes
import os
from pathlib import Path
from sparsedvfs.fuse import FUSE
from .backends import backend

class Controller(FUSE):
    def __init__(self, config, out_dir, library=None):
        self.directory = Path(out_dir); self.directory.mkdir(parents=True, exist_ok=True)
        self.handle = self.lib = self.cuda_handle = self.cuda_lib = None
        self.events = []; self.boost_khz = 0; self.backend_error = None
        self.lock_file = None
        self.backend = backend(config)
        self.apply = not self.backend.synthetic
        self.closed = False
        try:
            if config['backend']=='nvml':
                import fcntl
                import hashlib
                self.lock_file = open('/tmp/sparsedvfs-bridge-'+hashlib.sha256(config['uuid'].encode()).hexdigest()+'.lock', 'a')
                fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.capabilities = self.backend.probe()
            self.apply = not self.backend.synthetic
            self.rates = config['rates_mhz']
            if any(type(f) is not int or f <= 0 for f in self.rates):
                raise ValueError('GPU grid must contain positive integer MHz')
            if not self.rates or not set(self.rates)<=set(self.capabilities['rates_mhz']):
                raise ValueError('Requested rates outside probed GPU grid')
            self._open(library)
        except BaseException:
            self.backend.close()
            if self.lock_file: self.lock_file.close()
            raise

    def _open(self, library):
        self.trace_path = self.directory/'fuse_trace.csv'
        config_path = self.directory/'fuse.conf'
        config_path.write_text('gpu_rates='+' '.join(str(f*1000000) for f in self.rates)+'\nlog_path='+str(self.trace_path.resolve())+'\n')
        library = library or Path(os.environ.get('SPARSEDVFS_LIBRARY_DIR', str(Path(__file__).resolve().parent.parent/'build')))/'libfuse_dvfs.so'
        self.library_path = Path(library).resolve()
        self.lib = ctypes.CDLL(str(self.library_path))
        try:
            self.lib.fuse_reference_abi_version.argtypes = []
            self.lib.fuse_reference_abi_version.restype = ctypes.c_int
            abi = self.lib.fuse_reference_abi_version()
        except AttributeError as exc:
            raise RuntimeError('Old FUSE library; rebuild v5 with scripts/build_thor.sh --cuda') from exc
        if abi != 5:
            raise RuntimeError(f'FUSE reference ABI {abi} differs from v5; rebuild the matching sources')
        ptr, u64, integer = ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int
        self.lib.fuse_open.argtypes = [ctypes.c_char_p, integer]
        self.lib.fuse_open.restype = ptr
        self.lib.fuse_error.argtypes = []
        self.lib.fuse_error.restype = ctypes.c_char_p
        for name, types in [('fuse_set', [ptr, u64, u64, u64]), ('fuse_cpu', [ptr, u64]),
                            ('fuse_enqueue', [ptr, u64, u64, u64, u64]), ('fuse_cancel', [ptr]),
                            ('fuse_close', [ptr])]:
            getattr(self.lib, name).argtypes = types
            getattr(self.lib, name).restype = integer
        self.lib.fuse_last_write_us.argtypes = [ptr]
        self.lib.fuse_last_write_us.restype = ctypes.c_double
        if self.lib.fuse_gpu_backend_abi_version() != 1: raise RuntimeError('Rebuild GPU backend extension')
        callback_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint64)
        def dispatch(context, op, mhz):
            try:
                if op == 0: self.backend.set(int(mhz))
                elif op == 1: self.backend.reset()
                else: raise ValueError('Unknown GPU callback')
                return 0
            except BaseException as exc:
                self.backend_error = repr(exc)
                return -1
        self.callback = callback_type(dispatch)
        self.lib.fuse_open_gpu.argtypes = [ctypes.c_char_p, integer, callback_type, ptr]
        self.lib.fuse_open_gpu.restype = ptr
        self.handle = self.lib.fuse_open_gpu(str(config_path.resolve()).encode(), 2 if self.backend.synthetic else 1, self.callback, None)
        if not self.handle:
            raise RuntimeError(self.lib.fuse_error().decode())

    def close(self):
        if self.closed: return
        self.closed = True
        try:
            super().close()
        finally:
            self.backend.close()
            if self.lock_file: self.lock_file.close(); self.lock_file = None
