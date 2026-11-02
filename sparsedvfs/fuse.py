"""ctypes bridge to the demo C++ engine; defaults to logging without writes."""
import ctypes
import os
from pathlib import Path
from .hardware import snapshot, write_json


class FUSE:
    def __init__(self, hardware, out_dir, apply=False, library=None, mock=False, native_dry=False):
        self.hardware, self.apply = hardware, apply
        self.directory = Path(out_dir)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.handle, self.lib, self.events = None, None, []
        self.cuda_handle, self.cuda_lib = None, None
        self.boost_khz = max(hardware['cpu_common_khz'])
        self.trace_path = self.directory / 'fuse_trace.csv'
        if not apply and not mock and not native_dry:
            return
        if apply and not mock and (not hasattr(os, 'geteuid') or os.geteuid() != 0):
            raise RuntimeError('--apply must run as root on Thor')
        emc = hardware['emc']
        if not emc['frequencies']:
            raise RuntimeError('EMC OPP list unresolved')
        cfg = {'cpu_paths': ' '.join(d['path'] for d in hardware['cpu']),
               'gpu_path': hardware['gpu']['path'], 'emc_backend': emc['backend'],
               'emc_path': emc['path'], 'bwmgr_halt': emc.get('halt_path', ''),
               'cpu_rates': ' '.join(map(str, hardware['cpu_common_khz'])),
               'gpu_rates': ' '.join(map(str, hardware['gpu']['frequencies'])),
               'emc_rates': '' if emc['backend'] == 'off' else ' '.join(map(str, emc['frequencies'])),
               'log_path': str(self.trace_path), 'mock_root': hardware.get('mock_root', '')}
        if any('\n' in str(v) or '\r' in str(v) for v in cfg.values()):
            raise ValueError('Newline in FUSE configuration')
        config_path = self.directory / 'fuse.conf'
        config_path.write_text(''.join(f'{k}={v}\n' for k, v in cfg.items()), encoding='utf-8')
        library = library or Path(__file__).resolve().parent.parent / 'build/libfuse_dvfs.so'
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
        self.handle = self.lib.fuse_open(str(config_path.resolve()).encode(), 2 if mock else (1 if apply else 0))
        if not self.handle:
            raise RuntimeError(self.lib.fuse_error().decode())
        # C++ now owns the device lock, but has performed no frequency writes.
        # Journal the current state before the first explicit set/enqueue call.
        try:
            saved = snapshot(hardware)
            expected = [(d['original_min'], d['original_max']) for d in hardware['cpu'] + [hardware['gpu']]]
            actual = [(d['min'], d['max']) for d in saved['domains']]
            if actual != expected:
                raise RuntimeError('CPU/GPC constraints changed since probe; refresh the hardware report and reprofile')
            write_json(self.directory / 'clock_snapshot.json', saved)
        except BaseException:
            handle, self.handle = self.handle, None
            self.lib.fuse_close(handle)
            raise

    def call(self, name, *args):
        if self.handle and getattr(self.lib, name)(self.handle, *args) != 0:
            raise RuntimeError(self.lib.fuse_error().decode())

    def set(self, triplet, boost=False):
        c, g, e = map(int, triplet)
        c = self.boost_khz if boost else c
        self.events.append({'kind': 'boundary', 'triplet': [c, g, e]})
        self.call('fuse_set', c, g, e)

    def cpu(self, khz):
        self.call('fuse_cpu', int(khz))

    def enqueue(self, triplet, delay_us, boost=False):
        c, g, e = map(int, triplet)
        c = self.boost_khz if boost else c
        self.events.append({'kind': 'lookahead', 'triplet': [c, g, e], 'delay_us': delay_us})
        self.call('fuse_enqueue', c, g, e, max(0, int(delay_us)))

    def cancel(self):
        self.call('fuse_cancel')

    def enable_cuda(self, device, poll_us=100, expected_cuda=None):
        if self.cuda_handle:
            raise RuntimeError('CUDA adapter already initialized')
        if not self.handle:
            raise RuntimeError('CUDA stream integration requires the native core (native_dry also supports no-write operation)')
        ptr, u64, integer = ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int
        lib = ctypes.CDLL(str(self.library_path.with_name('libfuse_cuda.so')))
        try:
            lib.fuse_cuda_reference_abi_version.argtypes = []
            lib.fuse_cuda_reference_abi_version.restype = integer
            abi = lib.fuse_cuda_reference_abi_version()
        except AttributeError as exc:
            raise RuntimeError('Old CUDA adapter; rebuild both v5 libraries with --cuda') from exc
        if abi != 5:
            raise RuntimeError('CUDA adapter/reference ABI mismatch; rebuild both v5 libraries')
        lib.fuse_cuda_error.argtypes = []
        lib.fuse_cuda_error.restype = ctypes.c_char_p
        lib.fuse_cuda_runtime_version.argtypes = []
        lib.fuse_cuda_runtime_version.restype = integer
        runtime_version = lib.fuse_cuda_runtime_version()
        if runtime_version < 0:
            raise RuntimeError(lib.fuse_cuda_error().decode())
        if expected_cuda and int(str(expected_cuda).split('.')[0]) != runtime_version // 1000:
            raise RuntimeError('CUDA adapter/PyTorch CUDA major versions differ; build with the matching existing toolkit')
        lib.fuse_cuda_open.argtypes = [ptr, integer, ctypes.c_char_p, ctypes.c_uint]
        lib.fuse_cuda_open.restype = ptr
        lib.fuse_cuda_begin.argtypes = [ptr, ptr] + [u64] * 9 + [integer]
        lib.fuse_cuda_begin.restype = u64
        for name, types in [('fuse_cuda_end', [ptr, u64]), ('fuse_cuda_drain', [ptr]), ('fuse_cuda_close', [ptr])]:
            getattr(lib, name).argtypes = types
            getattr(lib, name).restype = integer
        handle = lib.fuse_cuda_open(self.handle, int(device), str(self.directory / 'cuda_timing.csv').encode(), int(poll_us))
        if not handle:
            raise RuntimeError(lib.fuse_cuda_error().decode())
        self.cuda_lib, self.cuda_handle = lib, handle

    def cuda_begin(self, stream, block, upcoming, switch_us, lookahead):
        if not self.cuda_handle:
            raise RuntimeError('CUDA adapter was not initialized; build scripts/build_thor.sh --cuda')
        _, gpu, emc = block['triplet']
        next_gpu, next_emc = upcoming['triplet'][1:] if upcoming else (gpu, emc)
        ticket = self.cuda_lib.fuse_cuda_begin(self.cuda_handle, int(stream), int(block['block_id']),
                                               self.boost_khz, int(block['triplet'][0]), int(gpu), int(emc),
                                               max(1, round(block['estimated_ms'] * 1000)), round(switch_us),
                                               int(next_gpu), int(next_emc), int(bool(lookahead and upcoming)))
        if not ticket:
            raise RuntimeError(self.cuda_lib.fuse_cuda_error().decode())
        self.events.append({'kind': 'cuda_block', 'ticket': ticket, 'block_id': block['block_id'],
                            'triplet': block['triplet'], 'lookahead': bool(lookahead and upcoming)})
        return ticket

    def cuda_end(self, ticket):
        if self.cuda_lib.fuse_cuda_end(self.cuda_handle, ticket) != 0:
            raise RuntimeError(self.cuda_lib.fuse_cuda_error().decode())

    def cuda_drain(self):
        if self.cuda_handle and self.cuda_lib.fuse_cuda_drain(self.cuda_handle) != 0:
            raise RuntimeError(self.cuda_lib.fuse_cuda_error().decode())

    def close(self):
        errors = []
        if self.cuda_handle:
            handle, self.cuda_handle = self.cuda_handle, None
            if self.cuda_lib.fuse_cuda_close(handle) != 0:
                errors.append(self.cuda_lib.fuse_cuda_error().decode())
        if self.handle:
            handle, self.handle = self.handle, None
            if self.lib.fuse_close(handle) != 0:
                errors.append(self.lib.fuse_error().decode())
        write_json(self.directory / 'requested_schedule.json', {'dvfs_applied': self.apply,
                                                              'reference_abi': 5 if self.lib else None,
                                                              'writes_are_constraints_not_pll_confirmation': True,
                                                              'events': self.events})
        if errors:
            raise RuntimeError('Controller cleanup: ' + '; '.join(errors))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
