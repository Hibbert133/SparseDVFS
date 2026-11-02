"""GPU-only backends. NVML is loaded from the installed NVIDIA driver; no pip dependency."""
import ctypes as C
import json
import os
import socket
import time


class Mock:
    synthetic = True
    def __init__(self, rates=(900, 1500, 2100), uuid='GPU-MOCK'):
        self.rates, self.uuid, self.current, self.calls = list(rates), uuid, 0, []
    def probe(self):
        return dict(uuid=self.uuid, rates_mhz=self.rates, synthetic=True, clock_mhz=self.current)
    def set(self, mhz):
        if type(mhz) is not int or mhz not in self.rates:
            raise ValueError('Unsupported GPU MHz')
        self.calls.append(mhz); self.current = mhz
    def read(self): return self.current
    def reset(self): self.calls.append('reset'); self.current = 0
    def close(self): pass


class NVML:
    synthetic = False
    def __init__(self, uuid):
        if not uuid.startswith('GPU-'): raise ValueError('Use a GPU UUID, not a mutable device index')
        self.uuid = uuid
        if os.name == 'nt':
            self.lib = C.CDLL(os.path.join(os.environ['SystemRoot'], 'System32', 'nvml.dll'))
        else: self.lib = C.CDLL('libnvidia-ml.so.1')
        self._call('nvmlInit_v2')
        self.handle = C.c_void_p()
        try: self._call('nvmlDeviceGetHandleByUUID', uuid.encode(), C.byref(self.handle))
        except BaseException:
            self._call('nvmlShutdown'); raise
    def _call(self, name, *args):
        rc = getattr(self.lib, name)(*args)
        if rc: raise RuntimeError(f'{name}: NVML error {rc} (3=no permission, 4=not supported)')
    def read(self):
        value = C.c_uint(); self._call('nvmlDeviceGetClockInfo', self.handle, C.c_uint(0), C.byref(value))
        return value.value
    def _list(self, name, *prefix):
        count = C.c_uint(0)
        rc = getattr(self.lib, name)(self.handle, *prefix, C.byref(count), None)
        if rc not in (0, 7): raise RuntimeError(f'{name}: NVML error {rc}')
        if not 0 < count.value <= 65536: raise RuntimeError('Invalid NVML clock list')
        values = (C.c_uint * count.value)()
        self._call(name, self.handle, *prefix, C.byref(count), values)
        return list(values)[:count.value]
    def probe(self):
        memory = self._list('nvmlDeviceGetSupportedMemoryClocks')
        # Memory remains system managed. Only use core rates valid for every reported memory state.
        rates = set.intersection(*(set(self._list('nvmlDeviceGetSupportedGraphicsClocks', C.c_uint(m))) for m in memory))
        if not rates: raise RuntimeError('No common supported GPU clocks')
        driver=C.create_string_buffer(128); name=C.create_string_buffer(128)
        self._call('nvmlSystemGetDriverVersion',driver,C.c_uint(len(driver)))
        self._call('nvmlDeviceGetName',self.handle,name,C.c_uint(len(name)))
        return dict(uuid=self.uuid, driver=driver.value.decode(), name=name.value.decode(), rates_mhz=sorted(rates), clock_mhz=self.read(), synthetic=False,
                    set_reset_verified=False, memory_policy='system-managed')
    def set(self, mhz):
        self._call('nvmlDeviceSetGpuLockedClocks', self.handle, C.c_uint(mhz), C.c_uint(mhz))
    def reset(self): self._call('nvmlDeviceResetGpuLockedClocks', self.handle)
    def close(self): self._call('nvmlShutdown')


class Bridge:
    """Persistent authenticated transport; elapsed times are measured on the caller's clock."""
    def __init__(self, host, port, token, uuid):
        self.socket = socket.create_connection((host, port), timeout=5)
        self.file = self.socket.makefile('rwb'); self.token = token; self.uuid = uuid; self.seq = 0
        self.synthetic = False
    def request(self, op, **kwargs):
        self.seq += 1
        request = dict(token=self.token, uuid=self.uuid, seq=self.seq, op=op, **kwargs)
        start = time.monotonic_ns()
        self.file.write(json.dumps(request).encode() + b'\n'); self.file.flush()
        line = self.file.readline(65537)
        if not line or len(line) > 65536: raise RuntimeError('Bridge disconnected or oversized reply')
        reply = json.loads(line)
        if reply.get('seq') != self.seq: raise RuntimeError('Bridge sequence mismatch')
        if not reply.get('ok'): raise RuntimeError('Bridge: ' + reply.get('error', 'unknown error'))
        self.last_roundtrip_us = (time.monotonic_ns()-start)/1000
        return reply['result']
    def probe(self):
        value = self.request('probe'); self.synthetic = value['synthetic']; return value
    def set(self, mhz): self.request('set', mhz=mhz)
    def read(self): return self.request('read')
    def reset(self): self.request('reset')
    def close(self):
        self.file.close(); self.socket.close()


def backend(config):
    kind = config['backend']
    if kind == 'mock': return Mock(config['rates_mhz'], config.get('uuid', 'GPU-MOCK'))
    if not config.get('default_auto_ack'):
        raise ValueError('Real control requires acknowledging initial default automatic clocks')
    if kind == 'nvml': return NVML(config['uuid'])
    if kind == 'bridge':
        token = os.environ.get('SPARSEDVFS_BRIDGE_TOKEN')
        if not token: raise ValueError('Missing SPARSEDVFS_BRIDGE_TOKEN')
        return Bridge(config.get('host', '127.0.0.1'), config.get('port', 50900), token, config['uuid'])
    raise ValueError('Unknown backend')
