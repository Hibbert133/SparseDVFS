"""Run on Windows with existing Python and NVIDIA driver. No service installation."""
import argparse
import hmac
import json
import os
import socket
import socketserver
import threading
import time
from pathlib import Path
from .backends import NVML, Mock


class Session:
    def __init__(self, device, token):
        self.device, self.token, self.seq = device, token, 0
        self.rates = device.probe()['rates_mhz']
        self.dirty = False
    def handle(self, req):
        if not hmac.compare_digest(str(req.get('token', '')), self.token): raise ValueError('Authentication failed')
        if req.get('uuid') != self.device.uuid: raise ValueError('GPU UUID mismatch')
        if type(req.get('seq')) is not int or req['seq'] != self.seq+1: raise ValueError('Out-of-order or duplicate request')
        self.seq = req['seq']; op = req.get('op')
        if op == 'probe': return self.device.probe()
        if op == 'read': return self.device.read()
        if op == 'set':
            mhz = req.get('mhz')
            if type(mhz) is not int or mhz not in self.rates: raise ValueError('Unapproved GPU MHz')
            self.dirty = True; self.device.set(mhz)
            return dict(requested_mhz=mhz, observed_mhz=self.device.read(), settled=False)
        if op == 'reset':
            self.device.reset(); self.dirty = False; return {'reset': True}
        raise ValueError('Unknown operation')
    def close(self):
        if self.dirty: self.device.reset(); self.dirty = False


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = False
    def __init__(self, address, device, token, log_path):
        self.device, self.token, self.log_path = device, token, Path(log_path)
        self.owner = threading.Lock(); self.failed = False
        super().__init__(address, Handler)
    def log(self, row):
        with self.log_path.open('a', encoding='utf-8') as f: f.write(json.dumps(row)+'\n')
    def server_bind(self):
        if os.name == 'nt':
            self.allow_reuse_address = False
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


class DeviceLock:
    """Cross-process ownership per GPU, including bridges on different ports."""
    def __init__(self, uuid):
        import hashlib
        key=hashlib.sha256(uuid.encode()).hexdigest()
        self.handle=None; self.file=None
        if os.name=='nt':
            import ctypes
            self.kernel=ctypes.WinDLL('kernel32',use_last_error=True)
            self.kernel.CreateMutexW.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_wchar_p]
            self.kernel.CreateMutexW.restype=ctypes.c_void_p
            self.kernel.CloseHandle.argtypes=[ctypes.c_void_p]
            self.handle=self.kernel.CreateMutexW(None,0,'Global\\SparseDVFS-'+key)
            if not self.handle: raise OSError(ctypes.get_last_error(),'GPU mutex failed')
            if ctypes.get_last_error()==183:
                self.close(); raise RuntimeError('Another bridge owns this GPU')
        else:
            import fcntl
            self.file=open('/tmp/sparsedvfs-bridge-'+key+'.lock','a')
            try: fcntl.flock(self.file,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BaseException: self.file.close(); raise
    def close(self):
        if self.handle: self.kernel.CloseHandle(self.handle); self.handle=None
        if self.file: self.file.close(); self.file=None


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        if self.server.failed or not self.server.owner.acquire(blocking=False): return
        session = None
        try:
            self.connection.settimeout(30)
            session = Session(self.server.device, self.server.token)
            while True:
                line = self.rfile.readline(65537)
                if not line: break
                if len(line) > 65536: raise ValueError('Oversized request')
                req = json.loads(line); seq = req.get('seq'); start = time.monotonic_ns()
                try:
                    result = session.handle(req)
                    reply = dict(seq=seq, ok=True, result=result)
                except Exception as exc:
                    reply = dict(seq=seq, ok=False, error=str(exc))
                reply['server_elapsed_us'] = (time.monotonic_ns()-start)/1000
                self.server.log({k:v for k,v in reply.items()})
                self.wfile.write(json.dumps(reply).encode()+b'\n'); self.wfile.flush()
                if not reply['ok']: break
        except Exception as exc: self.server.log(dict(connection_error=str(exc)))
        finally:
            try:
                if session: session.close()
            except Exception as exc:
                self.server.failed = True; self.server.log(dict(restore_failed=str(exc)))
            self.server.owner.release()


def main():
    p=argparse.ArgumentParser(); p.add_argument('--uuid', required=True); p.add_argument('--port', type=int, default=50900)
    p.add_argument('--log', default='bridge.jsonl'); p.add_argument('--mock', action='store_true')
    p.add_argument('--ack-default-auto', action='store_true'); a=p.parse_args()
    token=os.environ.get('SPARSEDVFS_BRIDGE_TOKEN', '')
    if len(token)<32: p.error('Set a random SPARSEDVFS_BRIDGE_TOKEN of at least 32 characters')
    if not a.mock and not a.ack_default_auto: p.error('Start with automatic GPU clocks and pass --ack-default-auto')
    lock=DeviceLock(a.uuid)
    device=None
    # Windows bind owns the port exclusively across bridge processes. WSL must use mirrored networking.
    try:
        device=Mock(uuid=a.uuid) if a.mock else NVML(a.uuid)
        with Server(('127.0.0.1',a.port),device,token,a.log) as server: server.serve_forever()
    finally:
        try:
            if device: device.close()
        finally: lock.close()


if __name__=='__main__': main()
