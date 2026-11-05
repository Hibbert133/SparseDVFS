"""Local JSON RPC and fail-stop two-rank rendezvous; no model collectives."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import socket
import socketserver
import threading
import time

from .observe_core import GPU_UUIDS
from .performance import safe_error

LIMIT = 1024*1024


def rpc(path, message, timeout=30):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.settimeout(timeout)
        conn.connect(str(path))
        with conn.makefile('rwb') as stream:
            stream.write(json.dumps(message, allow_nan=False).encode()+b'\n'); stream.flush()
            raw = stream.readline(LIMIT+1)
            if not raw or len(raw) > LIMIT: raise RuntimeError('Local RPC disconnected/oversized')
            response = json.loads(raw)
    if response.get('ok') is not True: raise RuntimeError(response.get('error', 'Local RPC failed'))
    return response['result']


class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        self.connection.settimeout(self.server.timeout_s)
        try:
            raw = self.rfile.readline(LIMIT+1)
            if not raw or len(raw) > LIMIT: raise ValueError('Invalid local RPC frame')
            result = self.server.callback(json.loads(raw))
            response = dict(ok=True, result=result)
        except BaseException as exc:
            response = dict(ok=False, error=safe_error(exc))
        try:
            self.wfile.write(json.dumps(response, allow_nan=False).encode()+b'\n'); self.wfile.flush()
        except OSError:
            if self.server.on_disconnect: self.server.on_disconnect('Client disconnected before release')


class LocalServer(socketserver.ThreadingMixIn, getattr(socketserver, 'UnixStreamServer', socketserver.TCPServer)):
    daemon_threads = True
    block_on_close = False

    def __init__(self, path, callback, timeout=30, on_disconnect=None):
        self.path = Path(path)
        parent = self.path.parent
        # Socket directory must be local and private, never a writable shared /mnt/d directory.
        if os.name != 'posix' or parent.stat().st_uid != os.getuid() or parent.stat().st_mode & 0o077:
            raise ValueError('RPC directory must be owned by this uid with mode 0700')
        self.callback, self.timeout_s, self.on_disconnect = callback, timeout, on_disconnect
        super().__init__(str(path), _Handler)
        os.chmod(path, 0o600)
        self.thread = threading.Thread(target=self.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.shutdown(); self.server_close(); self.thread.join(timeout=2)
        self.path.unlink(missing_ok=True)


class Rendezvous:
    """Exactly one action after both ranks complete the SAME GPU boundary.

    A generation remains poisoned after timeout/mismatch/action failure. Later
    arrivals cannot clear an error and accidentally proceed to a collective.
    """
    def __init__(self, action, fingerprint, timeout=10):
        self.action, self.fingerprint, self.timeout = action, fingerprint, timeout
        self.condition = threading.Condition()
        self.arrivals, self.completed, self.error = {}, {}, None
        self.next_seq = {0: 0, 1: 0}
        self.generation = 0

    def abort(self, reason):
        with self.condition:
            self.error = self.error or str(reason)
            self.condition.notify_all()

    def exchange(self, message):
        with self.condition:
            try:
                if self.error: raise RuntimeError(self.error)
                rank = message['rank']
                if (type(rank) is not int or rank not in (0, 1)
                        or message['uuid'] != GPU_UUIDS[rank]
                        or message['fingerprint'] != self.fingerprint
                        or message.get('gpu_complete') is not True):
                    raise ValueError('Boundary identity/completion mismatch')
                seq = message['seq']
                if type(seq) is not int or seq != self.next_seq[rank] or seq != self.generation:
                    raise ValueError('Duplicate or out-of-order boundary')
                if rank in self.arrivals: raise ValueError('Duplicate rank arrival')
                if self.arrivals and next(iter(self.arrivals.values()))['key'] != message['key']:
                    raise ValueError('Ranks reached different request/context/block boundaries')
                self.arrivals[rank] = message
                self.next_seq[rank] += 1
                if len(self.arrivals) == 2:
                    result = self.action(message['key'])
                    self.completed[seq] = [result, 2]
                    self.arrivals = {}
                    self.generation += 1
                    self.condition.notify_all()
                else:
                    deadline = time.monotonic()+self.timeout
                    while seq not in self.completed and not self.error:
                        remaining = deadline-time.monotonic()
                        if remaining <= 0: raise TimeoutError('Peer rank did not reach GPU boundary')
                        self.condition.wait(remaining)
                if self.error: raise RuntimeError(self.error)
                result = self.completed[seq][0]
                self.completed[seq][1] -= 1
                if self.completed[seq][1] == 0: del self.completed[seq]
                return result
            except BaseException as exc:
                self.error = self.error or safe_error(exc)
                self.condition.notify_all()
                raise


class PairSwitch:
    """Synchronous GPU-only FUSE adapter; physical transitions are bracketed.

    CPU/EMC are absent capabilities, not fake positive frequency placeholders.
    """
    def __init__(self, controller, plan, journal, timeout=5, interval=.01):
        self.controller, self.plan, self.journal = controller, plan, journal
        self.timeout, self.interval = timeout, interval
        self.current = None

    def __call__(self, key):
        if not isinstance(key, list) or len(key) != 3: raise ValueError('Invalid boundary key')
        _, context, index = key
        blocks = self.plan['contexts'][context]
        if type(index) is not int or not 0 <= index < len(blocks): raise ValueError('Invalid block index')
        target = blocks[index]['mhz']
        row = dict(key=key, target_mhz=target, begin_ns=time.monotonic_ns(), samples=[],
                   hardware_precision='query_bracket_not_exact_transition_time')
        try:
            for rank, device in enumerate(self.controller.devices):
                if self.plan.get('temperature_ranges'):
                    measured = device.telemetry()
                    lo, hi = self.plan['temperature_ranges'][str(rank)]
                    temp = measured.get('temperature_c')
                    if temp is None or not lo <= temp <= hi:
                        raise RuntimeError('Temperature left calibrated range; next block not submitted')
            if target != self.current:
                self.controller.set(target)
            row['api_return_ns'] = time.monotonic_ns()
            consecutive = 0
            deadline = time.monotonic()+self.timeout
            with ThreadPoolExecutor(max_workers=2) as pool:
                while time.monotonic() < deadline:
                    start = time.monotonic_ns()
                    values = list(pool.map(lambda d: d.read(), self.controller.devices))
                    row['samples'].append(dict(begin_ns=start, end_ns=time.monotonic_ns(), mhz=values))
                    good = len(values) == 2 and all(abs(v-target) <= 15 for v in values)
                    consecutive = consecutive+1 if good else 0
                    if consecutive == 3:
                        self.current = target
                        row.update(status='readback_verified', end_ns=time.monotonic_ns())
                        return dict(target_mhz=target, readback_verified=True)
                    time.sleep(self.interval)
            raise TimeoutError('Both clocks did not settle at boundary; no next block submitted')
        except BaseException as exc:
            row.update(status='failed', error=safe_error(exc))
            try: self.controller.reset()
            except BaseException as restore: row['restore_error'] = safe_error(restore)
            raise
        finally:
            with Path(self.journal).open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, allow_nan=False)+'\n')


class LookaheadSwitch:
    """One-ahead cancellable queue. Uses paired backend, not native GPU timestamps.

    Host timing is an explicit port choice. Physical overlap must be measured;
    this adapter never labels a queued command as an eliminated hardware stall.
    """
    def __init__(self, switch, journal):
        self.switch, self.plan, self.journal = switch, switch.plan, Path(journal)
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.cancel = threading.Event()
        self.pending = None
        self.lock = threading.Lock()

    def _queued(self, key, delay):
        if self.cancel.wait(max(0, delay)): raise RuntimeError('Lookahead queue cancelled')
        return self.switch(key)

    def __call__(self, key):
        with self.lock:
            if self.cancel.is_set(): raise RuntimeError('Lookahead queue is closed')
            begin = time.monotonic_ns()
            if self.pending:
                expected, future = self.pending
                if key != expected: raise RuntimeError('Lookahead boundary order changed')
                result = future.result(timeout=self.switch.timeout+5)
                self.pending = None
            else:
                result = self.switch(key)
            released = time.monotonic_ns()
            row = dict(key=key, boundary_ns=begin, release_ns=released,
                       boundary_wait_ns=released-begin, physical_overlap_verified=False)
            request, context, index = key
            blocks = self.plan['contexts'][context]
            if index+1 < len(blocks):
                current, upcoming = blocks[index], blocks[index+1]
                if current['mhz'] != upcoming['mhz']:
                    delay = current['estimated_s']-self.plan['switch_s']
                    if delay <= 0: raise ValueError('No positive lookahead horizon')
                    next_key = [request, context, index+1]
                    self.pending = next_key, self.pool.submit(self._queued, next_key, delay)
                    row.update(next_key=next_key, planned_delay_s=delay)
            with self.journal.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row)+'\n')
            return result

    def close(self):
        self.cancel.set()
        # Join an in-flight clock write BEFORE the owner restores clocks.
        self.pool.shutdown(wait=True, cancel_futures=True)
