"""Read-only telemetry in an independent spawned process.

Keeps HTTP parsing/GC and sampling out of the same interpreter. Does not fix
Windows/WSL transport stalls, relax gaps, or discard slow queries.
"""
import multiprocessing
import os

from .service_telemetry import Sampler, open_devices


def _error(exc):
    value = f'{type(exc).__name__}: {exc}'
    for key in ('SPARSEDVFS_BRIDGE_TOKEN', 'OPENAI_API_KEY'):
        if os.environ.get(key): value = value.replace(os.environ[key], '[REDACTED]')
    return value


def _sampling_process(connection, config, interval):
    devices, sampler, errors = [], None, []
    finished = False
    try:
        devices = open_devices(config)
        sampler = Sampler(devices, interval)
        sampler.start()
        connection.send(dict(kind='ready', pid=os.getpid()))
        if connection.recv() != 'finish':
            raise ValueError('Unexpected sampler control message')
        sampler.finish()
        finished = True
    except BaseException as exc:
        errors.append(_error(exc))
    finally:
        if sampler is not None and sampler.thread is not None and not finished:
            try: sampler.finish()
            except BaseException as exc: errors.append(_error(exc))
        for device in devices:
            try: device.close()
            except BaseException as exc: errors.append(_error(exc))
        try:
            connection.send(dict(kind='finished', pid=os.getpid(),
                frames=sampler.frames if sampler else [],
                errors=errors+(sampler.errors if sampler else [])))
        except (BrokenPipeError, EOFError, OSError):
            pass
        connection.close()


class ProcessSampler:
    def __init__(self, config, interval_s=.1):
        self.config, self.interval_s = config, interval_s
        self.frames, self.errors = [], []
        self.thread, self.connection, self.child_pid = None, None, None
        self._finished = False

    def _receive(self, message):
        if message.get('kind') != 'finished':
            raise RuntimeError('Invalid sampler result')
        self.frames = message['frames']
        self.errors.extend(message['errors'])
        self.child_pid = message['pid']
        self._finished = True

    def start(self):
        ctx = multiprocessing.get_context('spawn')
        self.connection, child = ctx.Pipe()
        self.thread = ctx.Process(target=_sampling_process,
            args=(child, self.config, self.interval_s), daemon=True)
        self.thread.start()
        child.close()
        try:
            if not self.connection.poll(15):
                raise TimeoutError('No initial isolated dual-card telemetry frame')
            message = self.connection.recv()
            if message.get('kind') == 'finished':
                self._receive(message)
                raise RuntimeError('; '.join(self.errors) or 'Sampler exited before ready')
            if message.get('kind') != 'ready': raise RuntimeError('Invalid sampler readiness')
            self.child_pid = message['pid']
            return self
        except BaseException as exc:
            self.errors.append(_error(exc))
            self._cleanup()
            raise

    def _cleanup(self):
        if self.thread is not None:
            self.thread.join(timeout=2)
            if self.thread.is_alive():
                self.thread.terminate()
                self.thread.join(timeout=2)
                self.errors.append('Isolated sampler terminated; no valid complete sampling certificate')
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        self._finished = True

    def finish(self):
        try:
            if not self._finished and self.connection is not None:
                self.connection.send('finish')
                if not self.connection.poll(35):
                    raise TimeoutError('Isolated sampler finish timed out')
                self._receive(self.connection.recv())
        except BaseException as exc:
            self.errors.append(_error(exc))
        finally:
            self._cleanup()
        return self.frames
