"""Exercise the new worker state machine with fake CUDA, not target acceptance."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from rtx5090 import eager_core as core
from rtx5090.observe_core import digest


class Event:
    def record(self): pass
    def elapsed_time(self, other): return .1


class Base:
    def execute_model(self, scheduler_output):
        for name in core.UNITS[:-1]:
            self._eg_begin(name); self._eg_end(name)
        return None

    def sample_tokens(self, grammar_output): return 'result'


def scheduler(step):
    return NS(finished_req_ids=[], scheduled_new_reqs=[NS(req_id='r', prompt_token_ids=[7]*512,
        mm_features=[], num_computed_tokens=0)] if step == 0 else [],
        scheduled_cached_reqs=NS(req_ids=['r'] if step else [],
            num_computed_tokens=[511+step] if step else [], resumed_req_ids=set()),
        num_scheduled_tokens={'r': 512 if step == 0 else 1}, total_num_scheduled_tokens=512 if step == 0 else 1,
        scheduled_spec_decode_tokens={}, preempted_req_ids=set())


class WorkerTests(unittest.TestCase):
    def worker(self, root):
        fake = NS(cuda=NS(Event=lambda **k: Event(), current_stream=lambda: NS(cuda_stream=5),
            is_current_stream_capturing=lambda: False, synchronize=Mock()))
        observer = ModuleType('rtx5090.observe_worker'); observer.ObservationWorker = Base
        path = Path(__file__).resolve().parents[1]/'rtx5090/eager_worker.py'
        spec = importlib.util.spec_from_file_location('rtx5090.eager_worker_fixture', path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'torch': fake, 'rtx5090.observe_worker': observer}):
            spec.loader.exec_module(module)
        worker = module.EagerWorker()
        worker.rank, worker.device = 0, 0
        worker._eg_root, worker._eg_session = Path(root), 's'
        worker._eg_lock = threading.RLock()
        worker._eg_running, worker._eg_context, worker._eg_poisoned = False, None, None
        worker._eg_mode, worker._eg_pending = 'off', None
        worker._eg_steps, worker._eg_identity = [], {'rank': 0}
        return worker, fake

    def test_complete_request_exports_then_returns_off(self):
        with tempfile.TemporaryDirectory() as root:
            worker, fake = self.worker(root)
            worker._eg_command(dict(op='arm', session='s', mode='units', out=root,
                                    request_id='r', prompt_hash=digest([7]*512)))
            for step in range(64):
                worker.execute_model(scheduler(step))
                self.assertEqual(worker.sample_tokens(None), 'result')
            artifact = json.loads((Path(root)/'rank-0-execution.json').read_text())
            self.assertEqual(len(artifact['rows']), 64*68)
            self.assertEqual(artifact['contexts'], ['prefill']+[f'decode.{i}' for i in range(63)])
            self.assertEqual(worker._eg_mode, 'off')
            self.assertFalse(worker._eg_running)
            fake.cuda.synchronize.assert_called_once_with(0)

    def test_unrelated_prompt_poisoned_before_execution(self):
        with tempfile.TemporaryDirectory() as root:
            worker, fake = self.worker(root)
            worker._eg_command(dict(op='arm', session='s', mode='units', out=root,
                                    request_id='r', prompt_hash='wrong'))
            with self.assertRaises(RuntimeError): worker.execute_model(scheduler(0))
            self.assertIsNotNone(worker._eg_poisoned)
            self.assertTrue((Path(root)/'rank-0-failure.json').exists())
            fake.cuda.synchronize.assert_not_called()

    def test_cannot_reconfigure_mid_request(self):
        with tempfile.TemporaryDirectory() as root:
            worker, _ = self.worker(root)
            worker._eg_command(dict(op='arm', session='s', mode='units', out=root,
                                    request_id='r', prompt_hash=digest([7]*512)))
            worker.execute_model(scheduler(0)); worker.sample_tokens(None)
            with self.assertRaises(RuntimeError): worker._eg_command(dict(op='arm',session='s',mode='off'))

    def test_disarm_cancels_pending_without_cuda(self):
        with tempfile.TemporaryDirectory() as root:
            worker, fake = self.worker(root)
            worker._eg_command(dict(op='arm',session='s',mode='units',out=root))
            worker._eg_command(dict(op='arm',session='s',mode='off'))
            self.assertIsNone(worker._eg_pending)
            fake.cuda.synchronize.assert_not_called()


if __name__ == '__main__': unittest.main()
