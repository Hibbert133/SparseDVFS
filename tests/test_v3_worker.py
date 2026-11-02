"""Exercise real adapter control flow with small fake modules/events, not fake benchmarks."""
import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch, Mock
from test_v3_contracts import scheduler
from sparsedvfs_vllm.contracts import UNITS
from sparsedvfs_vllm.spans import Spans


class Module:
    def __init__(self, function=lambda: None):
        self.pre, self.post, self.function = [], [], function

    def register_forward_pre_hook(self, hook):
        self.pre.append(hook)
        return NS(remove=lambda: self.pre.remove(hook))

    def register_forward_hook(self, hook):
        self.post.append(hook)
        return NS(remove=lambda: self.post.remove(hook))

    def __call__(self):
        for hook in self.pre:
            hook(self, ())
        output = self.function()
        for hook in self.post:
            hook(self, (), output)
        return output


class Event:
    def record(self, stream):
        self.stream = stream.cuda_stream
    def query(self):
        return True
    def elapsed_time(self, other):
        return 0.1


def fake_torch():
    return NS(__version__='test', version=NS(cuda='13.0'), cuda=NS(
        Event=lambda **kw: Event(), current_stream=lambda: NS(cuda_stream=42), synchronize=lambda: None,
        is_current_stream_capturing=lambda: False, get_device_name=lambda: 'mock GPU'))


class Qwen3_5Model:
    def __init__(self):
        self.start_layer, self.end_layer = 0, 64
        self.embed_tokens, self.norm = Module(), Module()
        self.layers = [Module() for _ in range(64)]
        for i, layer in enumerate(self.layers):
            layer.layer_type = 'full_attention' if i % 4 == 3 else 'linear_attention'


class Model:
    def __init__(self):
        self.language_model = NS(model=Qwen3_5Model())
        self.logits_calls = 0
        self.lm_head = Module(lambda: (_ for _ in ()).throw(AssertionError('lm_head.forward must not be called')))

    def compute_logits(self, *args, **kwargs):
        self.logits_calls += 1
        return 'logits'

    def forward(self):
        text = self.language_model.model
        # This also represents embedding being executed during vLLM preprocessing.
        text.embed_tokens()
        for layer in text.layers:
            layer()
        text.norm()
        return self.compute_logits('hidden')


class SpanTests(unittest.TestCase):
    def test_startup_inactive_then_all_units_in_actual_order(self):
        model = Model()
        trace = Spans(model, fake_torch(), lambda: None)
        model.forward()
        self.assertEqual(trace.events, [])
        self.assertEqual(trace.rows, [])
        trace.start(dict(phase='prefill', step=0, request_id='r'), 'test')
        model.forward()
        trace.begin('sample')
        trace.end('sample')
        trace.finish()
        result = trace.export()
        self.assertEqual(result['steps'][0]['order'], UNITS)
        self.assertEqual(len(result['rows']), 68)
        self.assertEqual(model.logits_calls, 2)
        trace.remove()

    def test_missing_sampling_fails_coverage(self):
        model = Model()
        trace = Spans(model, fake_torch(), lambda: None)
        trace.start(dict(phase='prefill', step=0, request_id='r'), 'x')
        model.forward()
        with self.assertRaisesRegex(RuntimeError, 'Incomplete'):
            trace.finish()

    def test_duplicate_out_of_order_and_capture_rejected(self):
        for kind in ('order', 'capture'):
            torch = fake_torch()
            trace = Spans(Model(), torch, lambda: None)
            trace.start(dict(phase='prefill', step=0, request_id='r'), 'x')
            if kind == 'capture':
                torch.cuda.is_current_stream_capturing = lambda: True
            with self.assertRaises(RuntimeError):
                trace.begin('embed' if kind == 'capture' else 'logits')

    def test_stream_change_is_rejected(self):
        torch = fake_torch()
        trace = Spans(Model(), torch, lambda: None)
        trace.start(dict(phase='prefill', step=0, request_id='r'), 'x')
        trace.begin('embed')
        torch.cuda.current_stream = lambda: NS(cuda_stream=43)
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            trace.end('embed')


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        torch = fake_torch()
        class Base:
            def load_model(self, *, load_dummy_weights=False):
                self.dummy = load_dummy_weights
            def execute_model(self, output):
                self.model_runner.get_model().forward()
                return None
            def sample_tokens(self, grammar_output):
                return 'sampled'
        modules = {'torch': torch, 'vllm': NS(__version__='0.19.0'),
                   'vllm.v1': ModuleType('vllm.v1'), 'vllm.v1.worker': ModuleType('vllm.v1.worker'),
                   'vllm.v1.worker.gpu_worker': NS(Worker=Base)}
        self.patch_modules = patch.dict(sys.modules, modules)
        self.patch_modules.start()
        sys.modules.pop('sparsedvfs_vllm.worker', None)
        self.module = importlib.import_module('sparsedvfs_vllm.worker')
        self.worker = self.module.SparseDVFSWorker()
        self.worker.vllm_config = None
        self.worker.use_v2_model_runner = False
        model = Model()
        self.worker.model_runner = NS(get_model=lambda: model)
        with patch.object(self.module, 'validate_engine'), patch.object(self.module.signal, 'signal'), \
                patch.object(self.module.signal, 'SIGUSR2', 12, create=True), \
                patch.object(self.module.atexit, 'register'), patch.dict('os.environ', {'SPARSEDVFS_WORKER_CONFIG': ''}):
            self.worker.load_model(load_dummy_weights=True)
        self.model = model
        self.verify = patch.object(self.module, 'verify_model', return_value={'manifest_sha256': 'mock'})
        self.verify.start()
        self.config = dict(mode='instrumented', enabled=True, workload={}, out_dir=self.temp.name, apply=False)
        self.worker.sdvfs_configure(self.config)

    def tearDown(self):
        self.verify.stop()
        self.patch_modules.stop()
        sys.modules.pop('sparsedvfs_vllm.worker', None)
        self.temp.cleanup()

    def test_execute_and_sample_are_one_complete_step(self):
        self.assertTrue(self.worker.dummy)
        self.assertIsNone(self.worker.execute_model(scheduler(prompt=3, count=3)))
        self.assertIsNotNone(self.worker._sd_trace.active)
        self.assertEqual(len(self.worker._sd_trace.order), 67)
        self.assertEqual(self.worker.sample_tokens(None), 'sampled')
        result = self.worker.sdvfs_export(reset=True)
        self.assertEqual(len(result['rows']), 68)
        self.assertEqual(self.worker._sd_trace.steps, [])

    def test_warmup_request_does_not_activate_hooks_or_controller(self):
        self.worker.sdvfs_configure(dict(self.config, enabled=False))
        self.worker.execute_model(scheduler(prompt=3, count=3))
        self.worker.sample_tokens(None)
        self.assertEqual(self.worker._sd_trace.rows, [])
        self.assertIsNone(self.worker._sd_fuse)

    def test_busy_configure_rejected(self):
        self.worker.execute_model(scheduler(prompt=3, count=3))
        with self.assertRaisesRegex(RuntimeError, 'idle'):
            self.worker.sdvfs_configure(self.config)

    def test_original_worker_failure_requests_cleanup(self):
        with patch.object(self.model, 'forward', side_effect=RuntimeError('forced')), \
                patch.object(self.worker, '_sd_close_controller') as closed:
            with self.assertRaisesRegex(RuntimeError, 'forced'):
                self.worker.execute_model(scheduler(prompt=3, count=3))
            closed.assert_called_once()

    def test_idle_close_handshake_exports_and_acknowledges(self):
        self.worker.execute_model(scheduler(prompt=3, count=3))
        self.worker.sample_tokens(None)
        (Path(self.temp.name) / 'close_request.json').write_text(json.dumps({'nonce': 'test', 'action': 'close'}))
        self.worker._sd_request_close(None, None)
        ack = json.loads((Path(self.temp.name) / 'close_ack.json').read_text())
        self.assertEqual(ack, dict(nonce='test', closed=True))
        self.assertTrue((Path(self.temp.name) / 'worker_trace.json').is_file())

    def test_profile_warmup_preserves_worker_owned_fixed_controller(self):
        controller = NS(set=Mock())
        self.worker._sd_cfg.update(mode='fixed', enabled=False, apply=True, keep_fixed_during_warmup=True)
        self.worker._sd_fuse = controller
        with patch.object(self.worker, '_sd_close_controller') as close:
            self.worker.execute_model(scheduler(prompt=3, count=3))
            self.worker.sample_tokens(None)
            self.assertEqual(self.worker._sd_trace.rows, [])
            self.worker.sdvfs_observe(True, 'measured')
            self.worker.execute_model(scheduler(rid='next', prompt=3, count=3, finished=['r']))
            self.worker.sample_tokens(None)
            self.assertEqual(len(self.worker.sdvfs_export()['rows']), 68)
            close.assert_not_called()
            controller.set.assert_not_called()  # warmup does not repeatedly change clocks

    def test_profile_opp_failure_attempts_restore(self):
        self.worker._sd_cfg['mode'] = 'fixed'
        self.worker._sd_fuse = NS(set=Mock(side_effect=RuntimeError('write failed')))
        with patch.object(self.worker, '_sd_close_controller') as close:
            with self.assertRaisesRegex(RuntimeError, 'write failed'):
                self.worker.sdvfs_profile_opp([1, 2, 3])
            close.assert_called_once()

    def test_export_failure_still_closes_controller(self):
        with patch.object(self.worker._sd_trace, 'export', side_effect=RuntimeError('GPU failed')), \
                patch.object(self.worker, '_sd_close_controller') as close:
            with self.assertRaisesRegex(RuntimeError, 'GPU failed'):
                self.worker.sdvfs_export(close=True)
            close.assert_called_once()

    def test_apply_requires_launcher_flag_and_boolean_config(self):
        with patch.dict('os.environ', {'SPARSEDVFS_APPLY': '0'}):
            with self.assertRaisesRegex(ValueError, 'explicit --apply'):
                self.worker.sdvfs_configure(dict(self.config, mode='fixed', apply=True))
        with self.assertRaisesRegex(ValueError, 'JSON boolean'):
            self.worker.sdvfs_configure(dict(self.config, apply='false'))

    def test_operator_apply_requires_coverage_before_controller_creation(self):
        with patch.dict('os.environ', {'SPARSEDVFS_APPLY': '1'}), patch.object(self.module, 'FUSE') as controller:
            with self.assertRaisesRegex(ValueError, 'coverage certificate'):
                self.worker.sdvfs_configure(dict(self.config, granularity='operator', mode='fixed', apply=True))
            controller.assert_not_called()

    def test_operator_observation_can_be_selected_without_clock_writes(self):
        with patch.object(self.module, 'FUSE') as controller:
            self.worker.sdvfs_configure(dict(self.config, granularity='operator'))
            self.assertEqual(self.worker._sd_granularity, 'operator')
            self.assertEqual(type(self.worker._sd_trace).__name__, 'OperatorSpans')
            controller.assert_not_called()


if __name__ == '__main__':
    unittest.main()
