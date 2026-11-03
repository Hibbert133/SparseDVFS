"""CPU-only observation tests. No vLLM installation or GPU is exercised."""
import copy
from contextlib import nullcontext
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS, ModuleType
import unittest
from unittest.mock import patch

from rtx5090 import observe
from rtx5090.observe_core import (DEVELOPMENT, GPU_UUIDS, MARKER, PhaseTracker,
    correlate, inventory, signature, validate_ranks, write_json, read_json)


def identity(rank):
    return dict(rank=rank, pid=100+rank, gpu_uuid=GPU_UUIDS[rank], tp_size=2,
                dtype='torch.bfloat16', v2_runner=False, enforce_eager=True,
                load_format='dummy', quantization=None,
                communication_env=dict(NCCL_P2P_DISABLE='1', NCCL_SHM_DISABLE='1', NCCL_IB_DISABLE='1'),
                worker=dict(sha256='test-worker'), vllm_sources_sha256={'source': 'test'})


def scheduler(new=None, computed=None, count=1, resumed=False):
    return NS(finished_req_ids=[], scheduled_new_reqs=[NS(req_id='r', prompt_token_ids=list(range(new)),
              mm_features=[], num_computed_tokens=0)] if new is not None else [],
              scheduled_cached_reqs=NS(req_ids=['r'] if computed is not None else [],
                  num_computed_tokens=[computed] if computed is not None else [],
                  resumed_req_ids={'r'} if resumed else set()),
              num_scheduled_tokens={'r': count}, total_num_scheduled_tokens=count,
              scheduled_spec_decode_tokens={}, preempted_req_ids=set())


def row(index=0, rank=0, phase='decode', opaque=False):
    return dict(marker=f'{MARKER}{rank}|{index}', rank=rank, phase=phase,
                overload='custom.fused.default' if opaque else 'aten.mm.default',
                opaque=opaque, inputs={'shape': [1, 4096]}, outputs={'shape': [1, 4096]},
                parent='model.layer', context=dict(num_computed_tokens=128,
                    num_scheduled_tokens=1, prompt_tokens=128), gpu_activities=[], stream_event_ms=.1)


def events(markers=None):
    markers = markers or [f'{MARKER}0|0']
    data = []
    for index, marker in enumerate(markers):
        data.extend([
            dict(ph='X', cat='user_annotation', name=marker, pid=10, tid=20, ts=index*20, dur=10),
            dict(ph='X', cat='cuda_runtime', name='cudaLaunchKernel', pid=10, tid=20,
                 ts=index*20+1, dur=1, args={'correlation': index+1}),
            # Deliberately nowhere near the CPU scope timestamps.
            dict(ph='X', cat='kernel', name='actual_gemm', ts=10000+index, dur=100,
                 args={'correlation': index+1, 'device': 0, 'stream': index+7}),
        ])
    return data


class CoreTests(unittest.TestCase):
    def test_rank_mapping_checks_both_devices_not_reply_order(self):
        self.assertEqual(validate_ranks([identity(1), identity(0)])[0]['rank'], 0)
        for field, bad in [('gpu_uuid', GPU_UUIDS[0]), ('pid', 100), ('tp_size', 1),
                           ('dtype', 'torch.float16'), ('load_format', 'auto'), ('v2_runner', True)]:
            with self.subTest(field=field):
                values = [identity(0), identity(1)]
                values[1][field] = bad
                with self.assertRaises(ValueError): validate_ranks(values)

    def test_partial_prefill_and_decode_use_scheduler_counts(self):
        tracker = PhaseTracker()
        first = tracker.step(scheduler(new=128, count=64))
        second = tracker.step(scheduler(computed=64, count=64))
        third = tracker.step(scheduler(computed=128, count=1))
        self.assertEqual([x['phase'] for x in (first, second, third)], ['prefill', 'prefill', 'decode'])
        self.assertEqual([x['step'] for x in (first, second, third)], [0, 1, 2])
        self.assertEqual(second['num_computed_tokens'], 64)

    def test_prefix_cache_resume_and_mixed_are_explicit(self):
        tracker = PhaseTracker()
        initial = scheduler(new=128, count=8)
        initial.scheduled_new_reqs[0].num_computed_tokens = 120
        result = tracker.step(initial)
        self.assertEqual(result['admitted_computed_tokens'], 120)
        self.assertEqual(result['phase'], 'prefill')
        result = tracker.step(scheduler(computed=127, count=2, resumed=True))
        self.assertEqual(result['phase'], 'mixed')
        self.assertTrue(result['resumed'])

    def test_unknown_cached_request_and_batch_fail_closed(self):
        with self.assertRaises(ValueError): PhaseTracker().step(scheduler(computed=128))
        value = scheduler(new=128)
        value.num_scheduled_tokens['other'] = 1
        value.total_num_scheduled_tokens = 2
        with self.assertRaises(ValueError): PhaseTracker().step(value)

    def test_multistream_correlation_preserves_kernel_identity(self):
        rows = [row(0), row(1)]
        result = correlate(events([r['marker'] for r in rows]), rows)
        self.assertTrue(result['fully_attributed'])
        self.assertEqual(result['gpu_activity_sum_us'], 200)
        self.assertEqual(rows[1]['gpu_activities'][0]['stream'], 8)
        self.assertEqual(rows[0]['gpu_activities'][0]['name'], 'actual_gemm')

    def test_unmatched_gpu_work_stays_in_denominator(self):
        rows, trace = [row()], events()
        trace.append(dict(ph='X', cat='kernel', name='nccl_other_stream', dur=100,
                          args={'correlation': 999, 'device': 0, 'stream': 20}))
        result = correlate(trace, rows)
        self.assertEqual(result['attribution_fraction'], .5)
        self.assertFalse(result['fully_attributed'])
        self.assertEqual(len(result['unattributed_activities']), 1)

    def test_gpu_annotation_is_not_duplicate_cpu_scope_or_extra_kernel_work(self):
        trace = events()
        trace.append(dict(trace[0], cat='gpu_user_annotation', pid=0,
                          tid=7, ts=10000, dur=100))
        rows = [row()]
        result = correlate(trace, rows)
        self.assertTrue(result['fully_attributed'])
        self.assertEqual(result['gpu_activities'], 1)
        self.assertEqual(result['gpu_activity_sum_us'], 100)
        self.assertEqual(len(rows[0]['gpu_activities']), 1)

    def test_real_duplicate_cpu_annotations_still_fail(self):
        trace = events()
        trace.append(dict(trace[0]))
        with self.assertRaisesRegex(ValueError, 'missing=0, duplicate=1'):
            correlate(trace, [row()])

    def test_gpu_annotation_cannot_replace_missing_cpu_scope(self):
        trace = events()
        trace[0]['cat'] = 'gpu_user_annotation'
        with self.assertRaisesRegex(ValueError, 'missing=1, duplicate=0'):
            correlate(trace, [row()])

    def test_wrong_cpu_thread_or_conflicting_correlation_not_guessed(self):
        for mutate in (lambda t: t[1].update(tid=21),
                       lambda t: t.append(dict(t[1], ts=500))):
            trace = events()
            mutate(trace)
            self.assertEqual(correlate(trace, [row()])['attribution_fraction'], 0)

    def test_missing_marker_fails_and_opaque_ops_are_not_split(self):
        with self.assertRaises(ValueError): correlate([], [row()])
        rows = [row(opaque=True)]
        trace = events()
        trace.append(dict(ph='X', cat='kernel', name='second_internal_kernel', dur=50,
                          args={'correlation': 1, 'device': 0, 'stream': 8}))
        correlate(trace, rows)
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0]['gpu_activities']), 2)

    def test_signature_separates_rank_phase_layout_and_opaque_state(self):
        original = row(opaque=True)
        for mutate in (lambda r: r.update(rank=1), lambda r: r.update(phase='prefill'),
                       lambda r: r['inputs'].update(stride=[4096, 1]),
                       lambda r: r['context'].update(num_computed_tokens=129)):
            changed = copy.deepcopy(original)
            mutate(changed)
            self.assertNotEqual(signature(original)[0], signature(changed)[0])

    def test_inventory_deduplicates_without_claiming_replay_or_holdout(self):
        rows = [row(0), row(1)]
        attribution = correlate(events([r['marker'] for r in rows]), rows)
        request = dict(split='development', status='complete', prompt_tokens=128, rank=0,
                       request_key='sample', operators=rows, attribution=attribution)
        result = inventory([request])
        self.assertEqual(result['summary']['unique_signatures'], 1)
        self.assertEqual(result['signatures'][0]['calls'], 2)
        self.assertEqual(result['summary']['equation_validated_calls'], 0)
        self.assertEqual(result['summary']['replay_supported_calls'], 0)
        request['prompt_tokens'] = 1024
        with self.assertRaises(ValueError): inventory([request])


class LauncherTests(unittest.TestCase):
    def test_offline_summary_requires_all_eighteen_rank_exports(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = dict(schema=observe.SCHEMA, status='collected', workers=[identity(0), identity(1)],
                            environment=dict(cuda_devices=[dict(uuid=x) for x in GPU_UUIDS]), requests=[])
            for n in DEVELOPMENT:
                for repeat in (1, 2, 3):
                    key = f'input-{n}-repeat-{repeat}'
                    manifest['requests'].append(dict(request_key=key, prompt_tokens=n,
                                                     status='complete', split='development'))
                    for rank in (0, 1):
                        rows = [row(rank=rank)]
                        attribution = correlate(events([rows[0]['marker']]), rows)
                        request = dict(rank=rank, identity=identity(rank), status='complete', split='development',
                                       request_key=key, prompt_tokens=n, operators=rows, attribution=attribution)
                        write_json(root/'requests'/key/f'rank-{rank}'/'request.json', request)
            write_json(root/'manifest.json', manifest)
            args = observe.parser().parse_args(['summarize', '--out', folder])
            with patch.dict(sys.modules, {'torch': None, 'vllm': None}):
                result = observe.summarize(args)
            self.assertEqual(result['summary']['requests'], 18)
            self.assertEqual(result['summary']['observed_calls'], 18)
            self.assertEqual(result['summary']['unique_signatures'], 2)
            self.assertEqual(result['summary']['equation_validated_calls'], 0)
            with self.assertRaises(FileExistsError): observe.summarize(args)

    def test_read_only_preparation_sets_reported_communication_environment(self):
        with patch.dict('os.environ', {'SPARSEDVFS_TRITON_PATH': '/existing/triton'}, clear=True), \
             patch.dict(sys.modules, {'torch': NS(cuda=NS(is_initialized=lambda: False))}):
            import os
            args = observe.parser().parse_args(['doctor', '--out', 'unused'])
            observe.prepare(args)
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], ','.join(GPU_UUIDS))
            self.assertEqual(os.environ['NCCL_P2P_DISABLE'], '1')
            self.assertEqual(os.environ['VLLM_USE_V2_MODEL_RUNNER'], '0')
            self.assertNotIn('SPARSEDVFS_APPLY', os.environ)

    def test_config_cannot_replace_dummy_or_tp_and_shape_holdout_is_sealed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_json(root/'config.json', {})
            args = observe.parser().parse_args(['doctor', '--out', folder, '--model', folder])
            self.assertEqual(observe.engine_options(args)['tensor_parallel_size'], 2)
            self.assertNotIn(1024, [x['prompt_tokens'] for x in observe.scenarios()])
            write_json(root/'extra.json', {'load_format': 'auto'})
            args.engine_config = str(root/'extra.json')
            with self.assertRaises(ValueError): observe.engine_options(args)

    def test_token_ids_valid_deterministic_and_output_length_checked(self):
        tokenizer = NS(all_special_ids=[0, 1], get_vocab=lambda: {str(i): i for i in range(100)})
        pool = observe.token_pool(tokenizer, 64)
        tokens = observe.prompt_ids(pool, 128)
        self.assertEqual(len(tokens), 128)
        self.assertTrue(all(2 <= t < 64 for t in tokens))
        self.assertEqual(tokens, observe.prompt_ids(pool, 128))
        output = NS(prompt_token_ids=tokens, request_id='r',
                    outputs=[NS(token_ids=[2]*64, finish_reason='length')])
        observe.check_output([output], tokens)
        output.outputs[0].token_ids.pop()
        with self.assertRaises(RuntimeError): observe.check_output([output], tokens)

    def test_doctor_failure_preserves_evidence_and_redacts_token(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict('os.environ', {'SPARSEDVFS_BRIDGE_TOKEN': 'secret-sentinel'}):
            args = observe.parser().parse_args(['doctor', '--out', folder])
            with patch.object(observe, 'engine_options', return_value={}), \
                 patch.object(observe, 'environment', side_effect=RuntimeError('secret-sentinel')):
                with self.assertRaises(RuntimeError): observe.doctor(args)
                original = (Path(folder)/'doctor.json').read_bytes()
                self.assertNotIn(b'secret-sentinel', original)
                with self.assertRaises(FileExistsError): observe.doctor(args)
                self.assertEqual(original, (Path(folder)/'doctor.json').read_bytes())

    def test_collect_lifecycle_has_warmups_three_repeats_and_cleanup(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as folder:
                calls = []
                class LLM:
                    def __init__(self, **kw):
                        self.llm_engine = NS(vllm_config=NS(model_config=NS(hf_text_config=NS(vocab_size=100))),
                                            engine_core=NS(shutdown=lambda: calls.append('shutdown')))
                    def get_tokenizer(self):
                        return NS(all_special_ids=[0], get_vocab=lambda: {str(i): i for i in range(100)})
                    def collective_rpc(self, name, **kw):
                        calls.append(name)
                        if name == 'observation_identity': return [identity(1), identity(0)]
                        if name == 'observation_stop': return [dict(rank=i, status='complete') for i in (0, 1)]
                        return []
                    def generate(self, prompts, params, **kw):
                        calls.append('generate')
                        if fail and calls.count('generate') == 2: raise RuntimeError('inference failed')
                        return [NS(prompt_token_ids=prompts[0]['prompt_token_ids'], request_id='r',
                                   outputs=[NS(token_ids=[2]*64, finish_reason='length')])]
                args = observe.parser().parse_args(['collect', '--out', folder])
                env = dict(observation_sources_sha256={'observe_worker.py': 'test-worker'},
                           vllm_sources_sha256={'source': 'test'})
                write_json(Path(folder)/'doctor.json', dict(status='ready', engine_options={}, environment=env))
                fake = NS(LLM=LLM, SamplingParams=lambda **kw: kw)
                with patch.dict(sys.modules, {'vllm': fake}), \
                     patch.object(observe, 'engine_options', return_value={}), \
                     patch.object(observe, 'environment', return_value=env), \
                     patch('sys.stdout', new_callable=io.StringIO):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, 'inference failed'): observe.collect(args)
                    else:
                        result = observe.collect(args)
                        self.assertEqual(len(result['requests']), 9)
                        self.assertEqual(calls.count('generate'), 12)
                        self.assertEqual(calls.count('observation_start'), 9)
                        self.assertEqual(result['holdout']['status'], 'sealed_not_collected')
                self.assertEqual(calls[-1], 'shutdown')
                self.assertIn('observation_stop', calls)
                self.assertEqual(read_json(Path(folder)/'manifest.json')['status'], 'failed' if fail else 'collected')


class ObserverTests(unittest.TestCase):
    def test_identity_rpc_contains_only_builtin_types_with_torch_version_subclass(self):
        class TorchVersion(str): pass
        name = 'rtx5090.observe_worker'
        with tempfile.TemporaryDirectory() as folder:
            package = Path(folder)/'triton'
            package.mkdir()
            source = package/'__init__.py'
            source.write_text('# fixture', encoding='utf-8')
            modules = {'torch': NS(__version__=TorchVersion('2.13.0+cu129'),
                       version=NS(cuda='12.9'), cuda=NS(get_device_properties=lambda device:
                           NS(uuid=GPU_UUIDS[0], name='test GPU'))),
                       'triton': NS(__version__='3.6.0', __file__=str(source)),
                       'vllm': NS(__version__='0.32.0'),
                       'vllm.v1.worker.gpu_worker': NS(Worker=type('Worker', (), {}))}
            with patch.dict(sys.modules, modules), patch.dict(observe.os.environ,
                    {'SPARSEDVFS_TRITON_PATH': folder, **observe.NCCL_ENV}):
                sys.modules.pop(name, None)
                try:
                    mod = importlib.import_module(name)
                    worker = mod.ObservationWorker.__new__(mod.ObservationWorker)
                    worker.rank = worker.local_rank = worker.device = 0
                    worker.use_v2_model_runner = False
                    worker._ob_sources = {}
                    worker.model_runner = NS(get_model=lambda: object())
                    worker.vllm_config = NS(
                        model_config=NS(dtype='torch.bfloat16', quantization=None, enforce_eager=True),
                        parallel_config=NS(tensor_parallel_size=2),
                        scheduler_config=NS(), cache_config=NS(), load_config=NS(load_format='dummy'))
                    with patch.object(mod, 'file_identity', return_value={'type': 'fixture'}):
                        result = worker.observation_identity()
                    def strict_builtin(value):
                        self.assertIn(type(value), (dict, list, str, int, float, bool, type(None)))
                        if type(value) is dict:
                            for key, item in value.items():
                                strict_builtin(key)
                                strict_builtin(item)
                        elif type(value) is list:
                            for item in value: strict_builtin(item)
                    strict_builtin(result)
                    self.assertEqual(result['torch_version'], '2.13.0+cu129')
                finally:
                    sys.modules.pop(name, None)

    def test_failed_finish_stops_profiler_removes_hooks_and_retains_evidence(self):
        name = 'rtx5090.observe_worker'
        modules = {'torch': NS(), 'vllm': NS(),
                   'vllm.v1.worker.gpu_worker': NS(Worker=type('Worker', (), {}))}
        with tempfile.TemporaryDirectory() as folder, patch.dict(sys.modules, modules):
            sys.modules.pop(name, None)
            mod = importlib.import_module(name)
            worker = mod.ObservationWorker.__new__(mod.ObservationWorker)
            worker.rank, worker._ob_active = 0, True
            worker._ob_config = dict(request_key='sample', prompt_tokens=128)
            worker._ob_root, worker._ob_steps = Path(folder), []
            worker.observation_identity = lambda: identity(0)
            calls = []
            def fail(): raise RuntimeError('synchronize failed')
            worker._ob_observer = NS(rows=[row()], finish=fail, close=lambda: calls.append('remove'))
            worker._ob_profiler = NS(stop=lambda: calls.append('stop'))
            with self.assertRaisesRegex(RuntimeError, 'synchronize failed'):
                worker.observation_stop()
            self.assertEqual(calls, ['stop', 'remove'])
            result = read_json(Path(folder)/'request.json')
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(len(result['operators']), 1)
            self.assertFalse(worker._ob_active)
            sys.modules.pop(name, None)

    def test_observer_runs_original_once_and_syncs_only_at_request_boundary(self):
        log, handles = [], []
        class Tensor: pass
        class Event:
            def __init__(self, **kw): pass
            def record(self, stream): log.append('record')
            def elapsed_time(self, other): return .25
        class Model:
            def named_modules(self): return [('', self)]
            def register_forward_pre_hook(self, fn):
                handle = NS(remove=lambda: log.append('remove'))
                handles.append(handle)
                return handle
            def register_forward_hook(self, fn, **kw): return self.register_forward_pre_hook(fn)
        fake_torch = NS(Tensor=Tensor, cuda=NS(Event=Event,
            current_stream=lambda: NS(cuda_stream=123), synchronize=lambda: log.append('sync')),
            profiler=NS(record_function=lambda name: nullcontext()))
        modules = {'torch': fake_torch, 'vllm': NS(),
                   'vllm.v1.worker.gpu_worker': NS(Worker=type('Worker', (), {}))}
        name = 'rtx5090.observe_worker'
        with patch.dict(sys.modules, modules):
            sys.modules.pop(name, None)
            mod = importlib.import_module(name)
            observer = mod.Observer(Model(), 0, lambda: dict(phase='prefill'), 100)
            def original(a, **kw):
                log.append('original')
                return a+1
            self.assertEqual(observer.dispatch(original, (2,), {}), 3)
            self.assertEqual(log.count('original'), 1)
            self.assertNotIn('sync', log)
            self.assertEqual(observer.rows[0]['outputs'], 3)
            observer.finish()
            observer.close()
            self.assertEqual(log.count('sync'), 1)
            self.assertEqual(log.count('remove'), 2)
            self.assertEqual(observer.rows[0]['stream_event_ms'], .25)
            sys.modules.pop(name, None)


if __name__ == '__main__': unittest.main()
