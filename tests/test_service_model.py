"""CPU fixtures only: not target vLLM/CUDA/NVML/energy acceptance."""
import asyncio
import copy
from contextlib import redirect_stdout
import io
import gzip
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch

from rtx5090 import service_core as core
from rtx5090 import service_client as client
from rtx5090 import service_model as cli
from rtx5090 import service_telemetry as telemetry
from rtx5090.bridge import Session
from rtx5090.observe_core import GPU_UUIDS, digest


def dataset(split='train'):
    windows = []
    for length in core.LENGTHS:
        for repeat in range(3):
            t, e = 5+length/512, 500+length/512*100
            windows.append(dict(status='valid', quality_errors=[], cache_status='verified_miss',
                energy_relative_uncertainty=.01, requests=2, frequency='current', length=length,
                ttft_s=.1+length/512*.01, tpot_s=.01+length/512*.001, window_s=t,
                energy_j=e, energy_by_uuid_j={GPU_UUIDS[0]: e/2, GPU_UUIDS[1]: e/2},
                clock_ranges={u: {'core': [2400, 2500]} for u in GPU_UUIDS},
                prompt_hashes=[f'{split}-p-{length}-{repeat}-{i}' for i in range(2)],
                request_ids=[f'{split}-r-{length}-{repeat}-{i}' for i in range(2)]))
    return dict(schema=core.SCHEMA, status='collected', split=split, synthetic=False,
        identity_ready=True, identity_fingerprint='same-live-boot', seed=1 if split == 'train' else 2,
        protocol=dict(lengths=list(core.LENGTHS), output_tokens=64, concurrency=1,
                      requests_per_window=2, windows_per_cell=3, frequencies=['current']),
        windows=windows, restoration=dict(required=False, verified=False),
        temperature_range_c={u: [45, 55] for u in GPU_UUIDS})


def trace_fixture(tail_missing=False, graph=True):
    events = []
    for i in range(3):
        events.append(dict(ph='X', cat='user_annotation', name='execute_context_0(0)_generation_1(1)',
                           pid=10, tid=10, ts=i*100, dur=80))
        if tail_missing and i == 2: continue
        events.append(dict(ph='X', cat='cuda_runtime', name='cudaGraphLaunch' if graph else 'cudaLaunchKernel',
                           pid=10, tid=10, ts=i*100+10, dur=5, args={'correlation': i}))
        for k in range(4):
            # GPU timestamps deliberately outside CPU scopes: only IDs may associate.
            events.append(dict(ph='X', cat='kernel', name=f'kernel{k}', ts=90000+i*100+k,
                               dur=10, args={'correlation': i, 'graph id': 30}))
    return {'traceEvents': events}


def sample(uuid, t, energy=None, power=100000, clock=2450, temp=50):
    fields = dict(energy_mj=t*100000 if energy is None else energy,
                  power_mw=power, clock_mhz=clock, temperature_c=temp,
                  memory_clock_mhz=13801, utilization_pct=95)
    return dict(uuid=uuid, raw=dict(uuid=uuid, synthetic=False, **fields),
                mapped={k: dict(t_ns=t*1e9, uncertainty_ns=10000) for k in fields})


def frames(duration=30, power_only=False):
    result = []
    for i in range(int(duration*10)+1):
        row = [sample(u, i/10, power=100000+100*(i%3)) for u in GPU_UUIDS]
        if power_only:
            for s in row: s['raw']['energy_mj'] = None
        result.append(row)
    return result


class Models(unittest.TestCase):
    def test_nnls_exact(self):
        self.assertEqual(core.nnls_line([1, 2, 3], [5, 7, 9]), {'intercept': 3, 'slope': 2})

    def test_nnls_boundary(self):
        m = core.nnls_line([1, 2, 3], [3, 2, 1])
        self.assertEqual(m, {'intercept': 2, 'slope': 0})

    def test_independent_validation(self):
        train, val = dataset(), dataset('validation')
        model = core.fit(train)
        val['frozen_model_sha256'] = digest(model)
        frozen = copy.deepcopy(model)
        result = core.validate(val, model)
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(model, frozen)
        self.assertEqual(result['summary']['energy_j']['wape'], 0)
        self.assertFalse(result['graph_partition_ready'])

    def test_not_passed_when_error_large(self):
        model, val = core.fit(dataset()), dataset('validation')
        val['frozen_model_sha256'] = digest(model)
        for r in val['windows']:
            r['energy_j'] *= 2
            r['energy_by_uuid_j'] = {u: r['energy_j']/2 for u in GPU_UUIDS}
        self.assertEqual(core.validate(val, model)['status'], 'not_passed')

    def test_temperature_outside_scope(self):
        model, val = core.fit(dataset()), dataset('validation')
        val['frozen_model_sha256'] = digest(model)
        val['temperature_range_c'][GPU_UUIDS[0]][1] = 60
        self.assertEqual(core.validate(val, model)['status'], 'not_passed')

    def test_current_frequency_outside_scope(self):
        model, val = core.fit(dataset()), dataset('validation')
        val['frozen_model_sha256'] = digest(model)
        val['windows'][0]['clock_ranges'][GPU_UUIDS[0]]['core'][1] = 2700
        self.assertEqual(core.validate(val, model)['status'], 'not_passed')

    def test_cli_freeze_then_validate_with_new_file(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            core.write_new(root/'train.json', dataset())
            self.assertEqual(cli.main(['fit', '--data', str(root/'train.json'), '--out', str(root/'fit')]), 0)
            model = core.read(root/'fit/model.json')
            val = dataset('validation'); val['frozen_model_sha256'] = digest(model)
            core.write_new(root/'val.json', val)
            self.assertEqual(cli.main(['validate', '--data', str(root/'val.json'),
                '--model', str(root/'fit/model.json'), '--out', str(root/'validate')]), 0)
            self.assertEqual(core.read(root/'validate/validation.json')['status'], 'passed')
            self.assertEqual(core.read(root/'fit/model.json'), model)

    def test_reject_fit_invalid_data(self):
        mutations = [lambda d: d.update(split='validation'), lambda d: d.update(synthetic=True),
            lambda d: d.update(status='failed'), lambda d: d.update(identity_ready=False),
            lambda d: d['windows'].pop(),
            lambda d: d['windows'][0].update(cache_status='unknown'),
            lambda d: d['windows'][0].update(length=1024),
            lambda d: d['windows'][0].update(energy_relative_uncertainty=.04),
            lambda d: d['windows'][0].update(energy_relative_uncertainty=float('nan')),
            lambda d: d.update(restoration=dict(required=True, verified=False)),
            lambda d: d['windows'][0].update(ttft_s=float('nan')),
            lambda d: d['windows'][0].update(quality_errors=['clock drift']),
            lambda d: d['windows'][0].update(energy_by_uuid_j={GPU_UUIDS[0]: 525})]
        for change in mutations:
            with self.subTest(change=change):
                data = dataset(); change(data)
                with self.assertRaises(ValueError): core.fit(data)

    def test_reject_validation_leakage_and_protocol_changes(self):
        model = core.fit(dataset())
        for change in (lambda d: d.update(seed=1),
                       lambda d: d.update(identity_fingerprint='restarted'),
                       lambda d: d.update(frozen_model_sha256='modified'),
                       lambda d: d['windows'][0]['prompt_hashes'].__setitem__(0, model['training_prompt_hashes'][0]),
                       lambda d: d['protocol'].update(requests_per_window=20)):
            val = dataset('validation'); val['frozen_model_sha256'] = digest(model); change(val)
            with self.assertRaises(ValueError): core.validate(val, model)

    def test_write_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'original.json'
            core.write_new(path, {'original': True})
            with self.assertRaises(FileExistsError): core.write_new(path, {})
            self.assertEqual(core.read(path), {'original': True})


class Trace(unittest.TestCase):
    def test_actual_rank_filenames_and_legacy_tp_names(self):
        for naming in ('rank', 'tp'):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                for rank in (0, 1):
                    (root/f'svc_128_prefill_{naming}{rank}.1793848827164411804.pt.trace.json.gz').touch()
                    (root/f'svc_128_prefill_{naming}{rank}.profiler_out.txt').touch()
                self.assertEqual(set(cli.rank_trace_files(root, 'svc_128_prefill')), {0, 1})
                (root/'svc_128_prefill_rank0.duplicate.trace.json').touch()
                with self.assertRaisesRegex(ValueError, 'Multiple'):
                    cli.rank_trace_files(root, 'svc_128_prefill')

    def test_wrong_prefix_or_rank_does_not_fill_missing_card(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('svc_rank0.1.pt.trace.json', 'svc_other_rank1.1.pt.trace.json',
                         'svc_rank10.1.pt.trace.json'):
                (root/name).touch()
            with self.assertRaisesRegex(ValueError, 'matched ranks=\\[0\\]'):
                cli.rank_trace_files(root, 'svc')

    def test_offline_recovery_keeps_failed_manifest_and_contacts_no_service(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(client, 'http', side_effect=AssertionError('HTTP forbidden')):
            root = Path(tmp)
            original = dict(schema=core.SCHEMA, status='failed', windows=[dict(prefix='svc',
                length=128, requested_phase='decode_early', max_iterations=3, cache_status='verified_miss')])
            core.write_new(root/'original.json', original)
            for rank in (0, 1):
                with gzip.open(root/f'svc_rank{rank}.123.pt.trace.json.gz', 'wt', encoding='utf-8') as f:
                    json.dump(trace_fixture(tail_missing=rank == 1), f)
            result = cli.analyze_observation(NS(observation=str(root/'original.json'), trace_dir=str(root), out=str(root/'analysis')))
            self.assertEqual(result['status'], 'not_passed')
            self.assertEqual(set(result['windows'][0]['ranks']), {'0', '1'})
            self.assertEqual(core.read(root/'original.json'), original)
            self.assertTrue((root/'analysis/svc-rank-1.json').is_file())

    def test_graph_one_launch_many_kernels(self):
        result = core.correlate_trace(trace_fixture(), 3)
        self.assertEqual(result['status'], 'boundary_checked')
        self.assertEqual(result['gpu_activities'], 12)
        self.assertEqual(result['attribution_fraction'], 1)
        self.assertTrue(result['graph_replay_observed'])
        self.assertFalse(result['full_operator_coverage_verified'])
        self.assertTrue(all(r['shape'] is None for r in result['signatures']))

    def test_tail_loss_rejects_even_high_fraction(self):
        result = core.correlate_trace(trace_fixture(tail_missing=True), 3)
        self.assertEqual(result['attribution_fraction'], 1)
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(result['iterations_with_gpu'], 2)

    def test_wrong_iteration_count(self):
        self.assertEqual(core.correlate_trace(trace_fixture(), 4)['status'], 'incomplete')

    def test_conflicting_correlations_unknown(self):
        trace = trace_fixture()
        trace['traceEvents'].append(dict(ph='X', cat='cuda_driver', name='cuLaunchKernel',
            pid=42, tid=42, ts=10, dur=1, args={'correlation': 0}))
        result = core.correlate_trace(trace, 3)
        self.assertAlmostEqual(result['attribution_fraction'], 2/3)
        self.assertEqual(result['status'], 'incomplete')

    def test_dropped_metadata(self):
        trace = trace_fixture(); trace['dropped_events'] = 1
        self.assertEqual(core.correlate_trace(trace, 3)['status'], 'incomplete')

    def test_missing_gpu_not_zero_cost(self):
        self.assertEqual(core.correlate_trace({'traceEvents': []}, 1)['status'], 'incomplete')


class Telemetry(unittest.TestCase):
    def test_counter_two_cards_same_window(self):
        energy = telemetry.energy_window(frames(), int(.5e9), int(29.5e9))
        self.assertAlmostEqual(energy['energy_j'], 5800)
        self.assertAlmostEqual(energy['power_w'], 200)
        self.assertFalse(energy['quality_errors'])

    def test_counter_reset_rejected(self):
        data = frames(); data[100][0]['raw']['energy_mj'] = 0
        with self.assertRaisesRegex(ValueError, 'reset'): telemetry.energy_window(data, int(.5e9), int(29.5e9))

    def test_power_fallback_resolution(self):
        e = telemetry.energy_window(frames(power_only=True), int(.5e9), int(29.5e9))
        self.assertAlmostEqual(e['power_w'], 200.2, places=1)
        self.assertTrue(all(v['source'] == 'power' for v in e['by_uuid'].values()))
        self.assertFalse(e['quality_errors'])

    def test_short_window_not_fitting_label(self):
        e = telemetry.energy_window(frames(2, power_only=True), int(.1e9), int(1.9e9))
        self.assertTrue(e['quality_errors'])

    def test_intermittent_counter_not_silent_power_fallback(self):
        data = frames(); data[3][0]['raw']['energy_mj'] = None
        with self.assertRaisesRegex(ValueError, 'Intermittent'):
            telemetry.energy_window(data, int(.5e9), int(29.5e9))

    def test_gap_and_missing_boundary(self):
        data = frames(); del data[30:40]
        self.assertTrue(telemetry.energy_window(data, int(.5e9), int(29.5e9))['quality_errors'])
        with self.assertRaisesRegex(ValueError, 'bracketed'):
            telemetry.energy_window(frames(), -1, int(29e9))

    def test_wrong_uuid_no_energy_sum(self):
        data = frames()
        for row in data: row[1]['uuid'] = 'other'
        with self.assertRaises(ValueError): telemetry.energy_window(data, int(.5e9), int(29.5e9))

    def test_remote_mapping_is_offset_invariant(self):
        class Device:
            uuid = GPU_UUIDS[0]
            def telemetry(self):
                return dict(uuid=self.uuid, synthetic=False, timestamp_ns=1000000,
                    field_query_times_ns={'energy_mj': dict(start=1000010, end=1000050)}, energy_mj=1)
        result = telemetry.sample_device(Device(), now=iter([5000000, 5000100]).__next__)
        self.assertEqual(result['clock_offset_interval_ns'], [4000000, 4000050])
        self.assertEqual(result['mapped']['energy_mj']['t_ns'], 5000055)
        self.assertEqual(result['mapped']['energy_mj']['uncertainty_ns'], 45)

    def test_missing_remote_timestamp_rejected(self):
        class Device:
            uuid = GPU_UUIDS[0]
            def telemetry(self): return dict(uuid=self.uuid, synthetic=False)
        with self.assertRaisesRegex(ValueError, 'timestamps'): telemetry.sample_device(Device())

    def test_idle_readback_is_not_restore_proof(self):
        after = frames(1)
        for f in after:
            for s in f: s['raw'].update(utilization_pct=0, clock_mhz=405)
        self.assertFalse(telemetry.restoration_evidence(frames(1), after, 2250)['verified'])
        self.assertTrue(telemetry.restoration_evidence(frames(1), frames(1), 2250)['verified'])

    def test_clock_drift_and_memory_change(self):
        self.assertTrue(telemetry.clock_errors(frames(1), target=2250))
        self.assertTrue(telemetry.clock_errors(frames(1), memory_reference={u: 13000 for u in GPU_UUIDS}))


class Clocks(unittest.TestCase):
    def device(self, index, fail_set=False, fail_reset=False):
        class Device:
            uuid = GPU_UUIDS[index]
            calls = []
            def set(self, mhz):
                self.calls.append(('set', mhz))
                if fail_set: raise RuntimeError('set failed')
            def reset(self):
                self.calls.append(('reset',))
                if fail_reset: raise RuntimeError('restore failed')
        return Device()

    def test_apply_mandatory(self):
        devices = [self.device(0), self.device(1)]
        with self.assertRaises(ValueError): telemetry.DualClocks(devices).set(2250)
        self.assertFalse(devices[0].calls)

    def test_partial_set_resets_both(self):
        devices = [self.device(0), self.device(1, fail_set=True)]
        control = telemetry.DualClocks(devices, True)
        try:
            control.set(2250)
        except RuntimeError:
            control.reset()
        self.assertTrue(all(('reset',) in d.calls for d in devices))
        self.assertFalse(control.restore_result['verified'])

    def test_restore_failure_is_not_suppressed(self):
        control = telemetry.DualClocks([self.device(0, fail_reset=True), self.device(1)], True)
        control.set(2250)
        with self.assertRaises(RuntimeError): control.reset()
        self.assertTrue(control.dirty)

    def test_new_set_clears_old_restore_witness(self):
        control = telemetry.DualClocks([self.device(0), self.device(1)], True)
        control.restore_result['verified'] = True
        control.set(2250)
        self.assertFalse(control.restore_result['verified'])

    def test_claimed_clock_gate_without_recovery_evidence_rejected(self):
        gate = dict(schema=core.SCHEMA, status='passed', synthetic=False,
                    identity_fingerprint='boot', telemetry_config_sha256='cfg',
                    controlled_exception_injected=True,
                    cases=[dict(target_mhz=n, loaded_readback_verified=True) for n in (2250, 2527)])
        with self.assertRaisesRegex(ValueError, 'physical restoration'):
            telemetry.require_clock_gate(gate, 'boot', 'cfg')

    def test_bridge_readonly_does_not_probe_clock_grid(self):
        class Device:
            uuid = GPU_UUIDS[0]
            def probe(self): raise AssertionError('Read-only must not enumerate clocks')
            def telemetry(self): return {'ok': True}
        session = Session(Device(), 'token')
        result = session.handle(dict(token='token', uuid=GPU_UUIDS[0], seq=1, op='telemetry'))
        self.assertEqual(result, {'ok': True})


class ClientContracts(unittest.TestCase):
    def test_proxy_environment_is_process_local_and_exhaustive(self):
        with patch.dict(os.environ, {k: 'http://invalid-proxy' for k in client.PROXY_KEYS}):
            client.isolate_proxy()
            self.assertTrue(all(k not in os.environ for k in client.PROXY_KEYS))
            self.assertIn('10.255.255.254', os.environ['NO_PROXY'])

    def test_http_explicitly_bypasses_proxy(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self): return b'{}'
        with patch.object(client.urllib.request, 'build_opener') as build:
            build.return_value.open.return_value = Response()
            client.http(client.BASE_URL, '/v1/models')
            self.assertEqual(build.call_args.args[0].proxies, {})

    def test_startup_not_assumed_dummy_bf16(self):
        self.assertIsNone(client.startup_identity('service ready')['dtype'])
        state = client.startup_identity('Initializing a V1 LLM engine config: dtype=torch.float16, '
            'load_format=auto, tensor_parallel_size=2\nLoading model weights took 12s')
        self.assertEqual(state['dtype'], 'torch.float16')
        self.assertEqual(state['load_format'], 'auto')

    def test_model_created_timestamp_is_not_service_identity(self):
        def response(created):
            return [{'data': [dict(id=client.MODEL, root=client.MODEL, created=created)]}, {'paths': {}}]
        with patch.object(client, 'http', side_effect=response(1)+response(2)):
            self.assertEqual(client.inspect_service(client.BASE_URL)['fingerprint'],
                             client.inspect_service(client.BASE_URL)['fingerprint'])

    def test_profiler_config_no_capture_or_stack(self):
        config = client.profiler_config('/tmp/profile')
        self.assertEqual(config['max_iterations'], 3)
        self.assertFalse(config['capture_torch_profiler'])
        self.assertFalse(config['torch_profiler_with_stack'])
        self.assertNotIn('enforce_eager', config)

    def test_cache_miss_requires_counters_and_exclusivity(self):
        before = {'vllm:prefix_cache_hits_total{model="x"}': 100,
                  'vllm:request_success_total{model="x"}': 10}
        after = dict(before); after['vllm:request_success_total{model="x"}'] = 30
        self.assertEqual(core.cache_miss_status(before, after, 20)[0], 'verified_miss')
        self.assertEqual(core.cache_miss_status(before, after, 19)[0], 'unknown')
        after['vllm:prefix_cache_hits_total{model="x"}'] = 101
        self.assertEqual(core.cache_miss_status(before, after, 20)[0], 'cache_hit')
        self.assertEqual(core.cache_miss_status({}, {}, 20)[0], 'unknown')

    def test_parse_prometheus_labels(self):
        result = core.metric_snapshot('# comment\nvllm:prefix_cache_hits_total{model="x",engine="0"} 12\n')
        self.assertEqual(list(result.values()), [12.0])

    def test_redact_errors(self):
        with patch.dict(os.environ, SPARSEDVFS_BRIDGE_TOKEN='secret-value'):
            self.assertNotIn('secret-value', cli.safe_error(ValueError('secret-value')))


class AsyncClient(unittest.IsolatedAsyncioTestCase):
    async def test_token_usage_mismatch_rejected(self):
        # Construct a client without importing target-only vLLM/tokenizer dependencies.
        from types import SimpleNamespace as NS
        bench = object.__new__(client.BenchClient)
        bench.base, bench.model_id, bench.session = client.BASE_URL, 'model', None
        async def request(inp, session):
            session.usage = dict(prompt_tokens=127, completion_tokens=64)
            return NS(success=True, output_tokens=64, prompt_len=128, latency=2, ttft=.1)
        bench.endpoint = NS(RequestFuncInput=lambda **kw: NS(**kw),
                            StreamedResponseHandler=lambda: None,
                            ASYNC_REQUEST_FUNCS={'openai': request})
        with self.assertRaisesRegex(ValueError, 'Actual token'):
            await bench.run(dict(tokens=[1]*128, length=128, request_id='r', prompt_hash='p'))

    async def test_reuse_benchmark_timing_and_raw_usage(self):
        from types import SimpleNamespace as NS
        bench = object.__new__(client.BenchClient)
        bench.base, bench.model_id, bench.session = client.BASE_URL, 'model', None
        async def request(inp, session):
            self.assertTrue(inp.ignore_eos)
            self.assertFalse(inp.extra_body['add_special_tokens'])
            session.usage = dict(prompt_tokens=128, completion_tokens=64)
            return NS(success=True, output_tokens=64, prompt_len=128, latency=6.4, ttft=.1, generated_text='out')
        bench.endpoint = NS(RequestFuncInput=lambda **kw: NS(**kw),
                            StreamedResponseHandler=lambda: None,
                            ASYNC_REQUEST_FUNCS={'openai': request})
        result = await bench.run(dict(tokens=[1]*128, length=128, request_id='r', prompt_hash='p'))
        self.assertAlmostEqual(result['tpot_s'], .1)
        self.assertNotIn('generated_text', result)


class Orchestration(unittest.IsolatedAsyncioTestCase):
    def identity(self, directory):
        return dict(fingerprint='boot', identity_ready=True, base_url=client.BASE_URL,
                    models={'data': [{'id': client.MODEL}]},
                    profiler_routes={'/start_profile': True, '/stop_profile': True},
                    profile_overrides_supported=True,
                    launch={'profiler_config': {'torch_profiler_dir': str(directory)}})

    class FakeClient:
        source_sha256, tokenizer_fingerprint = 'client-source', 'tokens'
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def prepare(self, length, seed, request_id):
            return dict(length=length, tokens=[seed]*length, seed=seed,
                        request_id=request_id, prompt_hash=digest([length, seed]))

    async def test_failed_stream_preserves_partial_replies_and_raw_telemetry(self):
        class Sampler:
            def __init__(self, *args):
                self.frames, self.errors, self.thread, self.stopped = frames(1), [], None, False
            def start(self): self.thread = True
            def finish(self): self.stopped = True
        bench = NS(run=AsyncMock(side_effect=[{'ttft_s': .1, 'tpot_s': .03}, RuntimeError('stream failed')]))
        requests = [dict(request_id=str(i), prompt_hash=str(i), length=128) for i in range(2)]
        with tempfile.TemporaryDirectory() as tmp, patch.object(telemetry, 'Sampler', Sampler), \
                patch.object(cli, 'metrics', return_value={}), patch.object(cli, 'log_cursor', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'stream failed'):
                await cli.measured_window(bench, {}, [], requests, Path(tmp), 'failed')
            row = core.read(Path(tmp)/'failed-window.json')
            self.assertEqual(row['status'], 'invalid')
            self.assertEqual(len(row['replies']), 1)
            self.assertEqual(len(core.read(Path(tmp)/'failed-telemetry.json')['frames']), 11)

    async def test_native_swallowed_profiler_error_blocks_measurement(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(client, 'http', return_value={}):
            identity = self.identity(tmp); identity['launch_dir'] = tmp
            (Path(tmp)/'startup.log').write_text('WARNING Failed to stop profiler: device failure', encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError, 'Native profiler failure'):
                await cli.stop_profile(identity)

    async def test_profiler_start_failure_still_stops_and_saves_report(self):
        def http(base, route, *args, **kwargs):
            if route == '/start_profile': raise RuntimeError('start failed')
            return {}
        with tempfile.TemporaryDirectory() as tmp:
            args = NS(identity='identity', out=str(Path(tmp)/'observe'))
            with patch.object(cli, 'checked_identity', return_value=self.identity(tmp)), \
                    patch.object(client, 'BenchClient', return_value=self.FakeClient()), \
                    patch.object(cli, 'warmup', new=AsyncMock()), patch.object(cli, 'metrics', return_value={}), \
                    patch.object(client, 'http', side_effect=http) as calls:
                with self.assertRaisesRegex(RuntimeError, 'start failed'): await cli.observe(args)
            self.assertEqual([c.args[1] for c in calls.call_args_list].count('/stop_profile'), 2)
            self.assertEqual(core.read(Path(args.out)/'observation.json')['status'], 'failed')

    async def test_collect_invalid_window_restores_both_and_stays_failed(self):
        class Device:
            def __init__(self, uuid): self.uuid, self.calls = uuid, []
            def set(self, target): self.calls.append(('set', target))
            def reset(self): self.calls.append(('reset',))
            def close(self): self.calls.append(('close',))
        devices = [Device(u) for u in GPU_UUIDS]
        baseline = frames(1)
        for f in baseline:
            for s in f: s['raw']['clock_mhz'] = 2700
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            config = [dict(backend='bridge', uuid=u, port=50900+i) for i, u in enumerate(GPU_UUIDS)]
            core.write_new(root/'telemetry.json', config)
            core.write_new(root/'pilot.json', dict(status='ready', identity_fingerprint='boot',
                telemetry_config_sha256=digest(config), requests_per_window=2, sample_interval_s=.1,
                sources={u: 'counter' for u in GPU_UUIDS}, memory_reference={u: 13801 for u in GPU_UUIDS},
                tokenizer_fingerprint='tokens', client_source_sha256='client-source'))
            core.write_new(root/'gate.json', {})
            args = NS(identity='identity', telemetry=str(root/'telemetry.json'), pilot=str(root/'pilot.json'),
                      out=str(root/'collect'), clock_mode='fixed', apply=True, clock_check=str(root/'gate.json'),
                      model=None, split='train', seed=1)
            with patch.object(cli, 'checked_identity', return_value=self.identity(tmp)), \
                    patch.object(telemetry, 'open_devices', return_value=devices), \
                    patch.object(telemetry, 'require_clock_gate'), \
                    patch.object(client, 'BenchClient', return_value=self.FakeClient()), \
                    patch.object(cli, 'stop_profile', new=AsyncMock()), \
                    patch.object(cli, 'warmup', new=AsyncMock()), patch.object(cli, 'settle', new=AsyncMock(return_value=[])), \
                    patch.object(cli, 'load_frames', new=AsyncMock(return_value=baseline)), \
                    patch.object(cli, 'measured_window', new=AsyncMock(return_value=dict(status='invalid', quality_errors=['clock drift']))):
                with self.assertRaisesRegex(ValueError, 'Invalid window'): await cli.collect(args)
            saved = core.read(root/'collect/measurements.json')
            self.assertEqual(saved['status'], 'failed')
            self.assertTrue(saved['restoration']['verified'])
            self.assertEqual(len(saved['windows']), 1)
            self.assertTrue(all(d.calls == [('set', 2527), ('reset',), ('close',)] for d in devices))
            with self.assertRaises(ValueError): core.fit(saved)


if __name__ == '__main__': unittest.main()
