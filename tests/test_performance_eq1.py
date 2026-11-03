"""Eq1 units, isolation and cleanup tests; all timing fixtures are synthetic."""
import copy
from contextlib import nullcontext
import ctypes as C
import csv
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import patch

from rtx5090.backends import NVML
from rtx5090.performance import (AUTHOR_COMMIT, ClockChecks, ClockGuard, ClockReadbackError,
                                 DEFAULT_GPU, TorchRuntime, cases,
                                 checked_records, collect, digest, doctor, fit_model,
                                 load, main, parser, protocol, save, settings,
                                 source_hashes, validate_model)


def arguments(command, out, *extra):
    return parser().parse_args([command, '--out', str(out), '--ack-default-auto', *extra])


class FakeDevice:
    synthetic = False  # Emulates real-backend metadata, not real hardware.

    def __init__(self, *, mismatch=False, memory_change=False, reset_failure=False,
                 set_failure=False, close_failure=False, missing_memory=False):
        self.uuid = DEFAULT_GPU
        self.rate, self.calls, self.readings = 2527, [], 0
        self.mismatch, self.memory_change = mismatch, memory_change
        self.reset_failure, self.set_failure = reset_failure, set_failure
        self.close_failure, self.missing_memory = close_failure, missing_memory

    def probe(self):
        return dict(uuid=self.uuid, rates_mhz=[2250, 2527], synthetic=False)

    def set(self, rate):
        self.calls.append(('set', rate))
        self.rate = rate
        if self.set_failure: raise RuntimeError('partial write failed')

    def reset(self):
        self.calls.append('reset')
        if self.reset_failure: raise RuntimeError('driver reset failed')

    def close(self):
        self.calls.append('close')
        if self.close_failure: raise RuntimeError('close failed')

    def telemetry(self):
        self.readings += 1
        return dict(uuid=self.uuid, synthetic=False, timestamp_ns=self.readings*1000,
                    clock_mhz=self.rate+(100 if self.mismatch else 0), temperature_c=45,
                    memory_clock_mhz=None if self.missing_memory else
                    (14001 if self.memory_change and self.readings >= 3 else 13801))


class FakeRuntime:
    real_cuda = False  # Never allow these samples to masquerade as target execution.

    def __init__(self, uuid):
        self.uuid = uuid

    def environment(self):
        return dict(gpu_uuid=self.uuid, gpu_name='TEST ONLY',
                    memory_free_bytes=8*1024**3, sources_sha256=source_hashes())

    def measure(self, row, warmup, repeat, observe):
        observe('before_wall', 0, lambda: 0)
        spec, rate = row['case'], row['requested_mhz']
        peak = 1e14*rate/2527
        roofline = max(spec['work_flop']/peak, spec['data_byte']/1e12)
        row['wall_samples_s'].extend([roofline+20e-6]*repeat)
        observe('after_wall', 0, lambda: 0)
        observe('before_cuda', 0, lambda: 0)
        row['cuda_event_samples_s'].extend([roofline]*repeat)
        observe('after_cuda', 0, lambda: 0)
        row['active_phase'] = 'complete'


def observations():
    """Analytic, full-shape fixture; does not allocate tensors or write clocks."""
    a = arguments('collect', '.', '--apply')
    runtime, device = FakeRuntime(DEFAULT_GPU), FakeDevice()
    guard = ClockGuard(device, DEFAULT_GPU)
    records = []
    for rate in a.rates_mhz:
        device.rate = rate
        for spec in cases():
            row = dict(record_id=f'{rate}/{spec["case_id"]}', case=spec,
                       requested_mhz=rate, status='complete', wall_samples_s=[],
                       cuda_event_samples_s=[], snapshots={})
            runtime.measure(row, 5, 5, ClockChecks(guard, row))
            records.append(row)
    return dict(schema=1, kind='eq1_isolated_operator_measurements', status='collected',
                synthetic=True, real_cuda_executed=False, config=settings(a),
                protocol=protocol(a), environment=runtime.environment(), cases=cases(),
                records=records, memory_clock_mhz=13801,
                restoration=dict(reset_returned=True, physical_restore_verified=False))


class ModelTests(unittest.TestCase):
    def test_flop_and_byte_units_include_output(self):
        specs = {s['case_id']: s for s in cases()}
        self.assertEqual(specs['gemm_1024']['work_flop'], 2*1024**3)
        self.assertEqual(specs['gemm_1024']['data_byte'], 6*1024**2)
        self.assertEqual(specs['add_8mib']['work_flop'], 8*1024**2//2)
        self.assertEqual(specs['add_8mib']['data_byte'], 3*8*1024**2)
        self.assertEqual(protocol(arguments('collect', '.'))['author_commit'], AUTHOR_COMMIT)

    def test_capacity_uses_cuda_time_and_overhead_uses_wall_time(self):
        model = fit_model(observations(), allow_synthetic=True)
        params = model['parameters']
        self.assertAlmostEqual(params['p_peak_flops_by_gpu_mhz']['2527']/1e14, 1)
        self.assertAlmostEqual(params['b_mem_logical_Bps']/1e12, 1)
        self.assertAlmostEqual(params['t_overhead_s'], 20e-6, places=12)
        self.assertFalse(model['full_paper_reproduced'])

    def test_validation_data_cannot_change_fitted_parameters(self):
        data = observations()
        initial = fit_model(data, allow_synthetic=True)
        for row in data['records']:
            if row['case']['role'] == 'validation':
                row['wall_samples_s'] = [value*20 for value in row['wall_samples_s']]
                row['cuda_event_samples_s'] = [value*50 for value in row['cuda_event_samples_s']]
        changed = fit_model(data, allow_synthetic=True)
        self.assertEqual(initial['parameters'], changed['parameters'])
        self.assertNotEqual(initial['source_observations_sha256'], changed['source_observations_sha256'])

    def test_low_core_bandwidth_is_diagnostic_not_a_free_parameter(self):
        data = observations()
        row = next(r for r in data['records'] if r['record_id'] == '2250/add_256mib')
        row['cuda_event_samples_s'] = [value*2 for value in row['cuda_event_samples_s']]
        model = fit_model(data, allow_synthetic=True)
        self.assertAlmostEqual(model['parameters']['b_mem_logical_Bps']/1e12, 1)
        report = validate_model(data, model, allow_synthetic=True)
        diagnostic = next(r for r in report['bandwidth_diagnostic'] if r['gpu_mhz'] == 2250)
        self.assertAlmostEqual(diagnostic['relative_to_reference'], -.5)
        self.assertFalse(diagnostic['used_for_B_mem'])

    def test_validation_reports_exact_predictions_without_refitting(self):
        data = observations()
        model = fit_model(data, allow_synthetic=True)
        before = digest(model)
        with patch.dict('sys.modules', {'torch': None, 'vllm': None}):
            report = validate_model(data, model, allow_synthetic=True)
        self.assertEqual(digest(model), before)
        self.assertEqual(report['summary']['cases'], 10)
        self.assertLess(report['summary']['max_relative_error'], 1e-12)
        self.assertEqual(report['summary']['fraction_within_10_percent'], 1)
        self.assertEqual(report['status'], 'synthetic_test_only')

    def test_one_nonnegative_overhead_is_shared_by_all_predictions(self):
        data = observations()
        for row in data['records']:
            if row['case']['role'] == 'calibration':
                row['wall_samples_s'] = [value/2 for value in row['cuda_event_samples_s']]
        model = fit_model(data, allow_synthetic=True)
        self.assertEqual(model['parameters']['t_overhead_s'], 0)
        self.assertGreater(model['overhead_estimation']['negative_residual_count'], 0)
        report = validate_model(data, model, allow_synthetic=True)
        self.assertEqual({r['overhead_s'] for r in report['predictions']}, {0})

    def test_real_cli_rejects_synthetic_and_incomplete_evidence(self):
        with self.assertRaisesRegex(ValueError, 'Synthetic'):
            fit_model(observations())
        for mutate in (lambda d: d['records'].pop(),
                       lambda d: d['records'].append(copy.deepcopy(d['records'][0])),
                       lambda d: d['restoration'].update(reset_returned=False),
                       lambda d: d.update(status='failed'),
                       lambda d: d['records'][0]['wall_samples_s'].__setitem__(0, float('nan')),
                       lambda d: d['records'][0].update(clock_observations=[]),
                       lambda d: d['records'][0]['clock_checks']['after_wall'].update(status='failed'),
                       lambda d: d['cases'][0].update(data_byte=1),
                       lambda d: d['environment'].update(sources_sha256={})):
            data = observations()
            mutate(data)
            with self.assertRaises(ValueError): checked_records(data, allow_synthetic=True)

    def test_changed_evidence_or_per_frequency_overhead_is_rejected(self):
        data = observations()
        model = fit_model(data, allow_synthetic=True)
        model['parameters']['overhead_by_frequency'] = {'2250': 1e-6}
        with self.assertRaisesRegex(ValueError, 'shared bandwidth/overhead'):
            validate_model(data, model, allow_synthetic=True)
        model = fit_model(data, allow_synthetic=True)
        data['records'][-1]['wall_samples_s'][0] *= 2
        with self.assertRaisesRegex(ValueError, 'Frozen model'):
            validate_model(data, model, allow_synthetic=True)


class CollectionTests(unittest.TestCase):
    def run_collect(self, folder, device, *flags, runtime=FakeRuntime):
        a = arguments('collect', folder, '--apply', *flags)
        return collect(a, device_factory=lambda cfg: device,
                       runtime_factory=runtime, sleep=lambda _: None)

    def test_missing_apply_never_constructs_a_backend(self):
        with tempfile.TemporaryDirectory() as folder:
            factory = unittest.mock.Mock()
            with self.assertRaisesRegex(ValueError, '--apply'):
                collect(arguments('collect', folder), device_factory=factory)
            factory.assert_not_called()

    def test_doctor_only_reads_and_does_not_reset(self):
        with tempfile.TemporaryDirectory() as folder:
            device = FakeDevice()
            report = doctor(arguments('doctor', folder), device_factory=lambda cfg: device,
                            runtime_factory=FakeRuntime)
            self.assertEqual(device.calls, ['close'])
            self.assertEqual(report['status'], 'ready')
            self.assertFalse(report['clock_writes_performed'])

    def test_success_preserves_full_raw_data_and_resets(self):
        with tempfile.TemporaryDirectory() as folder:
            device = FakeDevice()
            data = self.run_collect(folder, device)
            self.assertEqual(device.calls, [('set', 2250), ('set', 2527), 'reset', 'close'])
            self.assertEqual(len(data['records']), 24)
            self.assertTrue(data['synthetic'])
            self.assertFalse(data['real_cuda_executed'])
            self.assertEqual(len(checked_records(data, allow_synthetic=True)), 24)
            self.assertEqual(load(Path(folder)/'measurements.json'), data)

    def test_clock_and_memory_failures_keep_samples_and_reset(self):
        for options in (dict(mismatch=True), dict(memory_change=True), dict(missing_memory=True)):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as folder:
                device = FakeDevice(**options)
                with self.assertRaises(RuntimeError): self.run_collect(folder, device)
                result = load(Path(folder)/'measurements.json')
                self.assertEqual(result['status'], 'failed')
                self.assertTrue(result['restoration']['reset_returned'])
                self.assertEqual(device.calls[-2:], ['reset', 'close'])
                if options.get('memory_change'):
                    self.assertEqual(len(result['records'][0]['wall_samples_s']), 5)

    def test_reported_second_rate_failure_keeps_phase_and_raw_readback(self):
        class SecondRateMismatch(FakeDevice):
            def telemetry(self):
                result = super().telemetry()
                if self.rate == 2527:
                    result.update(clock_mhz=2407, sm_clock_mhz=2407,
                                  clock_event_reasons=4, performance_state=0,
                                  power_mw=410000, power_limit_mw=600000)
                return result
        with tempfile.TemporaryDirectory() as folder:
            device = SecondRateMismatch()
            with self.assertRaisesRegex(RuntimeError, 'observed 2407'):
                self.run_collect(folder, device)
            result = load(Path(folder)/'measurements.json')
            self.assertEqual(sum(r['status'] == 'complete' for r in result['records']), 12)
            row = result['records'][-1]
            self.assertEqual(row['record_id'], '2527/gemm_8192')
            self.assertEqual(row['wall_samples_s'], [])
            self.assertEqual(row['cuda_event_samples_s'], [])
            reading = row['snapshots']['before_wall']
            self.assertEqual(reading['clock_mhz'], 2407)
            self.assertEqual(reading['clock_event_reasons'], 4)
            self.assertFalse(reading['passed'])
            self.assertEqual(reading['error_kind'], 'core_mismatch')
            self.assertEqual(result['failure_context']['phase'], 'before_wall')
            self.assertEqual(result['failure_context']['last_readback'], reading)
            self.assertTrue(result['restoration']['reset_returned'])
            self.assertFalse(result['restoration']['physical_restore_verified'])
            with self.assertRaisesRegex(ValueError, 'complete collection'):
                fit_model(result, allow_synthetic=True)

    def test_failed_write_still_attempts_reset(self):
        with tempfile.TemporaryDirectory() as folder:
            device = FakeDevice(set_failure=True)
            with self.assertRaisesRegex(RuntimeError, 'partial write'): self.run_collect(folder, device)
            self.assertEqual(device.calls, [('set', 2250), 'reset', 'close'])
            self.assertTrue(load(Path(folder)/'measurements.json')['restoration']['reset_returned'])

    def test_reset_failure_cannot_be_fit(self):
        with tempfile.TemporaryDirectory() as folder:
            device = FakeDevice(reset_failure=True)
            with self.assertRaisesRegex(RuntimeError, 'reset failed'): self.run_collect(folder, device)
            result = load(Path(folder)/'measurements.json')
            self.assertEqual(result['status'], 'failed')
            self.assertFalse(result['restoration']['reset_returned'])
            with self.assertRaisesRegex(ValueError, 'successful reset'):
                fit_model(result, allow_synthetic=True)

    def test_interrupt_also_restores_and_keeps_failure(self):
        class Interrupted(FakeRuntime):
            def measure(self, *args): raise KeyboardInterrupt('test interruption')
        with tempfile.TemporaryDirectory() as folder:
            device = FakeDevice()
            with self.assertRaises(KeyboardInterrupt): self.run_collect(folder, device, runtime=Interrupted)
            self.assertEqual(device.calls[-2:], ['reset', 'close'])
            self.assertEqual(load(Path(folder)/'measurements.json')['status'], 'failed')

    def test_identity_mismatch_never_changes_clocks(self):
        with tempfile.TemporaryDirectory() as folder:
            device = FakeDevice()
            device.uuid = 'GPU-e2f02606-bac6-52cd-f8a1-15a027db7da0'
            with self.assertRaisesRegex(ValueError, 'UUID'): self.run_collect(folder, device)
            self.assertEqual(device.calls, ['close'])

    def test_evidence_is_not_overwritten_and_token_is_redacted(self):
        class Bad(FakeRuntime):
            def measure(self, *args): raise RuntimeError(os.environ['SPARSEDVFS_BRIDGE_TOKEN'])
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {'SPARSEDVFS_BRIDGE_TOKEN': 'test-secret-sentinel'}):
            device = FakeDevice()
            with self.assertRaises(RuntimeError): self.run_collect(folder, device, runtime=Bad)
            path = Path(folder)/'measurements.json'
            original = path.read_bytes()
            self.assertNotIn(b'test-secret-sentinel', original)
            with self.assertRaises(FileExistsError): self.run_collect(folder, FakeDevice())
            self.assertEqual(path.read_bytes(), original)

    def test_doctor_close_failure_is_not_reported_ready(self):
        with tempfile.TemporaryDirectory() as folder:
            device = FakeDevice(close_failure=True)
            with self.assertRaisesRegex(RuntimeError, 'close failed'):
                doctor(arguments('doctor', folder), device_factory=lambda cfg: device,
                       runtime_factory=FakeRuntime)
            self.assertEqual(load(Path(folder)/'doctor.json')['status'], 'failed')


class FakeClock:
    def __init__(self): self.ns = 1_000_000_000
    def now(self): return self.ns
    def sleep(self, seconds): self.ns += round(seconds*1e9)


class ClockCheckTests(unittest.TestCase):
    def checker(self, rates, timeout=0, *, phase='before_wall', query_seconds=0):
        clock, device = FakeClock(), FakeDevice()
        pending = iter(rates)
        original = device.telemetry
        def telemetry():
            clock.sleep(query_seconds)
            return dict(original(), clock_mhz=next(pending))
        device.telemetry = telemetry
        guard = ClockGuard(device, DEFAULT_GPU, now_ns=clock.now)
        row = dict(requested_mhz=2527, snapshots={})
        check = ClockChecks(guard, row, timeout, sleep=clock.sleep, now_ns=clock.now)
        probes = []
        def probe():
            probes.append(clock.now())
            clock.sleep(.001)
            return clock.now()
        return lambda: check(phase, clock.now()-1_000_000, probe), row, device, probes

    def test_default_is_one_strict_read_without_extra_work(self):
        run, row, device, probes = self.checker([2407, 2527])
        with self.assertRaises(ClockReadbackError): run()
        self.assertEqual(device.readings, 1)
        self.assertEqual(probes, [])
        reading = row['clock_observations'][0]
        self.assertEqual(reading['sync_to_query_start_s'], .001)
        self.assertEqual(reading['query_roundtrip_s'], 0)
        self.assertEqual(row['clock_checks']['before_wall']['status'], 'failed')

    def test_opt_in_requires_three_consecutive_reads_and_keeps_initial_failures(self):
        run, row, device, probes = self.checker([2407, 2527, 2512, 2407, 2542, 2527, 2512], 5)
        result = run()
        self.assertEqual([r['consecutive_matching_reads'] for r in row['clock_observations']],
                         [0, 1, 2, 0, 1, 2, 3])
        self.assertEqual(result['clock_mhz'], 2512)  # Exactly -15 MHz is accepted.
        self.assertEqual(len(probes), 6)
        self.assertEqual(device.calls, [])  # Checks never write/rewrite clocks.
        self.assertEqual(row['clock_checks']['before_wall']['status'], 'passed')
        p = protocol(arguments('collect', '.', '--clock-stability-timeout-s', '5'))
        self.assertTrue(p['clock_stability']['author_flow_deviation'])
        self.assertEqual(p['warmup'], 5)
        self.assertEqual(p['repeat'], 5)
        self.assertEqual(p['settle_s'], .5)
        self.assertEqual(p['rates_mhz'], [2250, 2527])
        self.assertEqual(p['core_tolerance_mhz'], 15)

    def test_timeout_stops_probes_and_keeps_all_failed_reads(self):
        run, row, device, probes = self.checker([2407]*10, .25)
        with self.assertRaisesRegex(TimeoutError, 'stability timeout'): run()
        self.assertEqual(device.readings, 3)
        self.assertEqual(len(probes), 2)
        self.assertEqual(row['snapshots']['before_wall']['clock_mhz'], 2407)
        self.assertAlmostEqual(row['clock_checks']['before_wall']['elapsed_s'], .25)

    def test_late_matching_query_cannot_pass(self):
        run, row, device, probes = self.checker([2527], .25, query_seconds=.3)
        with self.assertRaisesRegex(TimeoutError, 'during query'): run()
        reading = row['snapshots']['before_wall']
        self.assertTrue(reading['passed'])  # Clock comparison passed, overall checkpoint failed.
        self.assertFalse(reading['within_deadline'])
        self.assertEqual(row['clock_checks']['before_wall']['status'], 'failed')
        self.assertEqual(probes, [])

    def test_no_query_is_started_after_a_probe_exceeds_the_budget(self):
        clock, device = FakeClock(), FakeDevice()
        row = dict(requested_mhz=2527, snapshots={})
        check = ClockChecks(ClockGuard(device, DEFAULT_GPU, now_ns=clock.now), row, .25,
                            sleep=clock.sleep, now_ns=clock.now)
        def slow_probe():
            clock.sleep(.3)
            return clock.now()
        with self.assertRaisesRegex(TimeoutError, 'after extra operation'):
            check('before_wall', clock.now(), slow_probe)
        self.assertEqual(device.readings, 1)

    def test_optional_timeout_also_restores_and_blocks_fit(self):
        clock, device = FakeClock(), FakeDevice(mismatch=True)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(TimeoutError):
                collect(arguments('collect', folder, '--apply', '--clock-stability-timeout-s', '.25'),
                        device_factory=lambda cfg: device, runtime_factory=FakeRuntime,
                        sleep=clock.sleep, now_ns=clock.now)
            result = load(Path(folder)/'measurements.json')
            self.assertTrue(result['restoration']['reset_returned'])
            self.assertEqual(result['failure_context']['phase'], 'before_wall')
            self.assertEqual(result['records'][0]['clock_checks']['before_wall']['attempts'], 3)
            with self.assertRaises(ValueError): fit_model(result, allow_synthetic=True)

    def test_postcheck_never_retries_even_when_stability_option_is_set(self):
        for phase in ('after_wall', 'after_cuda'):
            with self.subTest(phase=phase):
                run, row, device, probes = self.checker([2407, 2527, 2527, 2527], 5, phase=phase)
                with self.assertRaises(ClockReadbackError): run()
                self.assertEqual(device.readings, 1)
                self.assertEqual(probes, [])

    def test_memory_identity_and_query_errors_are_fatal_not_retried(self):
        for kind in ('memory', 'identity', 'query'):
            with self.subTest(kind=kind), patch.dict(os.environ, {'SPARSEDVFS_BRIDGE_TOKEN': 'sentinel-secret'}):
                clock, device = FakeClock(), FakeDevice()
                guard = ClockGuard(device, DEFAULT_GPU, memory_mhz=14001, now_ns=clock.now)
                if kind == 'identity': device.uuid = 'GPU-other'
                if kind == 'query':
                    def fail(): raise RuntimeError('sentinel-secret')
                    device.telemetry = fail
                row = dict(requested_mhz=2527, snapshots={})
                with self.assertRaises(ClockReadbackError):
                    ClockChecks(guard, row, 5, sleep=clock.sleep, now_ns=clock.now)(
                        'before_cuda', clock.now(), lambda: self.fail('Must not probe'))
                self.assertEqual(len(row['clock_observations']), 1)
                self.assertNotIn('sentinel-secret', json.dumps(row))

    def test_invalid_timeout_rejected_before_constructing_backend(self):
        for value in ('-1', 'nan', 'inf'):
            with self.subTest(value=value):
                factory = unittest.mock.Mock()
                with self.assertRaises(ValueError):
                    collect(arguments('collect', '.', '--apply', '--clock-stability-timeout-s', value),
                            device_factory=factory)
                factory.assert_not_called()


class BackendAndCliTests(unittest.TestCase):
    def test_offline_cli_writes_report_and_does_not_change_model(self):
        # Synthetic permission is injected only by this test, never exposed by CLI.
        with tempfile.TemporaryDirectory() as folder:
            save(Path(folder)/'measurements.json', observations())
            with patch('rtx5090.performance.fit_model', side_effect=lambda d: fit_model(d, allow_synthetic=True)), \
                 patch('rtx5090.performance.validate_model', side_effect=lambda d,m: validate_model(d,m,allow_synthetic=True)), \
                 patch.dict('sys.modules', {'torch': None, 'vllm': None}), \
                 patch('sys.stdout', new_callable=io.StringIO):
                self.assertEqual(main(['fit', '--out', folder]), 0)
                before = (Path(folder)/'model.json').read_bytes()
                self.assertEqual(main(['validate', '--out', folder]), 0)
            self.assertEqual(before, (Path(folder)/'model.json').read_bytes())
            self.assertEqual(load(Path(folder)/'validation.json')['status'], 'synthetic_test_only')
            with (Path(folder)/'predictions.csv').open(encoding='utf-8', newline='') as source:
                self.assertEqual(len(list(csv.DictReader(source))), 10)

    def test_real_timing_code_separates_passes_and_preallocates(self):
        for kind in ('gemm', 'add'):
            with self.subTest(kind=kind):
                log = []
                class Generator:
                    def __init__(self, **kwargs): pass
                    def manual_seed(self, seed): return self
                class Event:
                    def __init__(self, **kwargs): log.append('event_create')
                    def record(self): log.append('event_record')
                    def synchronize(self): log.append('event_sync')
                    def elapsed_time(self, end): return 2.0
                def allocate(*args, **kwargs):
                    log.append('allocate')
                    return object()
                def operate(*args, **kwargs): log.append('operation')
                runtime = TorchRuntime.__new__(TorchRuntime)
                runtime.torch = NS(inference_mode=nullcontext, Generator=Generator,
                                   bfloat16='bf16', randn=allocate, empty=allocate,
                                   empty_like=allocate, mm=operate, add=operate,
                                   cuda=NS(synchronize=lambda: log.append('device_sync'), Event=Event))
                row = dict(case=next(s for s in cases() if s['operator'] == kind),
                           snapshots={}, wall_samples_s=[], cuda_event_samples_s=[])
                def observe(phase, last_sync_ns, probe):
                    log.append('readback')
                    row['snapshots'][phase] = {'test_only': True}
                    self.assertIsInstance(last_sync_ns, int)
                    return {'test_only': True}
                with patch('rtx5090.performance.time.perf_counter', side_effect=range(10)):
                    runtime.measure(row, 5, 5, observe)
                self.assertEqual(row['wall_samples_s'], [1]*5)
                self.assertEqual(row['cuda_event_samples_s'], [.002]*5)
                self.assertEqual(log.count('operation'), 20)
                self.assertLess(max(i for i,v in enumerate(log) if v == 'allocate'), log.index('operation'))
                self.assertEqual(log[:log.index('event_create')].count('readback'), 2)

    def test_nvml_memory_readback_uses_memory_clock_domain(self):
        class Library:
            def nvmlInit_v2(self): return 0
            def nvmlShutdown(self): return 0
            def nvmlDeviceGetHandleByUUID(self, uuid, ptr): return 0
            def nvmlDeviceGetClockInfo(self, handle, domain, ptr):
                C.cast(ptr, C.POINTER(C.c_uint))[0] = 13801 if domain.value == 2 else 2250
                return 0
        with patch('rtx5090.backends.C.CDLL', return_value=Library()):
            device = NVML(DEFAULT_GPU)
            result = device.telemetry()
            device.close()
        self.assertEqual(result['memory_clock_mhz'], 13801)
        self.assertEqual(result['clock_mhz'], 2250)

    def test_nvml_optional_diagnostics_and_legacy_reason_fallback(self):
        class Library:
            def nvmlInit_v2(self): return 0
            def nvmlShutdown(self): return 0
            def nvmlDeviceGetHandleByUUID(self, uuid, ptr): return 0
            def nvmlDeviceGetClockInfo(self, handle, domain, ptr):
                C.cast(ptr, C.POINTER(C.c_uint))[0] = [2407, 2407, 14001][domain.value]
                return 0
            def nvmlDeviceGetCurrentClocksThrottleReasons(self, handle, ptr):
                C.cast(ptr, C.POINTER(C.c_ulonglong))[0] = 4
                return 0
            def nvmlDeviceGetPowerManagementLimit(self, handle, ptr):
                C.cast(ptr, C.POINTER(C.c_uint))[0] = 600000
                return 0
        with patch('rtx5090.backends.C.CDLL', return_value=Library()):
            device = NVML(DEFAULT_GPU)
            result = device.telemetry()
            device.close()
        self.assertEqual(result['sm_clock_mhz'], 2407)
        self.assertEqual(result['power_limit_mw'], 600000)
        self.assertEqual(result['clock_event_reasons'], 4)
        self.assertEqual(result['clock_event_reasons_api'], 'nvmlDeviceGetCurrentClocksThrottleReasons')
        self.assertIn('nvmlDeviceGetCurrentClocksEventReasons', result['errors'])
        self.assertIsNone(result['performance_state'])
        self.assertIn('performance_state', result['errors'])
        for value in result['field_query_times_ns'].values():
            self.assertGreaterEqual(value['end'], value['start'])

    def test_cli_fit_rejects_synthetic_without_creating_model(self):
        with tempfile.TemporaryDirectory() as folder:
            save(Path(folder)/'measurements.json', observations())
            with patch('sys.stderr', new_callable=io.StringIO):
                self.assertEqual(main(['fit', '--out', folder]), 1)
            self.assertFalse((Path(folder)/'model.json').exists())


if __name__ == '__main__': unittest.main()
