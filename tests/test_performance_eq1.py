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
from rtx5090.performance import (AUTHOR_COMMIT, ClockGuard, DEFAULT_GPU, TorchRuntime, cases,
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
        row['snapshots']['before_wall'] = observe()
        spec, rate = row['case'], row['requested_mhz']
        peak = 1e14*rate/2527
        roofline = max(spec['work_flop']/peak, spec['data_byte']/1e12)
        row['wall_samples_s'].extend([roofline+20e-6]*repeat)
        row['snapshots']['after_wall'] = observe()
        row['snapshots']['before_cuda'] = observe()
        row['cuda_event_samples_s'].extend([roofline]*repeat)
        row['snapshots']['after_cuda'] = observe()


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
            runtime.measure(row, 5, 5, lambda: guard.observe(rate))
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
                def observe():
                    log.append('readback')
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

    def test_cli_fit_rejects_synthetic_without_creating_model(self):
        with tempfile.TemporaryDirectory() as folder:
            save(Path(folder)/'measurements.json', observations())
            with patch('sys.stderr', new_callable=io.StringIO):
                self.assertEqual(main(['fit', '--out', folder]), 1)
            self.assertFalse((Path(folder)/'model.json').exists())


if __name__ == '__main__': unittest.main()
