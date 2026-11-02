"""Synthetic mathematical/protocol tests. No model, GPU, jtop or real sysfs."""
import copy
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sparsedvfs.graph_partitioner import build_plan
from sparsedvfs.offline_modeler import predicted_table, select_operators
from sparsedvfs.runtime_policy import RuntimePlanner
from sparsedvfs.qwen_runner import Schedule
from sparsedvfs.hardware import PowerMonitor


def model_fixture(sparse=False, compute_bound=False):
    frequencies = [[100, g, m] for g in (100, 200) for m in (1000, 2000)]
    profile = {'synthetic': True, 'dvfs_applied': False, 'identity': {'test': True},
               'units': ['a'], 'reference': [100, 200, 2000],
               'candidates': [{'triplet': f, 'operators': [
                   {'phase': p, 'name': 'a', 'latency_ms': 1, 'power_w': 1} for p in ('prefill', 'decode')]}
                   for f in frequencies]}
    calibration = {'temperature_c': 40, 'opps': {}, 'operators': {}}
    for f in frequencies:
        key = ','.join(map(str, f))
        calibration['opps'][key] = {'voltage_v': 1, 'effective_frequency_hz': f[1] * 1e6 + f[2] * 1e3,
                                    'peak_flops': f[1] * 10000, 'bandwidth_Bps': f[2] * 1000,
                                    'k1': 0, 'k2': 1}
    for p in ('prefill', 'decode'):
        calibration['operators'][p + '/a'] = {'work_flop': 20000 if compute_bound else 10000,
                                              'data_byte': 1000 if compute_bound else 20000,
                                              's_comp': 0.8 if sparse else 0, 's_mem': 0,
                                              'effective_sparsity_verified': sparse, 'overhead_s': 0,
                                              'alpha_by_opp': {k: 1e-9 for k in calibration['opps']}}
    return profile, calibration


class PolicyTests(unittest.TestCase):
    def test_effective_sparse_policy_keeps_emc_high(self):
        p, c = model_fixture(sparse=True)
        ops, _ = select_operators(p, selector='paper', calibration=c, memory_policy='paper')
        self.assertEqual(ops[0]['triplet'], [100, 100, 2000])
        self.assertEqual(ops[0]['resource_class'], 'effective_sparse_memory_bound')

    def test_dense_compute_can_lower_emc_without_losing_compute_throughput(self):
        p, c = model_fixture(compute_bound=True)
        ops, _ = select_operators(p, selector='paper', calibration=c, memory_policy='paper')
        self.assertEqual(ops[0]['triplet'], [100, 200, 1000])
        self.assertEqual(ops[0]['resource_class'], 'compute_bound')

    def test_dense_memory_bound_is_not_called_sparse(self):
        p, c = model_fixture()
        ops, _ = select_operators(p, selector='paper', calibration=c, memory_policy='paper')
        self.assertEqual(ops[0]['resource_class'], 'dense_memory_bound')
        self.assertEqual(ops[0]['s_comp'], 0)

    def test_raw_zero_ratio_cannot_enable_skip_coefficients(self):
        p, c = model_fixture(sparse=True)
        c['operators']['prefill/a']['effective_sparsity_verified'] = False
        with self.assertRaisesRegex(ValueError, 'verified'):
            predicted_table(p, c)

    def test_memory_coordination_requires_calibrated_bottlenecks(self):
        p, _ = model_fixture()
        with self.assertRaisesRegex(ValueError, 'requires calibrated'):
            select_operators(p, memory_policy='paper')

    def test_thermal_feedback_changes_selected_frequency_and_partitions(self):
        p, c = model_fixture()
        # Retain two OPPs to make the physical-model tradeoff easy to inspect.
        p['candidates'] = [r for r in p['candidates'] if r['triplet'][2] == 2000]
        for r in p['candidates']:
            f = r['triplet']; key = ','.join(map(str, f)); fast = f[1] == 200
            c['opps'][key] = {'voltage_v': 1.2 if fast else 0.8, 'effective_frequency_hz': 1e6,
                              'peak_flops': 200000 if fast else 1000 / 0.0052,
                              'bandwidth_Bps': 1e9, 'k1': 1, 'k2': -20}
            for spec in c['operators'].values():
                spec.update(work_flop=1000, data_byte=1)
                spec['alpha_by_opp'][key] = (1 if fast else 1.1) / ((1.2 if fast else 0.8) ** 2 * 1e6)
        c.update(temperature_sensor='TEST_GPU', temperature_range_c=[20, 80], temperature_c=20)
        initial = build_plan(p, 100, selector='paper', calibration=c, allow_synthetic=True)
        with tempfile.TemporaryDirectory() as d:
            sensor = Path(d) / 'temp'; sensor.write_text('20000')
            hw = {'thermal': [{'name': 'TEST_GPU', 'path': str(sensor)}]}
            policy = RuntimePlanner(p, initial, hw, c, allow_synthetic=True)
            cold, cold_log = policy.replan('decode', 1)
            sensor.write_text('80000')
            hot, hot_log = policy.replan('decode', 2)
            self.assertEqual(cold['operator_choices'][0]['triplet'][1], 200)
            self.assertEqual(hot['operator_choices'][0]['triplet'][1], 100)
            self.assertEqual((cold_log['temperature_c'], hot_log['temperature_c']), (20, 80))
            sensor.write_text('81000')
            with self.assertRaisesRegex(ValueError, 'outside'):
                policy.replan('decode', 3)

    def test_runtime_profile_deployment_guard(self):
        p, _ = model_fixture()
        plan = build_plan(p, 100, allow_synthetic=True)
        with self.assertRaisesRegex(ValueError, 'synthetic'):
            RuntimePlanner(p, plan, {})


class ScheduleTests(unittest.TestCase):
    def plan(self):
        return {'switch_us': 100, 'blocks': [
            {'phase': 'prefill', 'block_id': i, 'operators': [name], 'triplet': [100, 1000, 10000],
             'estimated_ms': 3, 'amortized_strict_gt': True} for i, name in enumerate(('a', 'b'))]}

    def test_cuda_path_uses_ordered_scopes_without_host_gpu_fence(self):
        calls = []
        class Controller:
            def cuda_begin(self, stream, block, upcoming, lead, lookahead):
                calls.append(('begin', stream, block['block_id'], bool(upcoming and lookahead)))
                return block['block_id'] + 1
            def cuda_end(self, ticket): calls.append(('end', ticket))
            def cuda_drain(self): calls.append(('drain',))
        def forbidden(): raise AssertionError('Unexpected host CUDA synchronization')
        fake_stream = types.SimpleNamespace(cuda_stream=1234, synchronize=forbidden)
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=forbidden, current_stream=lambda: fake_stream))
        schedule = Schedule(self.plan(), Controller(), 'lookahead')
        with patch.dict(sys.modules, {'torch': fake_torch}):
            schedule.start_forward('prefill', 0)
            for name in ('a', 'b'):
                schedule.before('prefill', name); schedule.after('prefill', name)
            schedule.finalize()
        self.assertEqual(calls, [('begin', 1234, 0, True), ('end', 1), ('begin', 1234, 1, False), ('end', 2), ('drain',)])

    def test_host_timer_cannot_be_reported_as_v2_lookahead(self):
        with self.assertRaisesRegex(ValueError, 'CUDA stream adapter'):
            Schedule(self.plan(), object(), 'lookahead', transport='host')

    def test_runtime_repartition_callback_runs_at_forward_boundary(self):
        plan = self.plan()
        class Policy:
            def replan(self, phase, step): return plan, {'phase': phase, 'step': step}
        schedule = Schedule(plan, object(), 'serial', runtime_policy=Policy())
        schedule.start_forward('prefill', 0)
        self.assertEqual(schedule.repartitions, [{'phase': 'prefill', 'step': 0}])


class JtopTests(unittest.TestCase):
    def test_persistent_client_reads_only_total_and_converts_mw(self):
        calls = []
        class Client:
            interval = 0.01
            power = {'tot': {'power': 1234}, 'rail': {'GPU': {'power': 9999}}}
            def start(self): calls.append('start')
            def close(self): calls.append('close')
        monitor = PowerMonitor({'kind': 'jtop', 'scope': 'test'})
        with patch.dict(sys.modules, {'jtop': types.SimpleNamespace(jtop=Client)}):
            monitor.start(); monitor.stop()
        self.assertEqual(calls, ['start', 'close'])
        self.assertTrue(all(abs(p - 1.234) < 1e-9 for _, p in monitor.samples))

    def test_service_failure_never_generates_a_power_fallback(self):
        class Broken:
            def __init__(self): raise RuntimeError('service unavailable')
        monitor = PowerMonitor({'kind': 'jtop'})
        with patch.dict(sys.modules, {'jtop': types.SimpleNamespace(jtop=Broken)}):
            monitor.start(); monitor.stop()
        self.assertFalse(monitor.samples)
        self.assertGreater(monitor.errors, 0)

    def test_jtop_refresh_period_is_not_confused_with_host_poll_rate(self):
        monitor = PowerMonitor({'kind': 'jtop'})
        monitor.sensor_interval_s = 1
        monitor.samples = [(0, 10), (500000000, 10), (1000000000, 10)]
        self.assertFalse(monitor.summarize(100, 900000000)['valid'])


class TraceTests(unittest.TestCase):
    def test_trace_analysis_does_not_claim_physical_transition_hidden(self):
        spec = importlib.util.spec_from_file_location('trace_analyzer', ROOT / 'tools/analyze_cuda_trace.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'trace.csv'
            path.write_text('ticket,block_id,kind,host_ns,start_seen_ns,predicted_due_ns,gpu_elapsed_ms\n'
                            '1,0,lookahead_written_end_pending,100,10,90,-1\n'
                            '1,0,gpu_end_observed,120,10,90,0.1\n')
            result = module.analyze(path)
        self.assertEqual(result['requests_returned_while_block_end_still_pending'], 1)
        self.assertEqual(result['physical_clock_transition_hidden'], 'NOT_ESTABLISHED_BY_THIS_TRACE')


if __name__ == '__main__':
    unittest.main()
