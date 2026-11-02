import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sparsedvfs.offline_modeler import timeline_seconds, thermal_power_w, select_operators
from sparsedvfs.graph_partitioner import partition, build_plan
from sparsedvfs.hardware import PowerMonitor, power_w, probe, make_grid, snapshot, restore
spec = importlib.util.spec_from_file_location('demo_cli', ROOT / 'demo.py')
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def fixture():
    names = ['embed', 'layer.0', 'layer.1', 'head']
    candidates = []
    for f in ([100, 1000, 10000], [200, 1000, 10000], [100, 2000, 10000], [200, 2000, 10000]):
        rows = []
        for phase in ('prefill', 'decode'):
            for i, name in enumerate(names):
                rows.append({'phase': phase, 'name': name, 'latency_ms': 3 + i,
                             'power_w': 10 if f[0] == 100 else 20,
                             'sparsity_exact_zero': 0.0})
        candidates.append({'triplet': f, 'operators': rows})
    return {'synthetic': True, 'dvfs_applied': False, 'identity': {'fixture': True},
            'units': names, 'reference': [200, 2000, 10000], 'candidates': candidates}


def fake_hardware(root):
    root = Path(root)
    def put(path, value):
        p = root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(str(value))
    for i in (0, 4):
        p = f'devices/system/cpu/cpufreq/policy{i}/'
        for name, value in {'scaling_min_freq': 400000, 'scaling_max_freq': 1000000,
                            'scaling_available_frequencies': '400000 800000 1000000',
                            'scaling_cur_freq': 800000, 'scaling_governor': 'schedutil'}.items():
            put(p + name, value)
    for name, value in {'min_freq': 200000000, 'max_freq': 1000000000,
                        'available_frequencies': '200000000 500000000 1000000000',
                        'cur_freq': 500000000, 'governor': 'nvhost_podgov'}.items():
        put('class/devfreq/gpu-gpc-0/' + name, value)
    put('class/devfreq/gpu-nvd-0/decoy', 1)
    for name, value in {'rate': 1000000000, 'mrq_rate_locked': 0, 'min_rate': 1000000000,
                        'max_rate': 2000000000}.items():
        put('kernel/debug/bpmp/debug/clk/emc/' + name, value)
    put('kernel/debug/bpmp/debug/bwmgr/bwmgr_halt', 0)
    put('class/hwmon/hwmon9/name', 'ina238')
    put('class/hwmon/hwmon9/power1_input', 25000000)
    put('class/hwmon/hwmon9/in1_label', 'VIN')
    return probe(root, emc_rates=[1000000000, 2000000000])


class ModelTests(unittest.TestCase):
    def test_equation_1_bottleneck(self):
        self.assertEqual(timeline_seconds(100, 100, 0, 0, 10, 20, 1), 11)
        self.assertEqual(timeline_seconds(100, 100, 0.8, 0, 10, 20, 1), 6)
        self.assertAlmostEqual(timeline_seconds(100, 100, 0.8, 0.8, 10, 20, 1), 3)

    def test_invalid_physical_inputs(self):
        with self.assertRaises(ValueError):
            timeline_seconds(1, 1, 1.1, 0, 1, 1, 0)
        with self.assertRaises(ValueError):
            thermal_power_w(1, 0, 1, 1, 1, 50)

    def test_equation_2_temperature(self):
        self.assertAlmostEqual(thermal_power_w(1e-9, 1, 1e9, 0.1, 1, 40), 6)

    def test_synthetic_plan_gate(self):
        with self.assertRaisesRegex(ValueError, 'synthetic'):
            build_plan(fixture(), 100)
        plan = build_plan(fixture(), 100, allow_synthetic=True)
        self.assertTrue(plan['synthetic'])

    def test_invalid_power_is_never_fabricated(self):
        data = fixture()
        data['candidates'][0]['operators'][0]['power_w'] = None
        with self.assertRaisesRegex(ValueError, 'valid measured power'):
            select_operators(data)

    def test_incomplete_coverage(self):
        data = fixture()
        data['candidates'][0]['operators'].pop()
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            select_operators(data)

    def test_greedy_or_and_component_max(self):
        ops = [{'phase': 'prefill', 'name': 'a', 'triplet': [1, 2, 3]},
               {'phase': 'prefill', 'name': 'b', 'triplet': [2, 1, 4]},
               {'phase': 'prefill', 'name': 'c', 'triplet': [2, 2, 4]}]
        def duration(names, phase, freq):
            return len(names) * 0.4
        blocks = partition(ops, duration, switch_us=100, factor=5, tolerance=0)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]['triplet'], [2, 2, 4])

    def test_equality_and_short_terminal_block_visible(self):
        ops = [{'phase': 'decode', 'name': 'a', 'triplet': [1, 1, 1]},
               {'phase': 'decode', 'name': 'b', 'triplet': [2, 2, 2]}]
        blocks = partition(ops, lambda names, phase, freq: 0.5, 100, factor=5, tolerance=0)
        self.assertEqual(len(blocks), 2)
        self.assertFalse(blocks[-1]['amortized_strict_gt'])

    def test_phase_boundary_never_fused(self):
        ops = [{'phase': p, 'name': p, 'triplet': [1, 1, 1]} for p in ('prefill', 'decode')]
        self.assertEqual(len(partition(ops, lambda *a: 0.001, 100)), 2)


class HardwareTests(unittest.TestCase):
    def test_probe_is_read_only_and_chooses_gpc(self):
        with tempfile.TemporaryDirectory() as d:
            hw = fake_hardware(d)
            before = {str(p): p.read_bytes() for p in Path(d).rglob('*') if p.is_file()}
            same = probe(d, emc_rates=[1000000000, 2000000000])
            after = {str(p): p.read_bytes() for p in Path(d).rglob('*') if p.is_file()}
            self.assertEqual(before, after)
            self.assertTrue(same['gpu']['path'].endswith('gpu-gpc-0'))
            self.assertEqual(len(hw['cpu']), 2)
            self.assertEqual(hw['power']['scope'], 'system_module_and_carrier')

    def test_grid_is_cartesian_and_reference_present(self):
        with tempfile.TemporaryDirectory() as d:
            hw = fake_hardware(d)
            grid = make_grid(hw, 2)
            self.assertEqual(len(grid['triplets']), 8)
            self.assertIn(grid['reference'], grid['triplets'])
            cli.validate_grid(hw, grid)

    def test_power_units_and_missing_power(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'p'; p.write_text('25000000')
            self.assertEqual(power_w({'kind': 'direct', 'path': str(p), 'scale_to_w': 1e-6}), 25)
            v, i = Path(d) / 'v', Path(d) / 'i'
            v.write_text('12000'); i.write_text('2000')
            self.assertEqual(power_w({'kind': 'vi', 'voltage_path': str(v), 'current_path': str(i), 'scale_to_w': 1e-6}), 24)
            self.assertIsNone(power_w(None))

    def test_energy_integration_clips_window(self):
        monitor = PowerMonitor({'scope': 'test'})
        monitor.samples = [(0, 10), (1000000000, 20), (2000000000, 30)]
        result = monitor.summarize(500000000, 1500000000)
        self.assertTrue(result['valid'])
        self.assertAlmostEqual(result['energy_j'], 20)
        monitor.errors = 1
        self.assertIsNone(monitor.summarize(500000000, 1500000000)['energy_j'])

    def test_energy_requires_coverage_and_interior_sample(self):
        monitor = PowerMonitor(None)
        monitor.samples = [(0, 10), (1000, 10)]
        self.assertFalse(monitor.summarize(100, 900)['valid'])

    def test_restore_widens_before_narrowing(self):
        writes = []
        class Sink:
            def __init__(self, path): self.path = path
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def write(self, value): writes.append((self.path, int(value)))
        saved = {'domains': [{'min_path': 'min', 'max_path': 'max', 'min': 300, 'max': 500}], 'bpmp': None}
        with patch('sparsedvfs.hardware.integer', return_value=200), patch('builtins.open', side_effect=lambda p, m: Sink(p)):
            restore(saved)
        self.assertEqual(writes, [('max', 500), ('min', 300)])


class ComparisonTests(unittest.TestCase):
    def documents(self):
        run = {'elapsed_ms': 100, 'ttft_ms': 50, 'tpot_ms': 10,
               'generated_token_ids': [1, 2], 'power': {'energy_j': 2.0}}
        a = {'identity': {'id': 'same'}, 'mode': 'fixed', 'dvfs_applied': True,
             'power_source': {'scope': 'same'}, 'runs': [run]}
        b = copy.deepcopy(a)
        b['mode'] = 'serial'; b['runs'][0]['elapsed_ms'] = 110; b['runs'][0]['power']['energy_j'] = 1.5
        return a, b

    def test_actual_energy_and_latency(self):
        report = cli.compare_documents(*self.documents())
        self.assertAlmostEqual(report['energy_saving_pct'], 25)
        self.assertAlmostEqual(report['latency_change_pct'], 10)

    def test_different_tokens_invalidates_energy_claim(self):
        a, b = self.documents(); b['runs'][0]['generated_token_ids'] = [1, 3]
        report = cli.compare_documents(a, b)
        self.assertFalse(report['valid_energy_comparison'])
        self.assertIsNone(report['energy_saving_pct'])

    def test_dry_run_cannot_claim_hardware_gain(self):
        a, b = self.documents(); b['dvfs_applied'] = False
        with self.assertRaisesRegex(ValueError, 'did not apply'):
            cli.compare_documents(a, b)


if __name__ == '__main__':
    unittest.main()
