import copy
import unittest

from rtx5090.equation_predict import make_template, evaluate, demo_parameters
from rtx5090.observe_core import GPU_UUIDS
from rtx5090.eager_model import predict_latency, predict_power


def workloads():
    return dict(status='workload_accounting_only', identity_fingerprint='test', workloads=[
        dict(rank=r, uuid=GPU_UUIDS[r], phase=p, module=m, signature_id=f'{r}:{p}:{m}',
             M=1, N=5, K=10, work_flop=100, logical_data_bytes=130,
             effective_s_comp=0, effective_s_mem=0)
        for r in (0, 1) for p in ('prefill', 'decode') for m in ('gate_up_proj', 'down_proj')])


class EquationTests(unittest.TestCase):
    def test_hand_calculation(self):
        r = evaluate(demo_parameters(), allow_illustrative=True)
        c = r['cases'][0]
        self.assertEqual((c['compute_s'], c['memory_s'], c['latency_s']), (10, 5, 11))
        self.assertEqual((c['dynamic_power_w'], c['leakage_power_w'], c['power_w']), (1, 5, 6))
        self.assertEqual(c['model_energy_j'], 66)
        self.assertFalse(r['block_selection_authorized'])

    def test_explicit_demo(self):
        with self.assertRaisesRegex(ValueError, 'allow-illustrative'):
            evaluate(demo_parameters())

    def test_missing_parameters_not_defaults(self):
        w = workloads()
        p = make_template(w)
        r = evaluate(p, w)
        self.assertEqual(r['status'], 'parameters_missing')
        self.assertEqual(len(r['cases']), 8)
        self.assertIsNone(p['cases'][0]['eq1']['data_byte'])
        self.assertIsNone(r['cases'][0]['latency_s'])

    def test_eq1_without_voltage(self):
        w = workloads()
        p = make_template(w)
        for c in p['cases']:
            c['eq1'] = demo_parameters()['cases'][0]['eq1']
        r = evaluate(p, w)
        self.assertEqual(r['cases'][0]['latency_s'], 11)
        self.assertIsNone(r['cases'][0]['power_w'])
        self.assertIsNone(r['cases'][0]['model_energy_j'])

    def test_eq2_without_capacity(self):
        w = workloads()
        p = make_template(w)
        for c in p['cases']:
            c['eq2'] = demo_parameters()['cases'][0]['eq2']
        r = evaluate(p, w)
        self.assertEqual(r['cases'][0]['power_w'], 6)
        self.assertIsNone(r['cases'][0]['latency_s'])

    def test_wrong_identity_hash_case_or_units(self):
        w = workloads()
        base = make_template(w)
        variants = []
        for field in ('identity_fingerprint', 'workloads_sha256'):
            p = copy.deepcopy(base); p[field] = 'changed'; variants.append(p)
        p = copy.deepcopy(base); p['cases'][0]['workload']['work_flop'] = 200; variants.append(p)
        p = copy.deepcopy(base); p['units']['frequency'] = 'MHz'; variants.append(p)
        p = copy.deepcopy(base); p['cases'][0] = p['cases'][1]; variants.append(p)
        for p in variants:
            with self.subTest(p=p), self.assertRaises(ValueError): evaluate(p, w)

    def test_bad_numbers_even_when_incomplete(self):
        for value in (float('nan'), float('inf'), -1, True, 0):
            w = workloads(); p = make_template(w)
            p['cases'][0]['eq1']['peak_flops'] = value
            with self.subTest(value=value), self.assertRaises(ValueError): evaluate(p, w)

    def test_negative_leakage_rejected(self):
        p = demo_parameters(); p['cases'][0]['eq2']['k2'] = -4.5
        with self.assertRaisesRegex(ValueError, 'Negative leakage'):
            evaluate(p, allow_illustrative=True)

    def test_both_ranks_and_work_convention(self):
        for alteration in ('missing', 'uuid', 'flops'):
            w = workloads()
            if alteration == 'missing': w['workloads'].pop()
            elif alteration == 'uuid': w['workloads'][0]['uuid'] = 'wrong'
            else: w['workloads'][0]['work_flop'] = 50
            with self.assertRaises(ValueError): make_template(w)

    def test_existing_model_separate_equations(self):
        m = dict(workloads={'0:prefill/x': dict(work_flop=100, data_byte=100)},
                 operators={'0:prefill/x': dict(overhead_s=1)},
                 capacities={'0': dict(peak_flops={'1000': 10}, bandwidth_Bps=20)})
        self.assertEqual(predict_latency(m, 0, 'prefill', 'x', 1000), 11)
        p = dict(operators={'0:prefill/x': dict(alpha=1e-9, voltage_by_mhz={'1000': 1})},
                 thermal={'0': dict(k1=.1, k2=1, temperature_range_c=[30, 50])})
        self.assertEqual(predict_power(p, 0, 'prefill', 'x', 1000, 40), 6)
        with self.assertRaises(ValueError): predict_power(p, 0, 'prefill', 'x', 1000, 60)


if __name__ == '__main__':
    unittest.main()
