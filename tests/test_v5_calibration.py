"""Known-parameter regression, identifiability and measurement provenance tests."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch, Mock
from v5_demo import synthetic_fit_fixture
from sparsedvfs.calibration_fit import fit_calibration, least_squares, validate_fit
from sparsedvfs_vllm.calibration import collect_profile
from sparsedvfs_vllm.inside import measure_switches
from sparsedvfs_vllm.host import scenario
from sparsedvfs_vllm.workloads import describe


class FitTests(unittest.TestCase):
    def setUp(self):
        self.profile,self.obs,self.workloads,self.cert=synthetic_fit_fixture()

    def fit(self,threshold=.001):
        return fit_calibration(self.profile,self.obs,self.workloads,threshold)

    def test_known_thermal_activity_and_overhead_recovered(self):
        cal=self.fit()
        self.assertAlmostEqual(cal['device_thermal']['k1'],.01,places=12)
        self.assertAlmostEqual(cal['device_thermal']['k2'],.2,places=12)
        for i,spec in enumerate(cal['operators'].values()):
            self.assertAlmostEqual(spec['activity_factor'],(i+1)*1e-9,places=19)
            self.assertAlmostEqual(spec['overhead_s'],.0002,places=12)
        self.assertLess(cal['v5_validation']['max_relative_error'],1e-10)
        self.assertFalse(cal['v5_validation']['deployable'])

    def test_constant_temperature_is_rank_deficient(self):
        for row in self.obs['unit_records']:row['temperature_c']=45
        cal=self.fit()
        self.assertIn('Rank deficient',cal['v5_validation']['reasons'][0])
        self.assertNotIn('device_thermal',cal)

    def test_qr_handles_different_column_scales(self):
        result=least_squares([[1e9,1],[2e9,1],[3e9,1]],[3,5,7])
        self.assertAlmostEqual(result[0],2e-9,places=19)
        self.assertAlmostEqual(result[1],1,places=12)

    def test_qr_rejects_collinearity(self):
        with self.assertRaisesRegex(ValueError,'Rank deficient'):
            least_squares([[1,2],[2,4],[3,6]],[1,2,3])

    def test_unknown_opaque_kernel_traffic_remains_pending(self):
        key=next(k for k in self.workloads['operators'] if 'fused' in k)
        self.workloads['operators'][key]['data_byte']=None
        cal=self.fit()
        self.assertFalse(cal['v5_validation']['deployable'])
        self.assertTrue(any('unknown data_byte' in r for r in cal['v5_validation']['reasons']))

    def test_logical_bytes_cannot_masquerade_as_dram_measurement(self):
        next(iter(self.workloads['operators'].values()))['data_semantics']='tensor_numel_times_dtype'
        self.assertTrue(any('byte semantics' in r for r in self.fit()['v5_validation']['reasons']))

    def test_request_average_power_is_not_unit_power(self):
        self.obs['unit_records'][0]['power_attribution']='whole_request_average_proxy'
        with self.assertRaisesRegex(ValueError,'Request-average'):
            self.fit()

    def test_power_sampling_window_cannot_be_shorter_than_sensor_refresh(self):
        self.obs['unit_records'][0]['window_duration_s']=.001
        with self.assertRaisesRegex(ValueError,'two documented sensor intervals'):
            self.fit()

    def test_wrong_power_units_and_mixed_rails_are_rejected(self):
        self.obs['units']['power']='mW'
        with self.assertRaisesRegex(ValueError,'units'):
            self.fit()
        self.obs['units']['power']='W'
        self.obs['unit_records'][0]['power_scope']='carrier_VIN'
        with self.assertRaisesRegex(ValueError,'scope mismatch'):
            self.fit()

    def test_missing_voltage_is_not_filled_from_another_board(self):
        next(iter(self.obs['opps'].values()))['voltage_v']=None
        self.assertTrue(any('voltage evidence' in r for r in self.fit()['v5_validation']['reasons']))

    def test_explicit_error_threshold_required(self):
        cal=self.fit(None)
        self.assertTrue(any('explicit' in r for r in cal['v5_validation']['reasons']))

    def test_holdout_error_detected_without_leaking_into_coefficients(self):
        index={}
        for row in self.obs['unit_records']:
            unit=row['unit'];i=index.get(unit,0);index[unit]=i+1
            if i%4==3:row['power_w']*=1.5
        cal=self.fit()
        self.assertAlmostEqual(cal['device_thermal']['k1'],.01,places=12)
        self.assertGreater(cal['v5_validation']['max_relative_error'],.3)
        self.assertTrue(any('Held-out' in r for r in cal['v5_validation']['reasons']))

    def test_holdout_validates_the_opp_projection_actually_used_at_runtime(self):
        for opp in self.obs['opps'].values():opp['voltage_v']*=.5
        cal=self.fit()
        self.assertAlmostEqual(cal['device_thermal']['k1'],.01,places=12)
        self.assertGreater(cal['v5_validation']['max_relative_error'],.5)
        self.assertTrue(any('Held-out' in r for r in cal['v5_validation']['reasons']))

    def test_collect_does_not_generate_per_unit_power(self):
        profile=copy.deepcopy(self.profile)
        profile.update(synthetic=False,dvfs_applied=True)
        for candidate in profile['candidates']:
            candidate.update(measurement={'power':{'average_power_w':5}},readback={},microbench={})
        value=collect_profile(profile,{})
        self.assertEqual(value['unit_records'],[])
        self.assertEqual(value['request_records'][0]['power_attribution'],
                         'whole_request_average_proxy_not_unit_power')

    def test_matmul_flops_do_not_auto_fill_dram_bytes(self):
        entry=dict(overload='aten.mm.default',metadata=dict(args=[
            dict(shape=[512,5120],dtype='torch.bfloat16'),
            dict(shape=[5120,17408],dtype='torch.bfloat16')]))
        result=describe(entry)
        self.assertEqual(result['work_flop'],2*512*5120*17408)
        self.assertGreater(result['logical_tensor_bytes'],0)
        self.assertIsNone(result['data_byte'])

    def test_fused_attention_does_not_inherit_dense_mm_work(self):
        result=describe(dict(overload='_C.gated_delta_net.default',metadata={'shape':[512,5120]}))
        self.assertIsNone(result['work_flop'])
        self.assertIsNone(result['data_byte'])


class SwitchTests(unittest.TestCase):
    def test_directed_pairs_repeated_ten_times_without_noops(self):
        torch=NS(ones=lambda *a,**k:object(),bfloat16='bf16')
        grid={'triplets':[[1,2,3],[4,5,6],[7,8,9]]}
        with tempfile.TemporaryDirectory() as temp,patch.dict('sys.modules',{'torch':torch}), \
             patch('sparsedvfs_vllm.inside.wait_frequencies',return_value={'observed_us':1}), \
             patch('sparsedvfs.hardware.observation',return_value={'physical_pll_settled':False}):
            value=measure_switches(Mock(),{},grid,Path(temp))
        self.assertEqual(len(value['records']),60)
        self.assertTrue(all(r['from_triplet']!=r['triplet'] for r in value['records']))
        self.assertEqual({r['repeat'] for r in value['records']},set(range(10)))
        self.assertFalse(value['physical_pll_settle_verified'])

    def test_granularity_result_directories_do_not_collide(self):
        args=NS(results=Path('runs'),input_tokens=512,output_tokens=128,granularity='module')
        module=scenario(args);args.granularity='operator'
        self.assertEqual(scenario(args),module/'operator')
