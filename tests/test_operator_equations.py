import copy
import unittest
from rtx5090.operator_equations import audit,predict,select_no_slowdown,partition_input
from rtx5090.partition_dryrun import partition
from rtx5090.observe_core import GPU_UUIDS,digest
from rtx5090.operator_parameter_gaps import build,linear_work


def fixture():
    b=dict(schema='operator-equations-v1',scope='synthetic_test_only',identity_fingerprint='fixture',
        selection_policy='legacy_no_slowdown',
        current_identity_verified=True,rank_join_verified=True,rank_join_evidence_sha256='fixture',
        units_contract=dict(time='s',work='FLOP',data='byte',frequency='Hz',voltage='V',power='W'),
        operators=[],families={},thermal={},states={},switch_s=.001,switch_bound_verified=False,
        graph=dict(edges=[],timing_model='verified_serial',dependency_evidence_sha256='fixture',serial_nonoverlap_verified=True))
    for r in (0,1):
        b['thermal'][str(r)]=dict(k1=1,k2=2,identifiability_passed=True,evidence_sha256='fixture',temperature_range_c=[30,60],voltage_range_v=[.8,1.1])
        b['states'][str(r)]={str(f):dict(effective_frequency_hz=f*1e6-30e6,voltage_v=.9 if f==2250 else 1,
            temperature_c=40,known_before_prediction=True,evidence_sha256='fixture') for f in (2250,2527)}
        family=dict(rank=r,signature_ids=['sig'],same_path_verified=True,path_evidence_sha256='fixture',
            independent_capacity_calibration=True,capacity_evidence_sha256='fixture',alpha=1e-8,overhead_s=.000001,
            power_scope='same_path_repeated_unit_board_windows',paired_window_verified=True,power_evidence_sha256='fixture',
            capacities={str(f):dict(peak_flops=1e6 if f==2250 else 2e6,bandwidth_Bps=1e7,
                        frequency_range_hz=[f*1e6-100e6,f*1e6]) for f in (2250,2527)})
        family['validation']=dict(evidence_sha256='fixture',independent=True,training_ids=['train'],holdout_ids=['held'],
            scores={str(f):{m:dict(wape=.01,p90_relative_error=.02) for m in ('latency','energy')} for f in (2250,2527)},
            parameters_sha256=digest(dict(alpha=family['alpha'],overhead_s=family['overhead_s'],
                capacities=family['capacities'],thermal=b['thermal'][str(r)])))
        b['families'][str(r)]=family
    b['operators']=[dict(id='op',ranks=[dict(rank=r,uuid=GPU_UUIDS[r],signature_id='sig',family_id=str(r),
        work_flop=1000,data_byte=100,workload_provenance='fixture',data_semantics='logical_bytes',s_comp=0,s_mem=0) for r in (0,1)])]
    return b


class EquationTests(unittest.TestCase):
    def test_formula_selection_and_energy_free_partition(self):
        b=fixture();original=copy.deepcopy(b);r=predict(b)
        self.assertFalse(r['blockers']);self.assertEqual(b,original)
        op=r['operators']['op'];self.assertEqual(op['selected_mhz'],2527)
        self.assertAlmostEqual(op['predictions']['2250']['ranks']['0']['latency_s'],.001001)
        self.assertAlmostEqual(op['predictions']['2250']['ranks']['0']['power_w'],1e-8*.9**2*2220e6+42*.9)
        data=partition_input(b,r)
        self.assertNotIn('pair_energy_j',data['candidates']['op']['2250'])
        self.assertEqual(partition(data)['status'],'offline_partition_complete')
        self.assertFalse(partition(data)['deployable'])
        for row in b['operators'][0]['ranks']:row['data_byte']=100000
        self.assertEqual(predict(b)['operators']['op']['selected_mhz'],2250)

    def test_no_slowdown_not_five_percent_and_deterministic_ties(self):
        self.assertEqual(select_no_slowdown({2250:dict(latency_s=1.000000001,energy_j=.1),2527:dict(latency_s=1,energy_j=10)}),2527)
        self.assertEqual(select_no_slowdown({f:dict(latency_s=1,energy_j=1) for f in (2250,2527)}),2250)

    def test_parameter_leakage_identity_frequency_and_scope_guards(self):
        mutations=[lambda b:b['families']['0'].update(alpha=2),
            lambda b:b['families']['0']['validation'].update(holdout_ids=['train']),
            lambda b:b.update(current_identity_verified=False),
            lambda b:b['states']['0']['2250'].update(known_before_prediction=False),
            lambda b:b['states']['0']['2250'].update(effective_frequency_hz=1e9),
            lambda b:b['states']['0']['2250'].update(voltage_v=2),
            lambda b:b['families']['0'].update(power_scope='whole_request_allocation'),
            lambda b:b['operators'][0]['ranks'][0].update(s_comp=.5)]
        for change in mutations:
            b=fixture();change(b)
            self.assertEqual(predict(b)['status'],'blocked_parameter_evidence')

    def test_empty_or_wrong_uuid_rejected(self):
        b=fixture();b['operators']=[]
        with self.assertRaises(ValueError):predict(b)
        b=fixture();b['operators'][0]['ranks'][0]['uuid']='wrong'
        with self.assertRaises(ValueError):predict(b)


def inventory(rank):
    tensor=lambda shape:dict(tensor=True,dtype='torch.bfloat16',shape=shape,stride=[shape[-1],1],device=f'cuda:{rank}',layout='torch.strided')
    sig=dict(rank=rank,phase='decode',overload='aten.linear.default',
             inputs=dict(args=[tensor([1,5120]),tensor([17408,5120])]),outputs=tensor([1,17408]),kernel_names=['gemv'])
    nodes=[dict(id=f'{rank}:{i}',signature_id='sig',context='decode.0',operator='aten.linear.default',
        parent_label=f'model.layers.{i}.mlp.gate_up_proj',layer=i,kernels=['gemv']) for i in (0,1)]
    return dict(rank=rank,identity=dict(gpu_uuid=GPU_UUIDS[rank]),nodes=nodes,signatures=dict(sig=sig))


class GapTests(unittest.TestCase):
    def test_work_units_dedup_no_auto_certification(self):
        a,b=inventory(0),inventory(1)
        self.assertEqual(linear_work(a['signatures']['sig'])['work_flop'],178257920)
        bundle,report=build([a,b])
        self.assertEqual(report['pair_occurrences'],2)
        self.assertEqual(report['candidate_families'],2)
        self.assertEqual(report['rank_contract_mismatches'],[])
        self.assertEqual(report['tasks'][0]['representative_layer'],1)
        self.assertIsNone(bundle['graph']['edges'])
        self.assertTrue(audit(bundle))
        self.assertEqual(predict(bundle)['operators'],{})

    def test_unknown_work_and_rank_mismatch_preserved(self):
        a,b=inventory(0),inventory(1)
        a['signatures']['sig']['overload']='custom.attention'
        bundle,report=build([a,b])
        self.assertEqual(len(report['rank_contract_mismatches']),2)
        self.assertIsNone(bundle['operators'][0]['ranks'][0]['work_flop'])

class StaticWorkTests(unittest.TestCase):
    def test_broadcast_float_work_and_rejections(self):
        from rtx5090.operator_parameter_gaps import elementwise_work
        def tensor(shape,dtype='torch.float32'):
            return dict(tensor=True,shape=shape,dtype=dtype,layout='torch.strided')
        meta=dict(overload='aten.add.Tensor',inputs=dict(args=[tensor([2,3]),tensor([3])]),outputs=tensor([2,3]))
        self.assertEqual(elementwise_work(meta),dict(work_flop=6,data_byte=60))
        meta['inputs']['kwargs']={'alpha':2}
        self.assertIsNone(elementwise_work(meta))
        meta['inputs']['kwargs']={}
        meta['inputs']['args'][0]['dtype']='torch.int64'
        self.assertIsNone(elementwise_work(meta))
        meta['inputs']['args'][0]=tensor([0,3]);meta['outputs']=tensor([0,3])
        self.assertEqual(elementwise_work(meta)['work_flop'],0)
        meta['outputs']=tensor([2,3])
        self.assertIsNone(elementwise_work(meta))

    def test_rank_normalization_preserves_operation_arguments(self):
        from rtx5090.operator_parameter_gaps import normalize, differences
        a=dict(rank=0,kernel_names=['a'],inputs=dict(kwargs=dict(rank=0,device='cuda:0')))
        b=dict(rank=1,kernel_names=['b'],inputs=dict(kwargs=dict(rank=1,device='cuda:1')))
        delta=differences(normalize(a),normalize(b))
        self.assertEqual({x['path'] for x in delta},{'$.inputs.kwargs.rank','$.inputs.kwargs.device'})

    def test_missing_is_not_negative_or_out_of_range(self):
        bundle,report=build([inventory(0),inventory(1)])
        reasons={x['reason'] for x in audit(bundle)}
        self.assertIn('missing_k1',reasons)
        self.assertNotIn('negative_thermal_slope',reasons)
        self.assertNotIn('frequency_outside_capacity_calibration',reasons)
        self.assertNotIn('temperature_outside_calibration',reasons)
        bundle['thermal']['0']['k1']=-1
        self.assertIn('negative_thermal_slope',{x['reason'] for x in audit(bundle)})
