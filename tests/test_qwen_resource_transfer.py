import copy
import unittest
from rtx5090.qwen_resource_transfer import binding,prepare,evaluate
from rtx5090.observe_core import GPU_UUIDS


def collection():
    def tensor(rank,shape):
        return dict(tensor=True,dtype='torch.bfloat16',device=f'cuda:{rank}',layout='torch.strided',shape=shape,stride=[shape[-1],1])
    windows=[]
    for i,f in enumerate((2250,2527,2527,2250,2250,2527)):
        operators={}
        for rank in (0,1):
            meta=dict(overload='aten.linear.default',inputs=dict(args=[tensor(rank,[1,4]),tensor(rank,[8,4])],kwargs={}),outputs=tensor(rank,[1,8]))
            operators[f'{rank}:decode.0/000000']=dict(uuid=GPU_UUIDS[rank],parent='model.layers.1.mlp.down_proj',shape=meta,median_s=.0001)
        windows.append(dict(requested_mhz=f,operators=operators,actual_readbacks={u:dict(min_mhz=f-20,max_mhz=f,median_mhz=f-10) for u in GPU_UUIDS}))
    return dict(status='two_setting_operators_timing_complete',identity_fingerprint='old',restoration={'verified':True},windows=windows)


def model():
    return dict(schema='resource-model-v1',identity_fingerprint='new',models={f'{r}:linear':dict(
        latency=dict(status='candidate',blockers=[],overhead_s=1e-6,a_s_cycles_per_flop=1,b_s_per_byte=1e-9),
        frequency_range_hz=[2e9,3e9],training_shapes=['[512, 1024, 4096]']) for r in (0,1)})


class TransferTests(unittest.TestCase):
    def test_mapping_work_and_preserve_identity_barrier(self):
        c=collection();before=copy.deepcopy(c);m=prepare(c)
        self.assertEqual(m['counts'],{'linear':2});self.assertEqual(c,before)
        self.assertEqual(m['records'][0]['work']['work_flop'],64)
        frozen=model();unchanged=copy.deepcopy(frozen);result=evaluate(m,frozen)
        self.assertEqual(frozen,unchanged);self.assertEqual(result['scored_labels'],12)
        self.assertEqual(result['blocker_counts']['historical_service_identity_differs'],4)
        self.assertFalse(result['parameters_validated']);self.assertFalse(result['independent_holdout'])
        self.assertTrue(all(r['predicted_energy_j'] is None for r in result['rows']))

    def test_stateful_collective_and_dtype_not_mapped(self):
        for name in ('vllm.all_reduce.default','vllm.unified_attention_with_output.default','aten.to.dtype'):
            self.assertIsNone(binding({'overload':name},0)['work'])
        meta=collection()['windows'][0]['operators']['0:decode.0/000000']['shape']
        meta['inputs']['args'][0]['dtype']='torch.float32'
        self.assertIsNone(binding(meta,0)['work'])

    def test_missing_occurrences_and_contract_drift_rejected(self):
        c=collection();c['windows'][0]['operators'].pop('0:decode.0/000000')
        with self.assertRaises(ValueError):prepare(c)
        c=collection();c['windows'][1]['operators']['0:decode.0/000000']['parent']='other'
        with self.assertRaises(ValueError):prepare(c)

    def test_unidentified_zero_and_out_of_range_retained(self):
        c=collection()
        for w in c['windows']:w['operators']['0:decode.0/000000']['median_s']=0
        m=prepare(c);result=evaluate(m,model())
        self.assertEqual(len(result['rows']),4)
        self.assertEqual(result['blocker_counts']['zero_event_label_unresolved'],2)
        frozen=model();frozen['models']['0:linear']['frequency_range_hz']=[1,2]
        result=evaluate(m,frozen)
        self.assertEqual(result['blocker_counts']['frequency_extrapolation_refused'],2)

    def test_mapping_tamper_rejected(self):
        m=prepare(collection());m['records'][0]['work']['work_flop']=1
        with self.assertRaises(ValueError):evaluate(m,model())

if __name__=='__main__':unittest.main()
