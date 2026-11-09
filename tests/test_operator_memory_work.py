import copy
import unittest
from rtx5090.operator_memory_work import memory_work
from rtx5090.operator_equations import audit,predict
from rtx5090.operator_parameter_gaps import build
from tests.test_operator_equations import fixture,inventory


def tensor(dtype='torch.bfloat16'):
    return dict(tensor=True,shape=[2,3],stride=[3,1],dtype=dtype,device='cuda:0',layout='torch.strided')


def meta(dtype='torch.bfloat16'):
    dst=tensor(dtype)
    return dict(overload='aten.copy_.default',inputs=dict(args=[dst,tensor()]),outputs=copy.deepcopy(dst))


class MemoryWorkTests(unittest.TestCase):
    def test_copy_logical_bytes_and_nonzero_model_time(self):
        w=memory_work(meta(),['copy'])
        self.assertEqual((w['work_flop'],w['data_byte']),(0,24))
        b=fixture()
        for row in b['operators'][0]['ranks']:
            row.update(work_flop=0,data_byte=24)
        result=predict(b)
        self.assertFalse(result['blockers'])
        self.assertAlmostEqual(result['operators']['op']['predictions']['2250']['latency_s'],24/1e7+1e-6)

    def test_conversion_bytes_do_not_invent_flops(self):
        m=meta('torch.float32');w=memory_work(m,['cast'])
        self.assertIsNone(w['work_flop']);self.assertEqual(w['data_byte'],36)
        m['overload']='aten.to.dtype';m['inputs']['args']=[tensor(),'torch.float32']
        self.assertEqual(memory_work(m,['cast']),w)
        b=fixture()
        for row in b['operators'][0]['ranks']:
            row.update(work_flop=0,data_byte=36,unresolved_workload=['conversion_compute_path'])
        self.assertIn('workload_compute_path_unresolved',{r['reason'] for r in audit(b)})

    def test_unknown_traffic_rejected(self):
        self.assertIsNone(memory_work(meta(),[]))
        changes=[lambda m:m['inputs']['args'][1].update(device='cpu'),
                 lambda m:m['inputs']['args'][1].update(shape=[1,3]),
                 lambda m:m['inputs']['args'][1].update(stride=[1,2]),
                 lambda m:m['inputs']['args'][1].update(dtype='torch.complex64'),
                 lambda m:m['outputs'].update(dtype='torch.float64')]
        for change in changes:
            m=meta();change(m);self.assertIsNone(memory_work(m,['copy']))
        m=meta();m['overload']='aten.to.dtype';m['inputs']['args']=[tensor(),'torch.bfloat16']
        self.assertIsNone(memory_work(m,['copy']))

    def test_inventory_marks_partial_coverage_and_keeps_evidence_gates(self):
        invs=[inventory(r) for r in (0,1)]
        for r,inv in enumerate(invs):
            m=meta('torch.float32')
            for t in m['inputs']['args']+[m['outputs']]:t['device']=f'cuda:{r}'
            m.update(rank=r,phase='decode',kernel_names=['cast'])
            inv['signatures']['sig']=m
            for n in inv['nodes']:n.update(operator=m['overload'],kernels=['cast'])
        bundle,report=build(invs)
        self.assertEqual(report['static_work_coverage']['bytes_only'],4)
        self.assertEqual(report['static_work_coverage']['derived'],0)
        self.assertFalse(bundle['rank_join_verified'])
        self.assertEqual(predict(bundle)['status'],'blocked_parameter_evidence')
