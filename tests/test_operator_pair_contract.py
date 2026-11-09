import copy
import unittest
from rtx5090.operator_pair_contract import audit,CONTEXTS
from rtx5090.observe_core import GPU_UUIDS


def fixture():
    ops={}
    for rank in (0,1):
        for ctx in CONTEXTS:
            for index,op in enumerate(('vllm.all_reduce.default','aten.linear.default')):
                ops[f'{rank}:{ctx}/{index:06d}']=dict(uuid=GPU_UUIDS[rank],layer=1,
                    parent='model.layers.1.mlp.down_proj',source_ordinal=index+20,
                    shape=dict(overload=op,inputs=dict(args=[dict(tensor=True,device=f'cuda:{rank}',shape=[1,4])]),outputs=None),median_s=.001)
    return dict(status='two_setting_operators_timing_complete',restoration=dict(verified=True),identity_fingerprint='fixture',
        windows=[dict(requested_mhz=f,operators=copy.deepcopy(ops)) for f in (2250,2527,2527,2250,2250,2527)])

class PairTests(unittest.TestCase):
    def test_matched_stays_non_deployable(self):
        r=audit(fixture());self.assertEqual(len(r['candidates']),4)
        self.assertFalse(r['runtime_boundary_verified']);self.assertFalse(r['graph_join_verified'])
    def test_rank_argument_mismatch_blocks_all_candidates(self):
        d=fixture()
        for w in d['windows']:w['operators']['1:decode.0/000001']['shape']['inputs']['args'].append(7)
        r=audit(d);self.assertEqual(r['candidates'],[]);self.assertTrue(r['mismatches'])
    def test_incomplete_restore_rejected(self):
        d=fixture();d['restoration']['verified']=False
        with self.assertRaises(ValueError):audit(d)
