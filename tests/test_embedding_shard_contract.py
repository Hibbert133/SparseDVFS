import copy
import unittest
from rtx5090.operator_parameter_gaps import build
from tests.test_operator_equations import inventory


def pair():
    invs=[inventory(r) for r in (0,1)]
    for r,inv in enumerate(invs):
        def tensor(shape,dtype):
            return dict(tensor=True,shape=shape,dtype=dtype,device=f'cuda:{r}',layout='torch.strided')
        op='_C.vocab_parallel_embedding.default'
        inv['signatures']['sig']=dict(rank=r,phase='prefill',overload=op,
            inputs=dict(args=[tensor([512,5120],'torch.bfloat16'),tensor([512],'torch.int64'),
                tensor([124160,5120],'torch.bfloat16'),r*124160,(r+1)*124160,0,248320,248320]),
            outputs=None,kernel_names=['embedding'])
        for node in inv['nodes']:
            node.update(operator=op,kernels=['embedding'])
    return invs


class EmbeddingShardTests(unittest.TestCase):
    def test_known_out_first_shards_keep_graph_unverified(self):
        bundle,report=build(pair())
        self.assertEqual(report['rank_contract_mismatches'],[])
        self.assertEqual(len(report['recognized_embedding_shards']),2)
        evidence=report['recognized_embedding_shards'][0]
        self.assertEqual(evidence['ranks'][1]['original_range'],[124160,248320])
        self.assertEqual(len(evidence['original_differences']),2)
        self.assertFalse(bundle['rank_join_verified'])
        self.assertIsNone(bundle['operators'][0]['ranks'][0]['work_flop'])

    def test_invalid_or_unreviewed_contract_is_not_whitelisted(self):
        def args(x): return x[1]['signatures']['sig']['inputs']['args']
        changes=[lambda x: args(x).__setitem__(3,124159),
                 lambda x: args(x).__setitem__(3,124161),
                 lambda x: args(x).__setitem__(4,300000),
                 lambda x: args(x).__setitem__(7,248321),
                 lambda x: args(x).__setitem__(5,1),
                 lambda x: args(x).pop(0),
                 lambda x: args(x)[1].update(dtype='torch.float32'),
                 lambda x: x[1]['nodes'][0].update(parent_label='different')]
        for change in changes:
            invs=pair();change(invs)
            _,report=build(invs)
            self.assertTrue(report['rank_contract_mismatches'])

    def test_equal_malformed_tensor_contract_rejected(self):
        invs=pair()
        for inv in invs:
            inv['signatures']['sig']['inputs']['args'][0]['shape']=[1,5120]
        _,report=build(invs)
        self.assertEqual(len(report['rank_contract_mismatches']),2)
