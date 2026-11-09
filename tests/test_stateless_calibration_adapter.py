import copy
import unittest
from types import SimpleNamespace
from rtx5090.stateless_calibration_adapter import prepare,invoke
from tests.test_operator_equations import inventory

class AdapterTests(unittest.TestCase):
    def test_exact_call_once_and_unvalidated_evidence(self):
        meta=inventory(0)['signatures']['sig']
        spec=prepare(meta,'train','t',{'train':['t'],'holdout':['h']})
        class Op:
            _schema=SimpleNamespace(is_mutable=False)
            calls=0
            def __str__(self):return 'aten.linear.default'
            def __call__(self,*args,**kwargs):self.calls+=1;return 'result'
        op=Op()
        self.assertEqual(invoke(spec,op,(),{},meta),'result')
        self.assertEqual(op.calls,1)
        self.assertFalse(spec['independent_inputs_verified'])
        changed=copy.deepcopy(meta);changed['inputs']['args'][0]['device']='cuda:1'
        with self.assertRaises(ValueError):invoke(spec,op,(),{},changed)
        self.assertEqual(op.calls,1)
        op._schema=SimpleNamespace(is_mutable=True)
        with self.assertRaises(ValueError):invoke(spec,op,(),{},meta)
    def test_stateful_and_split_rejected(self):
        meta=inventory(0)['signatures']['sig']
        with self.assertRaises(ValueError):prepare(meta,'train','t',{'train':['t'],'holdout':['t']})
        for name in ('vllm.all_reduce.default','vllm.qwen_gdn_attention_core.default','aten.copy_.default'):
            meta['overload']=name
            with self.assertRaises(ValueError):prepare(meta,'train','t',{'train':['t'],'holdout':['h']})
