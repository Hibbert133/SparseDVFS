import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace as NS,ModuleType
from unittest.mock import patch,Mock
from concurrent.futures import ThreadPoolExecutor
from rtx5090.eager_ipc import Rendezvous
from rtx5090.observe_core import GPU_UUIDS
from rtx5090.operator_boundary_test import boundaries
from tests.test_operator_pair_contract import fixture
from tests.test_eager_operator_cost import Base

class BoundaryTests(unittest.TestCase):
    def test_plan_has_real_input_contracts(self):
        self.assertEqual(len(boundaries(fixture())),4)
        self.assertIn('contracts',boundaries(fixture())[0])
    def test_peer_timeout_poisoned(self):
        gate=Rendezvous(lambda key:{},'fp',timeout=.01)
        msg=dict(rank=0,uuid=GPU_UUIDS[0],fingerprint='fp',seq=0,gpu_complete=True,key=['r','prefill',0])
        with self.assertRaises(TimeoutError):gate.exchange(msg)
        msg.update(rank=1,uuid=GPU_UUIDS[1])
        with self.assertRaises(RuntimeError):gate.exchange(msg)
    def test_pair_releases_once(self):
        action=Mock(return_value={'released':True});gate=Rendezvous(action,'fp',timeout=1)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results=list(pool.map(gate.exchange,[dict(rank=r,uuid=GPU_UUIDS[r],fingerprint='fp',seq=0,
                 gpu_complete=True,key=['request','prefill',0]) for r in (0,1)]))
        self.assertEqual(len(results),2);action.assert_called_once()
    def test_worker_call_once_and_mismatch_prevents_execution(self):
        worker=ModuleType('rtx5090.observe_worker');worker.Observer=Base
        metadata=ModuleType('sparsedvfs_vllm.operators');metadata.tensor_metadata=lambda x,t:{}
        cuda=NS(synchronize=Mock())
        with patch.dict(sys.modules,{'torch':NS(cuda=cuda),'rtx5090.observe_worker':worker,
                                    'sparsedvfs_vllm.operators':metadata}):
            spec=importlib.util.spec_from_file_location('rtx5090._boundary_test',Path(__file__).parents[1]/'rtx5090/operator_boundary_worker.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        context=dict(prompt_tokens=512,admitted_computed_tokens=0,resumed=False,preempted=False,speculative_tokens=0,
                     phase='decode',num_scheduled_tokens=1,num_computed_tokens=512,request_id='r')
        class Op:
            calls=0
            def __str__(self):return 'aten.linear.default'
            def __call__(self):self.calls+=1;return None
        with tempfile.TemporaryDirectory() as root:
            c=dict(context='decode.0',operator='aten.linear.default',parent='model.layers.1.attn',source_ordinals=[0,0],
                   contracts=[dict(overload='aten.linear.default',inputs={},outputs={})]*2)
            command=dict(out=root,boundaries=[c],coordinator='unused',fingerprint='fp')
            obs=module.BoundaryObserver(None,0,lambda:context,command,{'gpu_uuid':GPU_UUIDS[0]})
            with patch.object(module,'rpc',return_value=dict(released=True,seq=0,clock_writes=False)):
                op=Op();obs.ordinals['decode.0']=17
                obs.dispatch(op,(),{});obs.finish();self.assertEqual(op.calls,1)
                bad=module.BoundaryObserver(None,0,lambda:context,command,{'gpu_uuid':GPU_UUIDS[0]})
                bad.candidates['decode.0']['contracts']=[dict(overload='aten.linear.default',inputs={'wrong':True},outputs={})]*2
                with self.assertRaises(ValueError):bad.dispatch(op,(),{})
                self.assertEqual(op.calls,1)

class FailureRetentionTests(unittest.TestCase):
    def test_first_failure_survives_sample_callback(self):
        import json
        from tests.test_eager_worker import WorkerTests
        with tempfile.TemporaryDirectory() as root:
            worker,_=WorkerTests().worker(root)
            worker._eg_failure(ValueError('original contract error'))
            worker._eg_failure(RuntimeError('secondary ordering error'))
            with self.assertRaisesRegex(RuntimeError,'original contract error'):
                worker.sample_tokens(None)
            record=json.loads((Path(root)/'rank-0-failure.json').read_text())
            self.assertIn('original contract error',record['error'])
    def test_duplicate_semantic_candidate_rejected(self):
        data=fixture()
        for window in data['windows']:
            for rank in (0,1):
                window['operators'][f'{rank}:prefill/000000']['shape']['overload']='aten.linear.default'
        with self.assertRaisesRegex(ValueError,'not unique'):boundaries(data)
