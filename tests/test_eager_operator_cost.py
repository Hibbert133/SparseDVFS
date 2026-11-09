"""Mock execution checks; not CUDA timing certification."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch


class Event:
    def __init__(self, **kwargs):pass
    def record(self, stream):pass


class Base:
    def __init__(self, model, rank, context, limit):
        self.rank,self.context,self.limit=rank,context,limit
        self.rows,self.events,self.parents=[],[],['model.layers.1.attn']
        self.busy=False


class OperatorExecutionTests(unittest.TestCase):
    def test_stateful_operator_exactly_once_and_later_context_untouched(self):
        worker=ModuleType('rtx5090.observe_worker');worker.Observer=Base
        metadata=ModuleType('sparsedvfs_vllm.operators');metadata.tensor_metadata=lambda x,t:{}
        stream=SimpleNamespace(cuda_stream=7)
        torch=SimpleNamespace(cuda=SimpleNamespace(current_stream=lambda:stream,Event=Event))
        with patch.dict(sys.modules,{'torch':torch,'rtx5090.observe_worker':worker,
                                    'sparsedvfs_vllm.operators':metadata}):
            path=Path(__file__).resolve().parents[1]/'rtx5090/eager_operator_cost.py'
            spec=importlib.util.spec_from_file_location('rtx5090._operator_cost_test',path)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        context=dict(prompt_tokens=512,admitted_computed_tokens=0,resumed=False,preempted=False,
                     speculative_tokens=0,phase='decode',num_scheduled_tokens=1,num_computed_tokens=512)
        observer=module.OperatorCost(None,0,lambda:context)
        calls=[]
        def attention():calls.append('mutate-state');return 'result'
        self.assertEqual(observer.dispatch(attention,(),{}),'result')
        self.assertEqual(len(calls),1)
        self.assertEqual(len(observer.rows),1)
        self.assertFalse(observer.busy)
        context['num_computed_tokens']=513
        observer.dispatch(attention,(),{})
        self.assertEqual(len(calls),2)
        self.assertEqual(len(observer.rows),1)
        for position in (31,62):
            context['num_computed_tokens']=512+position
            observer.dispatch(attention,(),{})
        self.assertEqual(len(observer.rows),3)
        context['num_computed_tokens']=512
        def failed():raise RuntimeError('CUDA failed')
        with self.assertRaises(RuntimeError):observer.dispatch(failed,(),{})
        self.assertFalse(observer.busy)
        self.assertEqual(len(observer.rows),4)  # Failed raw occurrence retained.
