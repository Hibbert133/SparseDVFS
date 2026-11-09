import copy
import asyncio
import itertools
import unittest
from unittest.mock import Mock,patch
from unittest.mock import AsyncMock
from types import SimpleNamespace
from pathlib import Path
import numpy as np
from rtx5090.resource_model import solve,fit_latency,select_budget,resource_class,fit,validate
from rtx5090.resource_worker import specification
from rtx5090.resource_calibration import HostWindow


class ResourceTests(unittest.TestCase):
    def test_budget_is_not_fastest_only(self):
        c={2250:dict(latency_s=1.02,energy_j=1),2527:dict(latency_s=1,energy_j=2)}
        self.assertEqual(select_budget(c,1.03),2250)
        self.assertEqual(select_budget(c,1),2527)
        with self.assertRaises(ValueError):select_budget(c,.9)
        with self.assertRaises(ValueError):select_budget(c,float('nan'))

    def test_exact_sensitivity_matches_corners(self):
        x=np.array([[1,1,1],[2,1,1],[1,2,1],[3,4,1],[4,1,1]],float)
        y=x@np.array([2,3,4]);budget=np.full(5,.005)
        r=solve(x,y,budget)
        corners=np.array([np.linalg.lstsq(x,y*(1+budget*s),rcond=None)[0]
                          for s in map(np.array,itertools.product((-1,1),repeat=5))])
        np.testing.assert_allclose(r['ranges'],np.column_stack((corners.min(0),corners.max(0))))
        self.assertEqual(r['blockers'],[])

    def test_degenerate_design_rejected(self):
        r=solve([[1,1,1]]*5,[1]*5,[0]*5)
        self.assertEqual(r['status'],'unidentified')

    def test_eq1_identified_with_both_regimes(self):
        rows=[]
        for f in (2e9,2.5e9):
            for w,d in ((1e9,1e6),(2e9,1e6),(1e6,1e8),(1e6,2e8)):
                rows.append(dict(work_flop=w,data_byte=d,frequency_hz=f,
                    latency_s=3e-6+max(w/f*1e-5,d*1e-12)))
        r=fit_latency(rows)
        self.assertEqual(r['status'],'candidate')
        self.assertAlmostEqual(r['a_s_cycles_per_flop'],1e-5,places=10)
        self.assertAlmostEqual(r['overhead_s'],3e-6,places=10)

    def test_special_and_memory_are_not_fake_flops(self):
        self.assertEqual(fit_latency([dict(work_flop=None)])['status'],'unidentified')
        rows=[dict(work_flop=0,data_byte=d,latency_s=2e-6+d*1e-12) for d in (1e6,2e6,3e6,4e6)]
        r=fit_latency(rows)
        self.assertEqual(r['a_s_cycles_per_flop'],0)
        self.assertAlmostEqual(r['b_s_per_byte'],1e-12,places=15)

    def test_shape_split_and_unsafe_family(self):
        with self.assertRaises(ValueError):specification('attention',256,'train',1)
        with self.assertRaises(ValueError):specification('linear',256,'holdout',1)
        self.assertEqual(resource_class('vllm.all_reduce.default'),'paired_collective')
        self.assertEqual(resource_class('vllm.qwen_gdn_attention_core.default'),'stateful_attention')

    def test_training_no_holdout_and_duplicate(self):
        with self.assertRaises(ValueError):fit([dict(split='holdout')])
        with self.assertRaises(ValueError):fit([dict(split='train',id='x')]*2)
        with self.assertRaises(ValueError):validate(dict(training_ids=['a']),[dict(split='holdout',id='a')])

    def test_sampler_timeout_kills_child(self):
        h=HostWindow(None,None);h.process=Mock();h.stderr=Mock()
        process=h.process
        process.communicate.side_effect=[TimeoutError('timeout'),(b'',b'')]
        process.poll.return_value=None
        with self.assertRaises(TimeoutError):h.finish()
        process.kill.assert_called_once();self.assertIsNone(h.process)
        h.stderr.close.assert_called_once()

    def test_single_card_set_failure_attempts_pair_restore(self):
        from rtx5090.resource_calibration import collect
        controller=Mock();controller.set.side_effect=RuntimeError('second card failed')
        controller.dirty=[];controller.restore_result={'required':True,'verified':True}
        client=Mock();context=Mock()
        context.__aenter__=AsyncMock(return_value=client);context.__aexit__=AsyncMock(return_value=False)
        identity={'base_url':'unused'};result={'protocol':[{'requested_mhz':2250,'case':{}}]}
        with patch('rtx5090.resource_calibration.telemetry.DualClocks',return_value=controller), \
             patch('rtx5090.resource_calibration.service_client.BenchClient',return_value=context), \
             patch('rtx5090.resource_calibration.service.model_id',return_value='unused'), \
             patch('rtx5090.resource_calibration.service.load_frames',new=AsyncMock(return_value=[])), \
             patch('rtx5090.resource_calibration.service.restore_checked',new=AsyncMock()) as restore:
            with self.assertRaisesRegex(RuntimeError,'second card failed'):
                asyncio.run(collect(SimpleNamespace(seed=1),identity,[],Path('unused'),result))
            restore.assert_awaited_once()
        self.assertEqual(result['restoration'],controller.restore_result)

    def test_budget_policy_needs_explicit_contract(self):
        from tests.test_operator_equations import fixture
        from rtx5090.operator_equations import predict
        b=fixture();b.pop('selection_policy')
        self.assertEqual(predict(b)['status'],'blocked_parameter_evidence')
        b.update(operator_latency_budgets_s={'op':.002},budget_provenance='synthetic allocation')
        r=predict(b)
        self.assertFalse(r['blockers']);self.assertEqual(r['selection_policy'],'latency_budget')

    def test_frozen_model_to_holdout_no_parameter_mutation(self):
        rows=[]
        for rate in (2250,2527):
            for temp in (35,55,65):
                for w,d in ((1e9,1e6),(1e6,1e8)):
                    f=rate*1e6;v=.9 if rate==2250 else 1.05
                    basis=[10*v*v*f,10*temp*v,10*v]
                    rows.append(dict(id=str(len(rows)),split='train',identity_fingerprint='synthetic',rank=0,
                        family='linear',shape=[w,d],work_flop=w,data_byte=d,frequency_hz=f,voltage_v=v,
                        temperature_c=temp,memory_mhz=13801,requested_mhz=rate,data_semantics='logical_tensor_bytes',
                        integral_basis=basis,energy_j=float(np.dot(basis,[2e-8,.5,40])),energy_relative_budget=0,
                        latency_s=3e-6+max(w/f*1e-3,d*1e-12)))
        model=fit(rows);before=copy.deepcopy(model)
        row=copy.deepcopy(rows[0]);row.update(id='held',split='holdout',shape=[111,222],
            requested_mhz=2385,frequency_hz=2385e6,voltage_v=1,temperature_c=50)
        row['integral_basis']=[10*2385e6,500,10]
        row['energy_j']=float(np.dot(row['integral_basis'],[2e-8,.5,40]))
        row['latency_s']=3e-6+max(row['work_flop']/row['frequency_hz']*1e-3,row['data_byte']*1e-12)
        result=validate(model,[row]);self.assertEqual(model,before)
        self.assertTrue(result['numeric_limits_passed']);self.assertFalse(result['parameters_validated'])

if __name__=='__main__':unittest.main()
