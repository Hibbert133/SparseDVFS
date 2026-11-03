"""Orchestrator fixtures only: do not claim vLLM/CUDA/hardware execution."""
import copy
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
from rtx5090.experiment import window, dataset_contract, tune, compare
from rtx5090.contracts import digest_json, UNITS
from rtx5090.run import write, read


class WindowTests(unittest.TestCase):
    def test_measurement_preserves_raw_telemetry_and_restores(self):
        self.run_window(False)

    def test_output_failure_still_closes_worker_and_preserves_meter_data(self):
        self.run_window(True)

    def run_window(self, fail):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root=Path(tmp); stopped=[]; calls=[]
            data=dict(input_tokens=512,output_tokens=128,model_manifest_sha256='model',
                      splits=dict(train=[[1]*512],test=[[2]*512]))
            write(root/'data.json',data)
            write(root/'base.json',dict(workload=dataset_contract(data),real_weights=True,environment={},
                  outputs={digest_json([1]*512):'correct',digest_json([2]*512):'correct'}))
            cfg=dict(backend='nvml',uuid='GPU-T',rates_mhz=[900,1500],default_auto_ack=True)
            a=NS(apply=True,ack_default_auto=True,model=tmp,dataset=str(root/'data.json'),
                 baseline=str(root/'base.json'),out=tmp,gpu_uuid='GPU-T',calibration='unused',power_evidence=None,min_seconds=30)
            class LLM:
                def __init__(self,**kwargs):self.llm_engine=NS(engine_core=NS(shutdown=lambda:stopped.append('worker')))
                def collective_rpc(self,fn):return [{'uuid':'GPU-T','pid':123,'runner':'fixture'}]
            class Meter:
                def __init__(self,*args):self.rows=[{'fixture':'raw telemetry'}]
                def start(self):calls.append('start')
                def stop(self):calls.append('stop');return dict(energy_j=100,seconds=30,uncertainty_fraction=.005)
            n=[0]
            def generate(*args):
                n[0]+=1
                return dict(output_sha256='bad' if fail and n[0]==4 else 'correct',elapsed_s=.1)
            def rpc(llm,name,**kwargs):
                calls.append((name,kwargs))
                if name=='sdvfs_configure':return {'fixture':'identity'}
                if name=='sdvfs_gpu_readback':return {'observed_mhz':900}
                if name=='sdvfs_export':
                    phases=(['prefill']+['decode']*127)*2
                    return dict(steps=[dict(phase=p,order=UNITS) for p in phases],
                                rows=[dict(phase=p,name=u,latency_ms=1,stream=42) for p in phases for u in UNITS])
            stack.enter_context(patch.dict(sys.modules,{'torch':NS(__version__='fixture',version=NS(cuda='12.9'),cuda=NS(get_device_name=lambda:'fixture')),
                                                       'vllm':NS(__version__='0.32.0',LLM=LLM)}))
            for name,value in [('verify_model',lambda *a,**k:dict(manifest_sha256='model')),
                               ('environment_record',lambda:{}),('identity',lambda *a:{'fixture':'identity'}),
                               ('generate',generate),('rpc',rpc),('Meter',Meter),('background',lambda x:{}),
                               ('backend',lambda c:NS(close=lambda:stopped.append('monitor')))]:
                stack.enter_context(patch('rtx5090.experiment.'+name,value))
            job=dict(gpu=cfg,mode='fixed',mhz=900,requests=2)
            if fail:
                with self.assertRaisesRegex(RuntimeError,'Output differs'):window(a,job)
            else:
                row=window(a,job)
                self.assertEqual(row['tokens'],256);self.assertTrue(row['coverage_complete'])
                self.assertTrue(row['restoration_ok']);self.assertEqual(row['energy']['energy_j'],100)
            self.assertEqual(read(root/'telemetry.json'),[{'fixture':'raw telemetry'}])
            self.assertEqual(set(stopped),{'worker','monitor'})
            self.assertTrue(any(name=='sdvfs_export' and kw.get('close') for name,kw in (c for c in calls if isinstance(c,tuple))))

    def test_frozen_selection_and_pair_order_end_to_end(self):
        # Entire pipeline runs with a substituted launch function. Results stay
        # in this temporary test directory and are never a hardware report.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); gpu=dict(backend='nvml',uuid='GPU-T',rates_mhz=[900,1500],default_auto_ack=True)
            for name,value in [('gpu',gpu),('data',{}),('cal',{})]:write(root/(name+'.json'),value)
            profile=dict(identity=dict(backend=__import__('rtx5090.contracts',fromlist=['PROTOCOL']).PROTOCOL,granularity='module'),
                         synthetic=False,switch_us=50,switch_source='measured',samples=[dict(mhz=f,latencies={p:{u:1 for u in UNITS} for p in ('prefill','decode')}) for f in (900,1500)])
            write(root/'profile.json',profile)
            a=NS(apply=True,ack_default_auto=True,gpu=str(root/'gpu.json'),gpu_uuid='GPU-T',
                 model=tmp,out=tmp,profile=str(root/'profile.json'),calibration=str(root/'cal.json'),
                 dataset=str(root/'data.json'),selection=str(root/'selection.json'),pairs=10)
            seen=[]
            def launch(args,folder,**job):
                seen.append(job)
                controlled=job['mode']!='auto';count=job.get('requests',8)
                return dict(identity=profile['identity'],outputs=['same']*count,tokens=128*count,
                            requests=[dict(elapsed_s=1)]*count,mean_latency_s=1,
                            energy=dict(energy_j=count*(90 if controlled else 100),seconds=31,uncertainty_fraction=.005),
                            real_weights=True,real_cuda=True,synthetic=False,split=job['split'],restoration_ok=True,
                            output_checked=True,frequency_applied=controlled,coverage_complete=controlled,
                            pair_id=job.get('pair_id'),selection_sha256=job.get('selection_sha256'))
            with patch('rtx5090.experiment.launch',side_effect=launch),patch('rtx5090.experiment.verify_model'),patch('builtins.print'):
                tune(a)
                self.assertTrue(all(j['split']=='train' for j in seen))
                self.assertEqual(read(a.selection)['mode'],'serial') # equal-energy lookahead is not selected
                compare(a)
            report=read(root/'report.json')
            self.assertTrue(report['success'])
            order=[p['order'][0] for p in read(root/'pairs.json')]
            self.assertEqual(set(order),{'auto','sparse'})
            self.assertEqual(len([j for j in seen if j['split']=='test']),20)
