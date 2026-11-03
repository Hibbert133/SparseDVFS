import copy
import ctypes as C
import json
from pathlib import Path
import socket
import tempfile
import threading
import unittest
from unittest.mock import patch
from types import SimpleNamespace as NS
from rtx5090.backends import Mock, Bridge, NVML
from rtx5090.bridge import Server
from rtx5090.energy import summarize_energy, acceptance
from rtx5090.planning import candidate, select_rates
from rtx5090.contracts import UNITS, PROTOCOL, validate_plan, digest_json
from rtx5090.experiment import dataset_contract, require_apply, launch


def samples():
    return [dict(uuid='GPU-T',synthetic=False,timestamp_ns=i*1_000_000_000,
                 energy_mj=i*100_000,power_mw=100_000,roundtrip_s=.001) for i in range(31)]


def pairs(ratio=.9,latency=1.01):
    rows=[]
    for i in range(10):
        auto=dict(pair_id=i,real_weights=True,real_cuda=True,synthetic=False,split='test',
                  restoration_ok=True,output_checked=True,frequency_applied=False,
                  identity={'same':'runtime'},outputs=['same-tokens'],tokens=128,
                  mean_latency_s=1,requests=[dict(elapsed_s=1)],
                  energy=dict(energy_j=100,seconds=31,uncertainty_fraction=.005))
        sparse=copy.deepcopy(auto)
        sparse.update(frequency_applied=True,coverage_complete=True,mean_latency_s=latency,
                      selection_sha256='frozen',requests=[dict(elapsed_s=latency)])
        sparse['energy']['energy_j']*=ratio
        rows.append(dict(pair_id=i,auto=auto,sparse=sparse))
    return rows


class EnergyTests(unittest.TestCase):
    def test_counter_units_and_source_clock(self):
        result=summarize_energy(samples(),'GPU-T')
        self.assertEqual(result['energy_j'],3000)
        self.assertEqual(result['seconds'],30)
        self.assertEqual(result['method'],'nvml_total_energy_counter')

    def test_invalid_readings_and_counter_reset(self):
        for change in (lambda r:r[0].update(synthetic=True),lambda r:r[0].update(uuid='GPU-other'),
                       lambda r:r[10].update(energy_mj=-1),lambda r:r[10].update(timestamp_ns=0)):
            rows=samples(); change(rows)
            with self.assertRaises(ValueError): summarize_energy(rows,'GPU-T')
        with self.assertRaisesRegex(ValueError,'30 seconds'): summarize_energy(samples()[:3],'GPU-T')

    def test_power_fallback_needs_evidence_and_cadence(self):
        rows=samples()
        for r in rows:r['energy_mj']=None
        with self.assertRaisesRegex(ValueError,'accuracy evidence'): summarize_energy(rows,'GPU-T')
        evidence=dict(uuid='GPU-T',validated=True,max_relative_error=.02,max_sample_interval_s=1.1,
                      reference_method='test fixture only; not an actual calibration')
        result=summarize_energy(rows,'GPU-T',evidence)
        self.assertEqual(result['energy_j'],3000)
        self.assertGreaterEqual(result['uncertainty_fraction'],.02)
        evidence['max_sample_interval_s']=.1
        with self.assertRaisesRegex(ValueError,'Sampling gaps'): summarize_energy(rows,'GPU-T',evidence)

    def test_positive_zero_negative_and_latency_failure(self):
        self.assertTrue(acceptance(pairs())['success'])
        for ratio,latency in ((1,1),(1.1,1),(.9,1.06),(.999,1)):
            self.assertFalse(acceptance(pairs(ratio,latency))['success'])

    def test_mock_dummy_training_mismatch_and_restore_rejected(self):
        changes=[dict(synthetic=True),dict(real_weights=False),dict(real_cuda=False),dict(split='train'),
                 dict(restoration_ok=False),dict(coverage_complete=False),dict(outputs=['wrong']),
                 dict(identity={'other':1}),dict(selection_sha256='changed'),dict(frequency_applied=False)]
        for change in changes:
            rows=pairs(); rows[3]['sparse'].update(change)
            with self.subTest(change=change),self.assertRaises(ValueError): acceptance(rows)
        with self.assertRaises(ValueError):acceptance(pairs()[:9])
        rows=pairs(); rows[2]=rows[1]
        with self.assertRaisesRegex(ValueError,'Duplicate'): acceptance(rows)

    def test_nvml_energy_width_and_power_units(self):
        class Lib:
            def nvmlInit_v2(self):return 0
            def nvmlShutdown(self):return 0
            def nvmlDeviceGetHandleByUUID(self,uuid,p):return 0
            def nvmlDeviceGetTotalEnergyConsumption(self,h,p):C.cast(p,C.POINTER(C.c_ulonglong))[0]=2**40;return 0
            def nvmlDeviceGetPowerUsage(self,h,p):C.cast(p,C.POINTER(C.c_uint))[0]=200000;return 0
        with patch('rtx5090.backends.C.CDLL',return_value=Lib()):
            d=NVML('GPU-T'); row=d.telemetry();d.close()
            self.assertEqual(row['energy_mj'],2**40)
            self.assertEqual(row['power_mw'],200000)
            self.assertIn('temperature_c',row['errors'])


class PlanningTests(unittest.TestCase):
    def profile(self):
        return dict(identity=dict(backend=PROTOCOL,granularity='module'),synthetic=False,
                    switch_us=500,switch_source='measured',samples=[dict(mhz=f,latencies={p:{u:(.1 if j%2==0 or f==1500 else .2)
                    for j,u in enumerate(UNITS)} for p in ('prefill','decode')}) for f in (900,1500)])
    def test_partition_preserves_coverage_and_disables_other_domains(self):
        plan=candidate(self.profile(),'profile.json')
        validate_plan(plan)
        self.assertLess(len(plan['blocks']),len(UNITS)*2)
        self.assertTrue(plan['partition_decisions'])
        self.assertTrue(all(b['triplet'][0]==b['triplet'][2]==0 for b in plan['blocks']))
        self.assertEqual([u for b in plan['blocks'] if b['phase']=='decode' for u in b['operators']],UNITS)
    def test_synthetic_profiles_rejected(self):
        value=self.profile();value['synthetic']=True
        with self.assertRaises(ValueError):candidate(value,'profile.json')
    def test_frequency_selection(self):
        self.assertEqual(select_rates([1500,1800,2100,2400,2700,3000],2700),[1800,2100,2400,2700])
        with self.assertRaises(ValueError):select_rates([1500],1500)
    def test_dataset_disjoint_and_fixed_shape(self):
        value=dict(input_tokens=512,output_tokens=128,splits=dict(train=[[1]*512],test=[[2]*512]))
        self.assertEqual(len(dataset_contract(value)['allowed_prompt_sha256']),2)
        value['splits']['test']=[[1]*512]
        with self.assertRaisesRegex(ValueError,'disjoint'):dataset_contract(value)
    def test_apply_is_explicit_and_cannot_be_mock(self):
        a=NS(apply=False,ack_default_auto=True)
        with self.assertRaisesRegex(ValueError,'--apply'):require_apply(a,dict(backend='nvml'))
        a.apply=True
        with self.assertRaisesRegex(ValueError,'Mock'):require_apply(a,dict(backend='mock'))

    def test_subprocess_passes_model_gpu_environment_and_apply(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=NS(model=tmp,gpu_uuid='GPU-T',library_dir=tmp,dataset=tmp+'/data.json',baseline=tmp+'/base.json',
                 min_seconds=30,calibration=None,power_evidence=None,apply=True,ack_default_auto=True)
            def fake_run(cmd,**kwargs):
                self.assertIn('--apply',cmd);self.assertIn('--ack-default-auto',cmd)
                self.assertEqual(cmd[cmd.index('--model')+1],str(Path(tmp).resolve()))
                self.assertEqual(cmd[cmd.index('--gpu-uuid')+1],'GPU-T')
                self.assertNotIn('shell',kwargs)
                (Path(tmp)/'result.json').write_text('{"fixture":true}')
            with patch('rtx5090.experiment.subprocess.run',side_effect=fake_run):
                self.assertEqual(launch(a,tmp,mode='auto'),dict(fixture=True))


class BridgeTelemetryTests(unittest.TestCase):
    def test_monitor_does_not_own_or_reset_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            device=Mock(uuid='GPU-T');server=Server(('127.0.0.1',0),device,'t'*32,Path(tmp)/'log')
            t=threading.Thread(target=server.serve_forever);t.start()
            clients=[]
            try:
                for _ in range(3):clients.append(Bridge('127.0.0.1',server.server_address[1],'t'*32,'GPU-T'))
                control,monitor,other=clients
                control.set(900);self.assertEqual(monitor.telemetry()['clock_mhz'],900)
                monitor.close();clients.remove(monitor)
                self.assertEqual(device.current,900)
                with self.assertRaisesRegex(RuntimeError,'owns GPU'):other.set(1500)
                control.reset();self.assertEqual(device.current,0)
            finally:
                for c in clients:c.close()
                server.shutdown();server.server_close();t.join()
    def test_read_only_server_rejects_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Mock(uuid='GPU-T');server=Server(('127.0.0.1',0),d,'t'*32,Path(tmp)/'log',allow_apply=False)
            t=threading.Thread(target=server.serve_forever);t.start()
            try:
                c=Bridge('127.0.0.1',server.server_address[1],'t'*32,'GPU-T')
                self.assertTrue(c.telemetry()['synthetic'])
                with self.assertRaisesRegex(RuntimeError,'without --apply'):c.reset()
                self.assertEqual(d.calls,[]);c.close()
            finally:server.shutdown();server.server_close();t.join()
