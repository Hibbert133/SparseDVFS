import ctypes as C
import importlib
import json
import socket
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace as NS
from rtx5090.backends import Mock, Bridge, NVML
from rtx5090.bridge import Session, Server
from rtx5090.controller import Controller
from rtx5090.contracts import PROTOCOL, UNITS, validate_plan, engine_args
from rtx5090.run import build_plan
from rtx5090.simulate import results

ROOT=Path(__file__).resolve().parents[1]
GPU=dict(backend='mock',rates_mhz=[900,1500,2100],uuid='GPU-MOCK')

class ProtocolTests(unittest.TestCase):
    def test_idle_memory_state_does_not_remove_high_clock_candidates(self):
        device=NVML.__new__(NVML); device.uuid='GPU-TEST'; device.handle=None
        device._call=lambda *args:None
        device.read=lambda:2527
        def listing(name,*args):
            if name=='nvmlDeviceGetSupportedMemoryClocks': return [405,13801]
            return [180,885] if args[0].value==405 else [180,885,2250,2527,3090]
        device._list=listing
        value=device.probe()
        self.assertEqual(value['rates_mhz'],[180,885,2250,2527,3090])
        self.assertFalse(value['set_reset_verified'])
        self.assertEqual(value['memory_clock_ranges'][0]['max_gpu_mhz'],885)
        self.assertEqual(value['clock_grid_basis'],'union_requires_loaded_apply_validation')

    def test_quantized_engine_and_model_lock(self):
        args=engine_args(); self.assertEqual(args['worker_cls'],'rtx5090.worker.SparseDVFSWorker')
        self.assertNotIn('quantization',args)
        lock=json.loads((ROOT/'configs/5090.lock.json').read_text())
        self.assertEqual(lock['quantization']['quant_method'],'compressed-tensors')
        self.assertEqual(lock['quantization']['config_groups']['group_0']['weights']['num_bits'],4)
    def test_plan_contiguity_gpu_only_and_stale_identity(self):
        profile=dict(identity=dict(backend=PROTOCOL,granularity='module'),synthetic=True,switch_us=100,
                     samples=[dict(mhz=f,latencies={p:{u:1 for u in UNITS} for p in ('prefill','decode')}) for f in (900,1500)])
        plan=build_plan(profile,'profile.json'); validate_plan(plan,profile['identity'])
        self.assertEqual(len(plan['blocks']),2); self.assertEqual(plan['blocks'][0]['triplet'],[0,900000000,0])
        with self.assertRaises(ValueError): validate_plan(plan,dict(backend='thor'))
        plan['blocks'][0]['operators'].reverse()
        with self.assertRaises(ValueError): validate_plan(plan)
    def test_effect_model_has_positive_zero_negative(self):
        rows=results()['scenarios']
        self.assertLess(rows[0]['lookahead']['energy_j'],rows[0]['fixed']['energy_j'])
        self.assertEqual(rows[1]['lookahead'],rows[1]['fixed'])
        self.assertGreater(rows[2]['lookahead']['energy_j'],rows[2]['fixed']['energy_j'])
    def test_session_rejects_auth_uuid_duplicate_and_rate(self):
        for changes in ({'token':'bad'},{'uuid':'GPU-WRONG'},{'seq':0},{'mhz':900000000}):
            d=Mock(); s=Session(d,'x'*32)
            request=dict(token='x'*32,uuid=d.uuid,seq=1,op='set',mhz=900); request.update(changes)
            with self.assertRaises(ValueError): s.handle(request)
            self.assertEqual(d.calls,[])
        d=Mock(); s=Session(d,'x'*32); req=dict(token='x'*32,uuid=d.uuid,seq=1,op='set',mhz=900)
        s.handle(req)
        with self.assertRaises(ValueError): s.handle(req)
        s.close(); self.assertEqual(d.calls,[900,'reset'])
    def test_set_and_restore_failures_surface(self):
        for error in ('not supported','no permission'):
            d=Mock(); s=Session(d,'x'*32)
            with patch.object(d,'set',side_effect=RuntimeError(error)):
                with self.assertRaisesRegex(RuntimeError,error): s.handle(dict(token='x'*32,uuid=d.uuid,seq=1,op='set',mhz=900))
            s.close(); self.assertEqual(d.calls,['reset'])
        d=Mock(); s=Session(d,'x'*32); s.dirty=True
        with patch.object(d,'reset',side_effect=RuntimeError('restore failed')):
            with self.assertRaisesRegex(RuntimeError,'restore failed'): s.close()
    def test_persistent_socket_roundtrip_delay_and_disconnect_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Mock(); server=Server(('127.0.0.1',0),d,'x'*32,Path(tmp)/'bridge.jsonl')
            t=threading.Thread(target=server.serve_forever); t.start()
            try:
                b=Bridge('127.0.0.1',server.server_address[1],'x'*32,d.uuid)
                self.assertTrue(b.probe()['synthetic']); b.set(900); self.assertEqual(b.read(),900)
                original=d.set
                with patch.object(d,'set',side_effect=lambda x:(time.sleep(.01),original(x))): b.set(1500)
                self.assertGreater(b.last_roundtrip_us,5000)
                b.close()
                deadline=time.monotonic()+3
                while 'reset' not in d.calls and time.monotonic()<deadline: time.sleep(.01)
                self.assertEqual(d.current,0)
            finally: server.shutdown(); server.server_close(); t.join()
    def test_nvml_calls_and_status(self):
        calls=[]
        class Lib:
            def nvmlInit_v2(self): return 0
            def nvmlShutdown(self): return 0
            def nvmlDeviceGetHandleByUUID(self,uuid,p): C.cast(p,C.POINTER(C.c_void_p))[0]=C.c_void_p(123); return 0
            def nvmlDeviceSetGpuLockedClocks(self,h,low,high): calls.append((low.value,high.value)); return 0
            def nvmlDeviceResetGpuLockedClocks(self,h): calls.append('reset'); return 0
            def nvmlDeviceGetClockInfo(self,h,kind,p): C.cast(p,C.POINTER(C.c_uint))[0]=900; return 0
        with patch('rtx5090.backends.C.CDLL',return_value=Lib()):
            n=NVML('GPU-TEST'); n.set(900); self.assertEqual(n.read(),900); n.reset(); n.close()
            self.assertEqual(calls,[(900,900),'reset'])
            with patch.object(n.lib,'nvmlDeviceSetGpuLockedClocks',return_value=4):
                with self.assertRaisesRegex(RuntimeError,'error 4'): n.set(1500)


@unittest.skipUnless((ROOT/'build/libfuse_dvfs.so').exists(),'Linux native library required')
class Native5090Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.c=Controller(GPU,self.tmp.name)
        self.addCleanup(self.tmp.cleanup); self.addCleanup(self.c.close)
    def test_units_dedup_queue_cancel_and_reset(self):
        c=self.c; c.set([0,900000000,0]); c.set([0,900000000,0])
        self.assertEqual(c.backend.calls,[900])
        with self.assertRaises(RuntimeError): c.set([1,900000000,0])
        with self.assertRaises(RuntimeError): c.set([0,900000000,1])
        c.enqueue([0,2100000000,0],10000000); c.cancel(); c.enqueue([0,1500000000,0],0)
        deadline=time.monotonic()+3
        while c.backend.current!=1500 and time.monotonic()<deadline: time.sleep(.001)
        self.assertEqual(c.backend.current,1500); c.close(); self.assertEqual(c.backend.current,0)
    def test_ticket_stale_and_unchanged(self):
        c=self.c; lib=c.lib; ptr=C.c_void_p; u=C.c_uint64
        lib.fuse_submission_begin.argtypes=[ptr,u,u]; lib.fuse_submission_begin.restype=u
        for name in ('fuse_gpu_boundary','fuse_gpu_lookahead'):
            getattr(lib,name).argtypes=[ptr,u,u,u]; getattr(lib,name).restype=C.c_int
        a=lib.fuse_submission_begin(c.handle,0,0); b=lib.fuse_submission_begin(c.handle,0,0)
        self.assertEqual(lib.fuse_gpu_boundary(c.handle,a,900000000,0),0)
        self.assertEqual(lib.fuse_gpu_boundary(c.handle,b,1500000000,0),0)
        self.assertEqual(lib.fuse_gpu_lookahead(c.handle,a,2100000000,0),1)
        self.assertEqual(lib.fuse_gpu_lookahead(c.handle,b,1500000000,0),2)
        self.assertEqual(c.backend.calls,[900,1500])
    def test_backend_error_propagation_and_restore(self):
        c=self.c
        with patch.object(c.backend,'set',side_effect=RuntimeError('NVML not supported')):
            with self.assertRaisesRegex(RuntimeError,'backend set'): c.set([0,900000000,0])
        self.assertIn('not supported',c.backend_error)
        with patch.object(c.backend,'reset',side_effect=RuntimeError('restore failed')):
            with self.assertRaisesRegex(RuntimeError,'reset failed'): c.close()
    def test_actual_native_controller_over_persistent_bridge(self):
        d=Mock(); server=Server(('127.0.0.1',0),d,'x'*32,Path(self.tmp.name)/'bridge.jsonl')
        thread=threading.Thread(target=server.serve_forever); thread.start()
        cfg=dict(backend='bridge',uuid=d.uuid,host='127.0.0.1',port=server.server_address[1],rates_mhz=d.rates,default_auto_ack=True)
        try:
            with patch.dict('os.environ',{'SPARSEDVFS_BRIDGE_TOKEN':'x'*32}):
                with Controller(cfg,Path(self.tmp.name)/'over_bridge') as c:
                    c.set([0,900000000,0]); c.set([0,1500000000,0]); c.set([0,1500000000,0])
                    self.assertFalse(c.apply); self.assertEqual(d.calls,[900,1500])
                self.assertEqual(d.current,0)
        finally: server.shutdown(); server.server_close(); thread.join()
    def test_5090_worker_schedule_and_native_core_joined(self):
        Worker5090Tests().exercise(native=True)


class Worker5090Tests(unittest.TestCase):
    def test_real_5090_worker_prefill_decode_and_cross_method_boundary(self):
        self.exercise(native=False)

    def exercise(self,native):
        import test_v3_worker as fx
        from test_v3_contracts import scheduler
        from test_virtual_chain import make_plan, RecordingController
        from sparsedvfs.qwen_runner import Schedule
        fixture=fx.WorkerTests('test_execute_and_sample_are_one_complete_step'); fixture.setUp()
        try:
            sys.modules.pop('rtx5090.worker',None); module=importlib.import_module('rtx5090.worker')
            w=module.SparseDVFSWorker(); w.vllm_config=None; w.use_v2_model_runner=False
            w.model_runner=fixture.worker.model_runner
            with patch.object(module,'validate_engine'),patch.object(module.signal,'signal'),patch.object(module.signal,'SIGUSR2',12,create=True),patch.object(module.atexit,'register'),patch.dict('os.environ',{'SPARSEDVFS_WORKER_CONFIG':''}):
                w.load_model()
            config=dict(mode='instrumented',enabled=True,workload={},out_dir=fixture.temp.name,apply=False,gpu=GPU)
            with patch.object(w,'_sd_identity',return_value=dict(backend=PROTOCOL)): w.sdvfs_configure(config)
            plan=make_plan()
            for i,b in enumerate(plan['blocks']): b['triplet']=[0,[900,1500,2100][i%3]*1000000,0]
            if native:
                c=Controller(GPU,Path(fixture.temp.name)/'native')
                module.torch.cuda.current_stream=lambda:NS(cuda_stream=42,synchronize=lambda:None)
            else: c=RecordingController(); c.boost_khz=0
            w._sd_fuse=c; w._sd_schedule=Schedule(plan,c,'serial' if native else 'lookahead',transport='host' if native else 'cuda')
            for output in [scheduler(prompt=3,count=3)]+[scheduler(computed=i) for i in (3,4,5)]:
                w.execute_model(output)
                if not native: self.assertIsNotNone(w._sd_schedule.ticket)
                w.sample_tokens(None); self.assertIsNone(w._sd_schedule.ticket)
            w._sd_schedule.finalize(); trace=w.sdvfs_export()
            self.assertEqual(len(trace['rows']),272)
            if native:
                self.assertEqual(len([x for x in c.events if x['kind']=='boundary']),16)
                c.close(); self.assertEqual(c.backend.current,0)
            else:
                self.assertEqual(len([x for x in c.events if x['kind']=='begin']),16)
                self.assertEqual(len([x for x in c.events if x['kind']=='end']),16)
            import os
            if os.environ.get('VIRTUAL_TRACE_DIR'):
                Path(os.environ['VIRTUAL_TRACE_DIR'],f'worker_5090_native_{native}.json').write_text(json.dumps(dict(
                    synthetic=True,real_vllm=False,real_cuda=False,native_cpp=native,trace=trace,requests=c.events),indent=2))
        finally:
            if native and 'c' in locals(): c.close()
            sys.modules.pop('rtx5090.worker',None); fixture.tearDown()
