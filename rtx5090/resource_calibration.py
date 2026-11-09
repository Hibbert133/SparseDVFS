"""Resident resource collection with Windows-local voltage/energy and pair recovery.

Training and holdout are separate commands; fit must be frozen before holdout.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import queue
import signal
import shutil
import statistics
import subprocess
import tempfile
import threading
from . import eager, service_model as service, service_client, service_telemetry as telemetry
from .eager_ipc import LocalServer, Rendezvous
from .observe_core import GPU_UUIDS, digest
from .service_core import read, write_new
from .resource_worker import FAMILIES, window_seconds


class HostWindow:
    def __init__(self, out, python):
        self.out,self.python=out,python;self.process=None;self.stderr=None

    def start(self):
        win=lambda p:subprocess.check_output(['wslpath','-w',str(p)],text=True).strip()
        env=dict(os.environ,PYTHONPATH=win(Path(__file__).resolve().parents[1]));env.pop('PYTHONHOME',None)
        self.stderr=(self.out/'windows-stderr.log').open('wb')
        self.process=subprocess.Popen([self.python,'-B','-m','rtx5090.host_load_check','--host','--voltage',
            '--out',win(self.out)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,env=env)
        channel=queue.Queue()
        threading.Thread(target=lambda:channel.put(self.process.stdout.readline()),daemon=True).start()
        try:ready=json.loads(channel.get(timeout=25))
        except queue.Empty:raise TimeoutError('Windows sampler startup timeout')
        if ready.get('status')!='host_ready':raise RuntimeError('Host sampler not ready')

    def finish(self):
        if self.process is None:return
        process,self.process=self.process,None
        try:
            stdout,_=process.communicate(b'finish\n',timeout=25)
            (self.out/'windows-stdout.log').write_bytes(stdout)
            if process.returncode:raise RuntimeError('Windows sampler failed')
        except BaseException:
            if process.poll() is None:process.kill();process.communicate(timeout=5)
            raise
        finally:
            if self.stderr:self.stderr.close()


async def one(identity, client, out, spec, a):
    out.mkdir();request=client.prepare(512,spec['seed'],'resource-'+out.name)
    isolation=await eager.isolate_observation_cache(identity,request)
    write_new(out/'request.json',request)
    host=HostWindow(out,a.windows_python)
    with tempfile.TemporaryDirectory(prefix='sdvfs-resource-',dir='/tmp') as runtime:
        os.chmod(runtime,0o700)
        def action(key):
            if key!=[request['request_id'],'resource',key[-1]] or key[-1] not in (0,1):raise ValueError('Unknown resource boundary')
            host.start() if key[-1]==0 else host.finish()
            return dict(released=True)
        gate=Rendezvous(action,identity['fingerprint'],timeout=35)
        server=LocalServer(Path(runtime)/'pair.sock',gate.exchange,timeout=45,on_disconnect=gate.abort)
        try:
            eager.rank_commands(identity,dict(op='arm',mode='resource_calibration',apply=True,out=str(out),
                request_id=request['request_id'],prompt_hash=request['prompt_hash'],resource_case=spec,
                resource_window_s=a.window_seconds,
                resource_coordinator=str(server.path),fingerprint=identity['fingerprint']))
            reply=await client.run(request)
            coverage=eager.coverage(out,identity)
            write_new(out/'observation.json',dict(reply=reply,coverage=coverage,cache_isolation=isolation))
            if reply['usage']['prompt_tokens']!=512 or reply['usage']['completion_tokens']!=64:raise ValueError('Usage changed')
        finally:
            gate.abort('resource request finished')
            try:await asyncio.to_thread(host.finish)
            finally:
                server.close()
                eager.rank_commands(identity,dict(op='arm',mode='off'))
    from .host_reference import local_energy
    from .eq2_identifiability import features
    raw=read(out/'windows-samples.json');energy=local_energy(raw)
    write_new(out/'energy.json',energy)
    if energy['quality_errors']:raise RuntimeError('Resource energy gate: '+str(energy['quality_errors']))
    rows=[]
    for rank,uuid in enumerate(GPU_UUIDS):
        report=read(out/f'rank-{rank}-resource.json')
        if report['status']!='resource_labels_collected' or report['identity_sha256']!=digest(identity['ranks'][rank]):raise ValueError('Resource worker evidence mismatch')
        actual_duration=(report['power_work_end_ns']-report['power_work_begin_ns'])/1e9
        if report.get('requested_window_s')!=a.window_seconds or actual_duration<a.window_seconds:
            raise ValueError('Resource duration contract mismatch; resident worker may be stale')
        f=features(raw,uuid)
        # Entire pair envelope, including coordinator and synchronization gaps.
        # No allocation of a mixed inference window to individual calls.
        values=[s for frame in raw['frames'] for s in frame if s['uuid']==uuid]
        row=dict(id=out.name+f'-rank{rank}',rank=rank,uuid=uuid,split=spec['split'],family=spec['family'],
            shape=report['shape'],identity_fingerprint=identity['fingerprint'],
            work_flop=report['work_flop'],data_byte=report['data_byte'],data_semantics=report['data_semantics'],
            latency_s=statistics.median(b['elapsed_s']/b['repeats'] for b in report['batches'] if b['repeats']==128),
            frequency_hz=statistics.mean(s['clock_mhz'] for s in values)*1e6,
            voltage_v=statistics.mean(s['voltage']['reported_core_voltage_v'] for s in values),
            temperature_c=statistics.mean(s['temperature_c'] for s in values),
            memory_mhz=energy['clock_ranges'][uuid]['memory_mhz'],
            integral_basis=f['integral_basis'],energy_j=energy['by_uuid'][uuid]['energy_j'],
            energy_relative_budget=energy['by_uuid'][uuid]['relative_uncertainty'],
            feature_ranges=f['ranges'],same_path_verified=False,
            timing_frequency_simultaneously_observed=False,
            power_scope='homogeneous_pair_host_envelope_not_exact_operator_energy')
        rows.append(row)
    return rows


async def collect(a, identity, devices, out, result):
    controller=telemetry.DualClocks(devices,apply=True)
    try:
        async with service_client.BenchClient(identity['base_url'],service.model_id(identity)) as client:
            baseline=await service.load_frames(client,devices,a.seed,out.name+'baseline',out/'baseline.json')
            try:
                for index,job in enumerate(result['protocol']):
                    result['clock_writes']=True;controller.set(job['requested_mhz'])
                    await service.load_frames(client,devices,a.seed+index,out.name+str(index),out/f'precondition-{index}.json')
                    tag=f"{index:03d}-{job['requested_mhz']}-{job['case']['family']}"
                    service.emit(dict(status='collecting_resource',index=index+1,total=len(result['protocol']),window=tag))
                    rows=await one(identity,client,out/tag,job['case'],a)
                    for r in rows:r['requested_mhz']=job['requested_mhz']
                    result['rows'].extend(rows);write_new(out/(tag+'.json'),dict(rows=rows))
                    service.emit(dict(status='resource_energy_gate_passed',window=tag,
                        relative_budgets={r['uuid']:r['energy_relative_budget'] for r in rows}))
            finally:
                await service.restore_checked(controller,client,devices,baseline,a.seed+90000,out.name+'restore',out/'restored.json')
    finally:
        try:
            if controller.dirty:controller.reset()
        finally:result['restoration']=dict(controller.restore_result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['check','pilot','train','holdout'])
    for k in ('run','telemetry','out'):p.add_argument('--'+k,required=True)
    p.add_argument('--base-url',default=service_client.BASE_URL)
    p.add_argument('--windows-python',default='/mnt/c/ProgramData/Anaconda3/python.exe')
    p.add_argument('--families',nargs='+',choices=FAMILIES,default=['linear','add','mul','copy'])
    p.add_argument('--model');p.add_argument('--seed',type=int,default=211100)
    p.add_argument('--window-seconds',type=float,default=30)
    p.add_argument('--apply',action='store_true');p.add_argument('--ack-default-auto',action='store_true')
    a=p.parse_args()
    window_seconds(a.window_seconds)
    if a.command!='check' and not(a.apply and a.ack_default_auto):p.error('Collection requires --apply --ack-default-auto')
    if a.command=='holdout' and not a.model:p.error('Freeze training model before holdout: --model required')
    import fcntl
    launch=read(Path(a.run)/'launch.json')
    with (Path(launch['runtime'])/'client.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity=eager.inspect(a.run,a.base_url);out=service.fresh(a.out)
        result=dict(status='failed',clock_writes=False,deployable=False,rows=[],protocol=[],
            requested_window_s=a.window_seconds,collection_role=a.command,
            identity_fingerprint=identity['fingerprint'],commit=eager.git('rev-parse','HEAD'))
        write_new(out/'identity.json',identity);devices=[]
        try:
            if any(r.get('resource_calibration_version')!=2 for r in identity['ranks']):
                raise RuntimeError('NEEDS_NORMAL_RESTART: resource_calibration_version=2 required; no writes')
            config=read(a.telemetry)
            if any(c['backend']!='bridge' for c in config):raise ValueError('Require disconnect-recovering bridge')
            devices=telemetry.open_devices(config)
            probes=[d.probe() for d in devices];write_new(out/'probes.json',probes)
            ncu=shutil.which('ncu')
            metrics=dict(status='unavailable',timing_capture_separate=True)
            if ncu:
                try:
                    query=subprocess.run([ncu,'--query-metrics'],capture_output=True,text=True,timeout=30)
                    (out/'ncu-query-metrics.txt').write_text(query.stdout+query.stderr,encoding='utf-8')
                    metrics.update(status='queried' if query.returncode==0 else 'query_failed',returncode=query.returncode,
                        requested_families=['dram__bytes','lts__t_bytes','sm__inst_executed_pipe_tensor','sm__warps_active'],
                        counters_collected=False)
                except subprocess.TimeoutExpired:metrics['status']='query_timeout'
            write_new(out/'counter-capability.json',metrics)
            common=set(probes[0]['rates_mhz'])&set(probes[1]['rates_mhz'])
            if not {2250,2527}<=common:raise ValueError('Training settings unavailable')
            inside=[f for f in common if 2250<f<2527]
            if not inside:raise ValueError('No intermediate holdout frequency')
            middle=min(inside,key=lambda f:(abs(f-2388.5),f));result['holdout_requested_mhz']=middle
            if a.model:
                model=read(a.model)
                if model['identity_fingerprint']!=identity['fingerprint']:raise ValueError('Frozen model identity changed')
                result['frozen_model_sha256']=digest(model)
            for repeat in range(3):
                for rate in ([middle] if a.command=='holdout' else [2250,2527] if repeat%2==0 else [2527,2250]):
                    for family in a.families:
                        for size in ([384] if a.command=='holdout' else [256,512]):
                            result['protocol'].append(dict(requested_mhz=rate,case=dict(family=family,size=size,
                                split='holdout' if a.command=='holdout' else 'train',
                                seed=a.seed+len(result['protocol'])+(100000 if a.command=='holdout' else 0))))
            if a.command=='pilot':
                result['protocol']=result['protocol'][:1]
                result['protocol'][0]['case']['seed']+=200000
            write_new(out/'protocol.json',dict(jobs=result['protocol'],window_seconds=a.window_seconds,
                role=a.command,energy_budget_limit=.03))
            if a.command=='check':result['status']='resource_collection_ready_no_writes'
            else:
                def stop(*_):raise KeyboardInterrupt('Interrupted; attempting pair restoration')
                signal.signal(signal.SIGTERM,stop)
                asyncio.run(collect(a,identity,devices,out,result));result['status']='resource_collection_complete'
        except BaseException as exc:result['error']=service.safe_error(exc)
        finally:
            for d in devices:
                try:d.close()
                except Exception as exc:result['status']='failed';result['close_error']=service.safe_error(exc)
            write_new(out/'collection.json',result)
        service.emit({k:v for k,v in result.items() if k not in ('rows','protocol')});print('Saved:',out)
        if result['status']=='failed':raise SystemExit(1)

if __name__=='__main__':main()
