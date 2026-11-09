"""Observed-band transition diagnostic. Not exact locks or hardware switch certification."""
import argparse
import asyncio
import signal
import time
from pathlib import Path
from . import eager, service_model as service, service_client, service_telemetry as telemetry
from .observe_core import GPU_UUIDS,digest
from .service_core import read,write_new


def bands_from(reference,identity):
    manifest=read(reference/'collection.json')
    if manifest['status']!='complete' or manifest['identity_fingerprint']!=identity or not manifest['restoration']['verified']:
        raise ValueError('Require same-service completed/restored two-condition reference')
    bands={}
    for rate in (2250,2527):
        summary=read(reference/('requested-'+str(rate))/'summary.json')
        if summary['status']!='valid_host_envelope': raise ValueError('Invalid reference window')
        bands[str(rate)]={u:[summary['actual_readbacks'][u]['p10_mhz']-15,
                            summary['actual_readbacks'][u]['p90_mhz']+15] for u in GPU_UUIDS}
    if any(bands['2250'][u][1]>=bands['2527'][u][0] for u in GPU_UUIDS):
        raise ValueError('Observed frequency bands overlap')
    return bands


def in_band(samples,bands):
    return (len(samples)==2 and {s['uuid'] for s in samples}==set(GPU_UUIDS) and all(
        (s['raw'].get('utilization_pct') or 0)>0 and not s['raw'].get('errors') and
        bands[s['uuid']][0]<=s['raw'].get('clock_mhz',-1)<=bands[s['uuid']][1] for s in samples))


async def probe(args,identity,out,result,bands):
    config=read(args.telemetry)
    if any(c['backend']!='bridge' for c in config): raise ValueError('Require bridges with EOF recovery')
    devices=telemetry.open_devices(config)
    controller=telemetry.DualClocks(devices,apply=True)
    try:
        async with service_client.BenchClient(identity['base_url'],service.model_id(identity)) as client:
            baseline=await service.load_frames(client,devices,args.seed,'transition-baseline-'+out.name,out/'baseline.json')
            async def load():
                for request in service.prepared(client,512,args.seed+100,20,'transition-load-'+out.name):
                    result['replies'].append(await client.run(request))
            task=asyncio.create_task(load())
            try:
                previous=None
                for i,rate in enumerate((2250,2527,2250,2527,2250)):
                    row=dict(from_requested_mhz=previous,to_requested_mhz=rate,loaded_samples=[],
                             confirmed=False,begin_ns=time.monotonic_ns())
                    try:
                        controller.set(rate)
                        row['api_return_ns']=time.monotonic_ns()
                        consecutive=0
                        deadline=time.monotonic()+10
                        while time.monotonic()<deadline:
                            if task.done():
                                await task
                                raise RuntimeError('Diagnostic load ended before transitions completed')
                            samples=await asyncio.gather(*(asyncio.to_thread(telemetry.sample_device,d) for d in devices))
                            row['loaded_samples'].append(samples)
                            consecutive=consecutive+1 if in_band(samples,bands[str(rate)]) else 0
                            if consecutive>=3:
                                row.update(confirmed=True,confirmed_ns=max(s['end_ns'] for s in samples))
                                row['observation_interval_ns']=[row['begin_ns'],row['confirmed_ns']]
                                row['observed_confirmation_s']=(row['confirmed_ns']-row['begin_ns'])/1e9
                                break
                            await asyncio.sleep(.02)
                        if not row['confirmed']: raise RuntimeError('Loaded observed-band confirmation timed out; no retry')
                        result['transitions'].append(row)
                        service.emit({k:v for k,v in row.items() if k!='loaded_samples'})
                        previous=rate
                    finally:
                        write_new(out/('transition-%d.json'%i),row)
                await task
            finally:
                # Reset promptly even if the load/client failed. Only then finish
                # any still-running finite load and verify loaded restoration.
                if controller.restore_result['required']: controller.reset()
                try:
                    if not task.done(): await task
                    else: task.result()
                finally:
                    if controller.restore_result['required']:
                        await service.restore_checked(controller,client,devices,baseline,args.seed+900,
                            'transition-restored-'+out.name,out/'restored.json')
            result['status']='observed_band_probe_complete'
            times=[r['observed_confirmation_s'] for r in result['transitions'][1:]]
            result['max_observed_confirmation_s']=max(times)
            result['five_times_observed_confirmation_s']=5*max(times)
    finally:
        result['restoration']=dict(controller.restore_result)
        for d in devices:
            try: d.close()
            except Exception as exc:
                result.update(status='failed',close_error=service.safe_error(exc))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('identity','reference','telemetry','out'): p.add_argument('--'+n,required=True)
    p.add_argument('--apply',action='store_true'); p.add_argument('--ack-default-auto',action='store_true')
    p.add_argument('--seed',type=int,default=13000100)
    a=p.parse_args()
    if a.apply and not a.ack_default_auto: p.error('Writes require --ack-default-auto')
    identity=eager.checked(a.identity)
    bands=bands_from(Path(a.reference),identity['fingerprint'])
    out=service.fresh(a.out)
    config=read(a.telemetry)
    result=dict(status='failed',identity_fingerprint=identity['fingerprint'],bands=bands,
                reference_manifest_sha256=digest(read(Path(a.reference)/'collection.json')),
                telemetry_config_sha256=digest(config),clock_writes=a.apply,transitions=[],replies=[],
                exact_lock_verified=False,conservative_bound_verified=False,
                fine_boundary_precision_verified=False,lookahead_authorized=False,
                note='20ms requested polling plus bridge/query/host scheduling; observed band entry is not exact hardware settling')
    import fcntl
    with (Path(identity['runtime'])/'client.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def stop(signum,frame): raise KeyboardInterrupt('Stop requested; restore both clocks')
        signal.signal(signal.SIGTERM,stop)
        try:
            if a.apply:
                asyncio.run(probe(a,identity,out,result,bands))
            else:
                devices=telemetry.open_devices(config)
                try: result.update(status='readonly_ready',readbacks=[d.telemetry() for d in devices])
                finally:
                    for d in devices: d.close()
        except BaseException as exc:
            result.update(status='failed',error=service.safe_error(exc))
        finally: write_new(out/'probe.json',result)
    service.emit({k:v for k,v in result.items() if k not in ('readbacks','transitions','replies')})
    if result['status']=='failed': raise SystemExit(1)


if __name__=='__main__': main()
