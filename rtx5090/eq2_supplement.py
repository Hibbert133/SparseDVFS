"""Two requested-clock diagnostic windows, measured actual V/f, no lock certificate."""
import argparse
import asyncio
from pathlib import Path
from statistics import median
from types import SimpleNamespace
from . import eager, service_model as service, service_client, service_telemetry as telemetry
from .host_load_check import workload
from .host_reference import local_energy
from .observe_core import GPU_UUIDS, digest
from .service_core import read, write_new


def changed_readbacks(data, baseline, target, previous=None):
    result={}
    for uuid in GPU_UUIDS:
        before=[s['raw']['clock_mhz'] for f in baseline for s in f if s['uuid']==uuid
                and (s['raw'].get('utilization_pct') or 0)>0]
        active=[s['clock_mhz'] for f in data['frames'] for s in f if s['uuid']==uuid
                and (s.get('utilization_pct') or 0)>0 and s.get('clock_mhz')]
        if len(before)<3 or len(active)<3: raise ValueError('Missing loaded readbacks')
        lo,hi=sorted(active)[int(.1*(len(active)-1))],sorted(active)[int(.9*(len(active)-1))]
        mid=median(active)
        if mid > target+15 or median(before)-mid < 100:
            raise ValueError('Requested setting did not produce a lower loaded frequency on '+uuid)
        if previous and previous[uuid]['median_mhz']-mid < 100:
            raise ValueError('Second condition did not add frequency separation on '+uuid)
        result[uuid]=dict(median_mhz=mid,p10_mhz=lo,p90_mhz=hi,
                          requested_mhz=target, exact_lock_verified=False)
    return result


async def collect(args, identity, out, result):
    config=read(args.telemetry)
    if any(d['backend']!='bridge' for d in config):
        raise ValueError('Require existing dual bridge with disconnect recovery')
    devices=telemetry.open_devices(config)
    controller=telemetry.DualClocks(devices,apply=True)
    baseline=[]
    try:
        async with service_client.BenchClient(identity['base_url'], service.model_id(identity)) as client:
            baseline=await service.load_frames(client,devices,args.seed,'eq2-baseline-'+out.name,out/'baseline.json')
        previous=None
        try:
            for i,target in enumerate((2527,2250)):
                name='requested-'+str(target)
                folder=out/name; folder.mkdir()
                row=dict(status='failed',replies=[])
                controller.set(target)
                try:
                    service.emit(dict(status='collecting_requested_clock_diagnostic',requested_mhz=target,
                                      actual_frequency_used_for_model=True,exact_lock_verified=False))
                    current=SimpleNamespace(identity=args.identity,windows_python=args.windows_python,
                                            seed=args.seed+100+i*100,verify_metrics=True,voltage=True)
                    await workload(current,folder,row)
                    if row['status']!='loaded_host_sampling_passed' or row.get('cache_status')!='verified_miss':
                        raise ValueError('Sampling/cache gate failed')
                    data=read(folder/'windows-samples.json')
                    energy=local_energy(data)
                    if energy['quality_errors']: raise ValueError(str(energy['quality_errors']))
                    accepted=changed_readbacks(data,baseline,target,previous)
                    previous=accepted
                    summary=dict(name=name,status='valid_host_envelope',requests=20,
                        host_window_s=energy['host_window_s'], requested_mhz=target,
                        actual_readbacks=accepted, energy_by_uuid_j={u:v['energy_j'] for u,v in energy['by_uuid'].items()},
                        energy_relative_uncertainty=energy['relative_uncertainty'])
                    write_new(folder/'summary.json',summary)
                    result['windows'].append(name)
                    service.emit(summary)
                except BaseException as exc:
                    row.update(status='failed',error=service.safe_error(exc)); raise
                finally:
                    write_new(folder/'window.json',row)
        finally:
            if controller.restore_result['required']:
                # Reset first even if a new HTTP connection cannot be opened.
                controller.reset()
                async with service_client.BenchClient(identity['base_url'],service.model_id(identity)) as client:
                    await service.restore_checked(controller,client,devices,baseline,args.seed+900,
                                                  'eq2-restored-'+out.name,out/'restored.json')
                result['restoration']=dict(controller.restore_result)
                service.emit(dict(status='dual_card_restoration_verified',evidence=result['restoration']))
        eager.checked(args.identity)
        result['status']='complete'
    finally:
        result['restoration']=dict(controller.restore_result)
        for device in devices:
            try: device.close()
            except Exception as exc:
                result['status']='failed'; result['close_error']=service.safe_error(exc)


def main():
    import signal
    def stop(signum, frame): raise KeyboardInterrupt('Stop requested; restore both clocks')
    signal.signal(signal.SIGTERM,stop)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--identity',required=True); p.add_argument('--telemetry',required=True)
    p.add_argument('--out',required=True); p.add_argument('--seed',type=int,default=11062700)
    p.add_argument('--windows-python',default='/mnt/c/ProgramData/Anaconda3/python.exe')
    p.add_argument('--apply',action='store_true'); p.add_argument('--ack-default-auto',action='store_true')
    p.add_argument('--frozen',help='Bind holdout acquisition to an already frozen candidate file')
    args=p.parse_args()
    if not args.apply or not args.ack_default_auto: p.error('Require --apply and --ack-default-auto for restoration to default policy')
    import fcntl
    identity=eager.checked(args.identity)
    frozen=read(args.frozen) if args.frozen else None
    if frozen and frozen['identity_fingerprint']!=identity['fingerprint']:
        raise ValueError('Frozen candidate service identity changed')
    with (Path(identity['runtime'])/'client.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        out=service.fresh(args.out)
        result=dict(status='failed',identity_fingerprint=identity['fingerprint'],windows=[],
                    frozen_sha256=digest(frozen) if frozen else None,
                    commit=eager.git('rev-parse','HEAD'),telemetry_config_sha256=digest(read(args.telemetry)),
                    clock_writes=True, exact_lock_verified=False, dvfs_validated=False,
                    restoration=dict(required=False,verified=False))
        write_new(out/'protocol.json',dict(requested_mhz=[2527,2250],requests=20,input=512,output=64,
                   frequency_label='actual_synchronous_readbacks_not_requested_setpoint',
                   scope='diagnostic_excitation_not_fixed_clock_acceptance',seed=args.seed))
        try: asyncio.run(collect(args,identity,out,result))
        except BaseException as exc:
            result.update(status='failed',error=service.safe_error(exc))
        finally: write_new(out/'collection.json',result)
        service.emit(result)
        if result['status']!='complete': raise SystemExit(1)


if __name__=='__main__': main()
