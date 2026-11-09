"""Two-setting real-tensor linear timing diagnostic using resident eager workers.

Not Eq1 calibration, block power labels, exact-lock acceptance, or a live plan.
"""
import argparse
import asyncio
import math
import signal
from pathlib import Path
from statistics import median
from types import SimpleNamespace

from . import eager, service_model as service, service_client, service_telemetry as telemetry
from .observe_core import GPU_UUIDS, digest
from .service_core import read, write_new


def summarize(observation, identity):
    if observation.get('status') != 'complete' or observation['identity_fingerprint'] != identity:
        raise ValueError('Incomplete or changed worker observation')
    if not observation['cache_isolation'].get('miss_verified') or not observation['coverage']['passed']:
        raise ValueError('Require full cache-miss coverage')
    usage = observation['reply']['usage']
    if usage.get('prompt_tokens') != 512 or usage.get('completion_tokens') != 64:
        raise ValueError('Actual request is not 512/64')
    result = {}
    for rank in observation['matrix_replay']:
        if rank['rank'] not in (0, 1) or rank['uuid'] != GPU_UUIDS[rank['rank']]:
            raise ValueError('Rank/UUID mismatch')
        for row in rank['rows']:
            key = '%s:%s/%s' % (rank['rank'], row['phase'], row['module'])
            if key in result:
                raise ValueError('Duplicate linear case')
            values = [b['mean_call_s'] for b in row['batches'] if b['repeats'] == 32]
            if len(values) != 3 or any(not math.isfinite(x) or x <= 0 for x in values):
                raise ValueError('Require three positive N32 batch means')
            result[key] = dict(uuid=rank['uuid'], parent=row['parent'],
                               shape=[row['M'], row['N'], row['K']],
                               batch_means_s=values, median_s=median(values),
                               kernel_names_verified=row.get('kernel_names_verified', False))
    expected = {'%s:%s/%s' % (r, p, m) for r in (0, 1)
                for p in ('prefill', 'decode') for m in ('gate_up_proj', 'down_proj')}
    if set(result) != expected:
        raise ValueError('Missing/extra dual-rank linear cases')
    return result


def compare(windows):
    if len(windows) != 6:
        raise ValueError('Require all six observations; no partial fitting')
    output = []
    for key in sorted(windows[0]['operators']):
        reference = windows[0]['operators'][key]
        for w in windows:
            op = w['operators'][key]
            if any(op[k] != reference[k] for k in ('uuid', 'parent', 'shape')):
                raise ValueError('Operator identity/shape changed')
        cells = {}
        for rate in (2250, 2527):
            selected = [w for w in windows if w['requested_mhz'] == rate]
            if len(selected) != 3:
                raise ValueError('Require three requests per setting')
            values = [w['operators'][key]['median_s'] for w in selected]
            cells[str(rate)] = dict(request_medians_s=values, median_s=median(values))
        output.append(dict(key=key, settings=cells,
                           latency_ratio_2250_over_2527=cells['2250']['median_s']/cells['2527']['median_s']))
    return output


async def transaction(controller, work, restore):
    try:
        return await work()
    finally:
        if controller.restore_result['required']:
            # Physical restoration must be attempted even after observation failure.
            await restore()


async def collect(a, identity, devices, out, result):
    controller = telemetry.DualClocks(devices, apply=True)
    try:
        async with service_client.BenchClient(identity['base_url'], service.model_id(identity)) as client:
            baseline = await service.load_frames(client, devices, a.seed, out.name+'-baseline', out/'baseline.json')

            async def work():
                for repeat in range(3):
                    for rate in ((2250, 2527) if repeat % 2 == 0 else (2527, 2250)):
                        tag = '%d-%d' % (repeat, rate)
                        result['clock_writes'] = True
                        controller.set(rate)
                        # Explicit preconditioning, not certification of exact requested frequency.
                        await service.load_frames(client, devices, a.seed+repeat*100+rate,
                                                  out.name+tag, out/(tag+'-precondition.json'))
                        sampler = telemetry.Sampler(devices)
                        try:
                            await asyncio.to_thread(sampler.start)
                            observation = await eager.observe(SimpleNamespace(
                                identity=str(out/'identity.json'), out=str(out/tag),
                                seed=a.seed+10000+repeat*10+(rate==2527),
                                matrix_replay=True, matrix_calibrate=False, trace=False))
                        finally:
                            await asyncio.to_thread(sampler.finish)
                            write_new(out/(tag+'-telemetry.json'), dict(frames=sampler.frames, errors=sampler.errors))
                        if sampler.errors:
                            raise RuntimeError('Telemetry failed: '+str(sampler.errors))
                        actual = {}
                        for uuid in GPU_UUIDS:
                            clocks = [s['raw']['clock_mhz'] for frame in sampler.frames for s in frame
                                      if s['uuid']==uuid and (s['raw'].get('utilization_pct') or 0)>0
                                      and not s['raw'].get('errors') and s['raw'].get('clock_mhz') is not None]
                            if not clocks:
                                raise ValueError('Missing loaded readbacks')
                            actual[uuid] = dict(min_mhz=min(clocks), max_mhz=max(clocks), median_mhz=median(clocks))
                        window = dict(requested_mhz=rate, repeat=repeat, actual_readbacks=actual,
                                      observation_sha256=digest(observation),
                                      prompt_hash=observation['reply']['prompt_hash'],
                                      operators=summarize(observation, identity['fingerprint']))
                        result['windows'].append(window)
                        write_new(out/(tag+'-summary.json'), window)
                        service.emit(dict(window=tag, status='timing_diagnostic_saved', actual_readbacks=actual))
                if len({w['prompt_hash'] for w in result['windows']}) != 6:
                    raise ValueError('Repeated measurement prompts')
                result['comparison'] = compare(result['windows'])

            async def restore():
                # observe normally disarms itself; preserve failures and attempt reset regardless.
                try:
                    eager.rank_commands(identity, dict(op='arm', mode='off'))
                except BaseException as exc:
                    result['disarm_error'] = service.safe_error(exc)
                await service.restore_checked(controller, client, devices, baseline, a.seed+90000,
                                              out.name+'-restore', out/'restored.json')

            await transaction(controller, work, restore)
            if result.get('disarm_error'):
                raise RuntimeError('Worker disarm not confirmed')
            result['status'] = 'two_setting_linear_timing_complete'
    finally:
        if controller.dirty:
            try:
                controller.reset()
            except BaseException as exc:
                result['reset_error'] = service.safe_error(exc)
        result['restoration'] = dict(controller.restore_result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'telemetry', 'out'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--base-url', default=service_client.BASE_URL)
    p.add_argument('--seed', type=int, default=16000100)
    p.add_argument('--apply', action='store_true')
    p.add_argument('--ack-default-auto', action='store_true')
    a = p.parse_args()
    if a.apply and not a.ack_default_auto:
        p.error('Real writes require --ack-default-auto')
    # Acquire existing service lease BEFORE inspecting/arming any worker.
    import fcntl
    launch = read(Path(a.run)/'launch.json')
    with (Path(launch['runtime'])/'client.lock').open('a') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = eager.inspect(a.run, a.base_url)
        if any(rank.get('matrix_replay_version', 0) < 1 for rank in identity['ranks']):
            raise ValueError('Resident workers do not support matrix replay; no writes')
        config = read(a.telemetry)
        if any(c['backend'] != 'bridge' for c in config):
            raise ValueError('Require bridges with disconnect recovery')
        out = service.fresh(a.out)
        write_new(out/'identity.json', identity)
        result = dict(status='failed', identity_fingerprint=identity['fingerprint'], windows=[],
                      clock_writes=False, exact_lock_verified=False, deployable=False,
                      energy_measured=False, frequency_scaling_validated=False,
                      scope='first_MLP_projection_per_phase_repeated_linear_timing_not_whole_blocks',
                      telemetry_config_sha256=digest(config), commit=eager.git('rev-parse','HEAD'))
        devices = []
        def stop(signum, frame):
            raise KeyboardInterrupt('Stop requested; restore touched clocks')
        signal.signal(signal.SIGTERM, stop)
        try:
            devices = telemetry.open_devices(config)
            if a.apply:
                asyncio.run(collect(a, identity, devices, out, result))
            else:
                result['status'] = 'readonly_ready'
        except BaseException as exc:
            result['status'] = 'failed'
            result['error'] = service.safe_error(exc)
        finally:
            result['close_errors'] = []
            for device in devices:
                try:
                    device.close()
                except BaseException as exc:
                    result['close_errors'].append(service.safe_error(exc))
                    result['status'] = 'failed'
            write_new(out/'collection.json', result)
        lines = [result['status'], 'Two requested settings; actual readbacks retained. No operator energy or block plan.']
        for row in result.get('comparison', []):
            lines.append('%s | 2250 %.3f us | 2527 %.3f us | ratio %.4f' % (
                row['key'], row['settings']['2250']['median_s']*1e6,
                row['settings']['2527']['median_s']*1e6, row['latency_ratio_2250_over_2527']))
        lines.append('restoration: '+str(result.get('restoration')))
        if result.get('error'): lines.append(result['error'])
        (out/'REPORT.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
        print('\n'.join(lines)); print('Saved:', out)
        if result['status']=='failed': raise SystemExit(1)


if __name__ == '__main__':
    main()
