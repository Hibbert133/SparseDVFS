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


def summarize_operator_cost(observation, identity, directory):
    import re
    if (observation.get('status') != 'complete' or
            observation['identity_fingerprint'] != identity['fingerprint'] or
            not observation['coverage']['passed'] or not observation['cache_isolation']['miss_verified'] or
            observation['reply']['usage'].get('prompt_tokens') != 512 or
            observation['reply']['usage'].get('completion_tokens') != 64):
        raise ValueError('Operator collection requires verified same-session 512/64 cache miss')
    result = {}
    for rank in (0, 1):
        report = read(Path(directory)/('rank-%d-operator-cost.json' % rank))
        if (report['status'] != 'operator_event_envelopes_collected' or report['rank'] != rank or
                report['uuid'] != GPU_UUIDS[rank] or report['identity_sha256'] != digest(identity['ranks'][rank])):
            raise ValueError('Operator cost rank/identity mismatch')
        covered = set()
        indices = {}
        ordinals = {}
        for row in report['rows']:
            context = row['context']
            if row['ordinal'] != ordinals.get(context, 0):
                raise ValueError('Operator ordinals missing/duplicated/reordered')
            ordinals[context] = row['ordinal']+1
            match = re.search(r'(?:^|\.)layers\.(\d+)(?:\.|$)', row['parent'])
            if not match:
                continue  # Explicit transformer-layer subgraph, not full request costs.
            layer = int(match.group(1)); context = row['context']
            if context not in ('prefill', 'decode.0') or not 0 <= layer < 64 or row.get('rank') != rank:
                raise ValueError('Unexpected operator context/layer/rank')
            if row['stream_changed']:
                raise ValueError('Current stream changed across operator; cannot use event envelope')
            value = row['stream_event_ms']/1000
            if not math.isfinite(value) or value < 0:
                raise ValueError('Invalid operator event duration')
            covered.add((context, layer))
            index = indices.get(context, 0); indices[context] = index+1
            key = '%d:%s/%06d' % (rank, context, index)
            contract = dict(overload=row['overload'], inputs=row['inputs'], outputs=row['outputs'])
            result[key] = dict(uuid=GPU_UUIDS[rank], parent=row['parent'], shape=contract,
                batch_means_s=[value], median_s=value, host_dispatch_s=row['host_dispatch_s'],
                kernel_names_verified=False, zero_duration_unresolved=value == 0,
                source_ordinal=row['ordinal'], layer=layer,
                timing_scope='single_execution_current_stream_envelope_not_kernel_time')
        if covered != {(c, l) for c in ('prefill', 'decode.0') for l in range(64)}:
            raise ValueError('Missing transformer layer/context in operator capture')
    return result


def summarize_units(observation, identity, directory):
    from .eager_core import UNITS
    if (observation.get('status') != 'complete' or
            observation['identity_fingerprint'] != identity['fingerprint'] or
            not observation['coverage']['passed'] or
            not observation['cache_isolation'].get('miss_verified')):
        raise ValueError('Incomplete module coverage or identity/cache mismatch')
    usage = observation['reply']['usage']
    if usage.get('prompt_tokens') != 512 or usage.get('completion_tokens') != 64:
        raise ValueError('Require actual 512/64')
    contexts = ['prefill'] + ['decode.%d' % i for i in range(63)]
    expected = {(c, u) for c in contexts for u in UNITS}
    result = {}
    for rank in (0, 1):
        execution = read(Path(directory)/('rank-%d-execution.json' % rank))
        if (execution['status'] != 'complete' or execution['mode'] != 'units' or
                execution['rank'] != rank or execution['identity_sha256'] != digest(identity['ranks'][rank])):
            raise ValueError('Module execution identity mismatch')
        seen = set()
        for row in execution['rows']:
            pair = (row['context'], row['unit'])
            if pair in seen or pair not in expected:
                raise ValueError('Duplicate/unknown module context')
            seen.add(pair)
            value = row['stream_event_ms'] / 1000
            if not math.isfinite(value) or value < 0:
                raise ValueError('Invalid CUDA event duration')
            key = '%s:%s/%s' % (rank, pair[0], pair[1])
            result[key] = dict(uuid=GPU_UUIDS[rank], parent=row['unit'], shape=None,
                               batch_means_s=[value], median_s=value,
                               kernel_names_verified=False,
                               timing_scope='current_stream_module_events_not_all_stream_critical_path',
                               zero_duration_unresolved=value == 0)
        if seen != expected:
            raise ValueError('Missing module contexts')
    return result


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
    if any(set(w['operators']) != set(windows[0]['operators']) for w in windows):
        raise ValueError('Operator occurrence set changed; no missing rows dropped')
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
                           latency_ratio_2250_over_2527=(cells['2250']['median_s']/cells['2527']['median_s']
                               if cells['2527']['median_s'] > 0 else None)))
    return output


async def transaction(controller, work, restore):
    try:
        return await work()
    finally:
        if controller.restore_result['required']:
            # Physical restoration must be attempted even after observation failure.
            await restore()


def module_input(result):
    from .eager_core import UNITS
    contexts = ['prefill'] + ['decode.%d' % i for i in range(63)]
    nodes = [c+'/'+u for c in contexts for u in UNITS]
    lookup = {row['key']: row for row in result['comparison']}
    table = {}
    for node in nodes:
        table[node] = {}
        for rate in ('2250', '2527'):
            times = [lookup['%d:%s' % (rank, node)]['settings'][rate]['median_s'] for rank in (0, 1)]
            table[node][rate] = dict(pair_latency_s=max(times) if all(t>0 for t in times) else None,
                                    pair_energy_j=None, rank_event_medians_s=times,
                                    timing_scope='max_rank_module_event_diagnostic_not_block_prediction')
    return dict(schema='pair-partition-input-v1', synthetic=False,
                identity_fingerprint=result['identity_fingerprint'],
                scope='full_request_ordered_module_projection_not_kernel_DAG',
                nodes=nodes, edges=list(zip(nodes, nodes[1:])), candidates=table,
                switch_s=None, request_budget=None, initial_mhz=None,
                costs_validated=False, deployable=False,
                note='Energy, request overhead/reference and certified switching missing; do not fill from window energy allocation')


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
                                matrix_replay=a.scope == 'linear', matrix_calibrate=False, trace=False,
                                operator_cost=a.scope == 'operators'))
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
                                      operators=(summarize(observation, identity['fingerprint']) if a.scope == 'linear'
                                                 else summarize_operator_cost(observation, identity, out/tag) if a.scope == 'operators'
                                                 else summarize_units(observation, identity, out/tag)))
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
            result['status'] = 'two_setting_%s_timing_complete' % a.scope
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
    p.add_argument('--scope', choices=('linear', 'modules', 'operators'), default='linear')
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
        if a.scope == 'operators' and any(r.get('operator_cost_version', 0) < 1 for r in identity['ranks']):
            raise ValueError('Resident workers lack operator_cost_version=1; normal restart required BEFORE clock writes')
        if a.scope == 'linear' and any(rank.get('matrix_replay_version', 0) < 1 for rank in identity['ranks']):
            raise ValueError('Resident workers do not support matrix replay; no writes')
        config = read(a.telemetry)
        if any(c['backend'] != 'bridge' for c in config):
            raise ValueError('Require bridges with disconnect recovery')
        out = service.fresh(a.out)
        write_new(out/'identity.json', identity)
        result = dict(status='failed', identity_fingerprint=identity['fingerprint'], windows=[],
                      clock_writes=False, exact_lock_verified=False, deployable=False,
                      energy_measured=False, frequency_scaling_validated=False,
                      scope=('first_MLP_projection_per_phase_repeated_linear_timing_not_whole_blocks' if a.scope == 'linear'
                             else 'transformer_layer_operator_events_prefill_decode0_only' if a.scope == 'operators'
                             else 'full_module_current_stream_event_diagnostics_not_whole_block_critical_path'),
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
        comparisons = result.get('comparison', [])
        if a.scope in ('modules', 'operators'):
            if result['status'] == 'two_setting_modules_timing_complete':
                write_new(out/'module-budget-input.json', module_input(result))
                lines.append('module-budget-input.json contains measured event diagnostics and explicit missing energy/budget fields.')
            lines.append('Full per-context table saved in collection.json: %d cases' % len(comparisons))
            for rank in (0, 1):
                for phase in ('prefill', 'decode'):
                    values = [v['latency_ratio_2250_over_2527'] for v in comparisons
                              if v['key'].startswith('%d:%s' % (rank, phase))]
                    usable = sorted(v for v in values if v is not None)
                    lines.append('rank %d %s: cases=%d zero-high-unresolved=%d median ratio=%s (unweighted diagnostic)' % (
                        rank, phase, len(values), len(values)-len(usable), median(usable) if usable else None))
            comparisons = []
        for row in comparisons:
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
