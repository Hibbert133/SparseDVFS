"""Windows-clock envelope energy with WSL 512/64 clients; no clock writes.

The local Windows window contains ready/finish coordination and idle energy.
It is not silently substituted for the earlier exact WSL request window.
"""
import argparse
import asyncio
from pathlib import Path
import statistics
from types import SimpleNamespace

from . import eager, host_load_check as check, service_model as service
from . import service_telemetry as telemetry
from .service_core import read, write_new
from .observe_core import GPU_UUIDS


def local_energy(data, memory_reference=None):
    if data.get('errors') or data.get('clock_domain') != 'Windows_perf_counter_ns':
        raise ValueError('Require error-free local Windows sampling')
    start, end = data.get('host_window_start_ns'), data.get('host_window_end_ns')
    if type(start) is not int or type(end) is not int or start >= end:
        raise ValueError('Missing host envelope boundary markers; old diagnostic cannot be relabeled')
    frames = []
    for frame in data['frames']:
        if len(frame) != 2 or {r['uuid'] for r in frame} != set(GPU_UUIDS):
            raise ValueError('Need both UUIDs in every host frame')
        converted = []
        for raw in frame:
            if raw.get('synthetic') is not False or raw.get('timestamp_clock', {}).get('name') != 'perf_counter':
                raise ValueError('Wrong host clock or synthetic sample')
            resolution = raw['timestamp_clock']['resolution_s']*1e9
            if resolution < 0: raise ValueError('Negative clock resolution')
            times = raw['field_query_times_ns']
            mapped = {}
            for name, t in times.items():
                if not raw['timestamp_ns'] <= t['start'] <= t['end']:
                    raise ValueError('Invalid local query interval')
                mapped[name] = dict(t_ns=(t['start']+t['end'])/2,
                    uncertainty_ns=(t['end']-t['start'])/2+resolution)
            converted.append(dict(uuid=raw['uuid'], raw=raw, mapped=mapped))
        frames.append(converted)
    # Same 350 ms gap and 3% temporal/quantization budget as previous path.
    sources = {u:'counter' for u in GPU_UUIDS}
    result = telemetry.energy_window(frames, start, end, sources=sources, interval_s=.1)
    result['quality_errors'].extend(telemetry.clock_errors(frames, memory_reference=memory_reference))
    clocks = {}
    for uuid in GPU_UUIDS:
        rows = [s['raw'] for f in frames for s in f if s['uuid'] == uuid]
        memory = sorted({r['memory_clock_mhz'] for r in rows if r.get('memory_clock_mhz') is not None})
        core = [r['clock_mhz'] for r in rows if r.get('clock_mhz') is not None]
        if len(memory) != 1: result['quality_errors'].append(uuid+': memory frequency not constant')
        clocks[uuid] = dict(core_range_mhz=[min(core), max(core)] if core else None,
                           memory_mhz=memory[0] if len(memory) == 1 else None)
    result.update(host_window_s=(end-start)/1e9, clock_ranges=clocks,
        scope='two_GPU_board_host_envelope_including_coordination_not_exact_WSL_request_interval')
    return result


def run(args):
    import fcntl
    saved = read(args.identity)
    with (Path(saved['runtime'])/'client.lock').open('a') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = eager.checked(args.identity)
        out = service.fresh(args.out)
        result = dict(status='failed', identity_fingerprint=identity['fingerprint'],
            commit=eager.git('rev-parse', 'HEAD'), clock_writes=False,
            clock_policy='current_policy_not_verified_AUTO',
            scope='host_envelope_reference_not_exact_WSL_window_or_DVFS_validation',
            energy_sources={u:'counter' for u in GPU_UUIDS},
            energy_window_aligned_to_WSL=False, equation_1_validated=False,
            equation_2_validated=False, dvfs_validated=False, windows=[])
        names = ['pilot']+[f'{part}-{i}' for part in ('reference', 'holdout') for i in range(1, 4)]
        memory, seen = None, set()
        try:
            write_new(out/'protocol.json', dict(requests=20, input=512, output=64, concurrency=1,
                seed=args.seed, windows=names, interval_s=.1, gap_limit_s=.35, uncertainty_limit=.03,
                window='Windows_QPC_ready_marker_to_finish_marker',
                includes_coordination=True, repeats_are_not_model_validation=True))
            for i, name in enumerate(names):
                folder = out/name
                folder.mkdir()
                row = dict(status='failed', replies=[])
                try:
                    current = SimpleNamespace(identity=args.identity, windows_python=args.windows_python,
                                              seed=args.seed+20*i, verify_metrics=True)
                    asyncio.run(check.workload(current, folder, row))
                    if row['status'] != 'loaded_host_sampling_passed' or row.get('cache_status') != 'verified_miss':
                        raise RuntimeError('Host sampling or request/cache gate failed')
                    requests = read(folder/'requests.json')
                    hashes = {r['prompt_hash'] for r in requests}
                    if len(hashes) != 20 or hashes & seen:
                        raise ValueError('Request sets are not independent')
                    seen.update(hashes)
                    energy = local_energy(read(folder/'windows-samples.json'), memory)
                    row['energy'] = energy
                    if energy['quality_errors']:
                        raise RuntimeError('Energy gate failed: '+str(energy['quality_errors']))
                    if len(row['replies']) != 20: raise ValueError('Incomplete workload')
                    memory = {u:r['memory_mhz'] for u,r in energy['clock_ranges'].items()}
                    summary = dict(name=name, status='valid_host_envelope', requests=20,
                        host_window_s=energy['host_window_s'],
                        wsl_window_s=(row['wsl_load_end_ns']-row['wsl_load_begin_ns'])/1e9,
                        mean_request_s=statistics.mean(r['latency_s'] for r in row['replies']),
                        ttft_s=statistics.mean(r['ttft_s'] for r in row['replies']),
                        tpot_s=statistics.mean(r['tpot_s'] for r in row['replies']),
                        host_envelope_energy_j=energy['energy_j'],
                        energy_by_uuid_j={u:v['energy_j'] for u,v in energy['by_uuid'].items()},
                        energy_relative_uncertainty=energy['relative_uncertainty'],
                        clock_ranges=energy['clock_ranges'])
                    result['windows'].append(summary)
                    write_new(folder/'summary.json', summary)
                    service.emit(summary)
                except BaseException as exc:
                    row.update(status='failed', error=service.safe_error(exc))
                    raise
                finally:
                    write_new(folder/'window.json', row)
            result['status'] = 'host_envelope_reference_complete'
        except BaseException as exc:
            result['error'] = service.safe_error(exc)
            raise
        finally:
            write_new(out/'reference.json', result)
            lines = ['Host envelope includes coordination; not exact WSL request energy or DVFS proof.',
                     'status: '+result['status'],
                     'window | mean request s | TTFT s | TPOT s | host s | WSL s | two-board envelope J | budget']
            for w in result['windows']:
                lines.append(' | '.join(str(w[k]) for k in ('name', 'mean_request_s', 'ttft_s',
                    'tpot_s', 'host_window_s', 'wsl_window_s', 'host_envelope_energy_j', 'energy_relative_uncertainty')))
            if result.get('error'): lines.append(result['error'])
            (out/'REPORT.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--identity', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=1110000)
    p.add_argument('--windows-python', default='/mnt/c/ProgramData/Anaconda3/python.exe')
    try:
        result = run(p.parse_args())
        service.emit(dict(status=result['status'], windows=len(result['windows']), dvfs_validated=False))
    except Exception as exc:
        service.emit(dict(status='failed', error=service.safe_error(exc)))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
