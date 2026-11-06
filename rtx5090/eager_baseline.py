"""One pilot plus six uninstrumented 512/64 current-clock reference windows.

No clock writes, no model fit, and no certification of AUTO policy or DVFS benefit.
"""
import argparse
import asyncio
import math
import os
from pathlib import Path
import secrets
import statistics

from . import eager, service_model as service, service_client as client_api
from . import service_telemetry as telemetry
from .service_core import read, write_new
from .observe_core import digest
from .telemetry_process import ProcessSampler


def window_plan(seed, count):
    if count < 20:
        raise ValueError('At least 20 requests per window')
    return [('pilot', seed)] + [(f'{split}-{i+1}', seed+(1+j*3+i)*count)
        for j, split in enumerate(('reference', 'holdout')) for i in range(3)]


def compact(row, name):
    return dict(name=name, status=row['status'], requests=row['requests'],
        cache_status=row.get('cache_status'), energy_j=row.get('energy_j'),
        energy_by_uuid_j=row.get('energy_by_uuid_j'), window_s=row.get('window_s'),
        mean_request_s=statistics.mean(r['latency_s'] for r in row['replies']) if row['replies'] else None,
        ttft_s=row.get('ttft_s'), tpot_s=row.get('tpot_s'),
        energy_relative_uncertainty=row.get('energy_relative_uncertainty'),
        temperature_range_c=row.get('temperature_range_c'), clock_ranges=row.get('clock_ranges'),
        quality_errors=row['quality_errors'])


async def collect(args, identity, out, result):
    config = read(args.telemetry)
    result['telemetry_config_sha256'] = digest(config)
    devices = []
    try:
        devices = telemetry.open_devices(config)
        async with client_api.BenchClient(identity['base_url'], service.model_id(identity)) as client:
            result.update(client_source_sha256=client.source_sha256,
                          tokenizer_fingerprint=client.tokenizer_fingerprint)
            await service.stop_profile(identity)
            await service.warmup(client, 512, args.seed+1000000, out.name+'-warm')
            sources, memory = None, None
            seen = set()
            for name, seed in window_plan(args.seed, args.requests):
                # Verify both workers remain idle/off before each measured window.
                eager.checked(args.identity)
                requests = service.prepared(client, 512, seed, args.requests, out.name+'-'+name)
                hashes = {r['prompt_hash'] for r in requests}
                if seen & hashes:
                    raise ValueError('Request lists overlap; cannot claim independent windows')
                seen.update(hashes)
                isolation = await eager.isolate_observation_cache(identity, requests[0])
                if isolation['method'] == 'unique_request_cache_salt':
                    for request in requests:
                        request['cache_salt'] = secrets.token_hex(32)
                # The measured_window metrics gate, NOT the salt, certifies cache misses.
                write_new(out/(name+'-isolation.json'), isolation)
                row = await service.measured_window(client, identity, devices, requests,
                    out, name, sources=sources, interval_s=.1, target=None, memory_reference=memory,
                    sampler_factory=lambda _devices, interval: ProcessSampler(config, interval))
                summary = compact(row, name)
                result['windows'].append(summary)
                write_new(out/(name+'-summary.json'), summary)
                service.emit(summary)
                if row['status'] != 'valid':
                    if name == 'pilot' and row.get('energy_relative_uncertainty') is not None:
                        result['suggested_requests_for_new_run'] = max(args.requests,
                            math.ceil(args.requests*row['energy_relative_uncertainty']/.03*1.25))
                    raise RuntimeError('Invalid window preserved; stopped without retries or dropping samples')
                if name == 'pilot':
                    sources = {u: value['source'] for u, value in row['energy']['by_uuid'].items()}
                    memory = {u: value['memory_mode'] for u, value in row['clock_ranges'].items()}
                    result['frozen_energy_sources'] = sources
                    result['frozen_memory_reference'] = memory
                    write_new(out/'protocol.json', dict(requests=args.requests, input=512, output=64,
                        concurrency=1, interval_s=.1, sources=sources, memory_reference=memory,
                        plan=window_plan(args.seed, args.requests), fixed_after_pilot=True))
            eager.checked(args.identity)
            result['status'] = 'current_clock_reference_complete'
    finally:
        eager.close_devices(devices, result)


def run(args):
    if os.name != 'posix':
        raise ValueError('Run on target WSL')
    import fcntl
    window_plan(args.seed, args.requests)
    saved = read(args.identity)
    with (Path(saved['runtime'])/'client.lock').open('a') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = eager.checked(args.identity)
        out = service.fresh(args.out)
        result = dict(schema='eager-current-reference-v1', status='failed',
            identity_fingerprint=identity['fingerprint'], commit=eager.git('rev-parse', 'HEAD'),
            client_sha256=client_api.sha(__file__), identity_sha256=client_api.sha(args.identity),
            clock_writes=False, clock_policy='current_policy_not_verified_AUTO',
            instrumentation='worker_off_no_profiler_no_detailed_observation',
            telemetry_execution='independent_spawned_process_same_WSL_monotonic_domain',
            energy_scope='two_GPU_boards_not_whole_machine',
            equation_1_validated=False, equation_2_validated=False, dvfs_validated=False,
            comparison_or_model_validation=False, windows=[])
        try:
            asyncio.run(collect(args, identity, out, result))
        except BaseException as exc:
            result.update(status='failed', error=service.safe_error(exc))
            raise
        finally:
            write_new(out/'baseline.json', result)
            lines = ['Current-clock 512/64 reference (not DVFS/Eq1/Eq2 validation)',
                     'status: '+result['status'],
                     'window | requests | mean request s | TTFT s | TPOT s | two-board J | status']
            for w in result['windows']:
                lines.append(' | '.join(str(w.get(k)) for k in
                    ('name', 'requests', 'mean_request_s', 'ttft_s', 'tpot_s', 'energy_j', 'status')))
            if result.get('error'): lines.append(result['error'])
            (out/'REPORT.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('identity', 'telemetry', 'out'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--requests', type=int, default=20)
    p.add_argument('--seed', type=int, default=1106000)
    try:
        result = run(p.parse_args())
        service.emit(dict(status=result['status'], windows=len(result['windows']),
                          clock_writes=False, dvfs_validated=False))
    except Exception as exc:
        service.emit(dict(status='failed', error=service.safe_error(exc)))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
