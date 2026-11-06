"""One WSL command coordinates Windows-local telemetry with WSL request load.

This is a sampling-path diagnostic, not an energy or performance certificate.
Host mode supports the existing Windows Python 3.8 without vLLM dependencies.
"""
import argparse
import json
import os
from pathlib import Path
import threading
import time


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def summarize(frames, uuids, voltage=False):
    rows = []
    for uuid in uuids:
        samples = [s for frame in frames for s in frame if s['uuid'] == uuid]
        stamps = [s['timestamp_ns'] for s in samples]
        gaps = [(b-a)/1e6 for a, b in zip(stamps, stamps[1:])]
        if len(stamps) < 4 or any(g <= 0 for g in gaps):
            raise ValueError('Insufficient or nonmonotonic host samples')
        if any(s.get('synthetic') is not False or
               s.get('timestamp_clock', {}).get('name') != 'perf_counter' for s in samples):
            raise ValueError('Require real NVML and high-resolution host clock')
        rows.append(dict(uuid=uuid, samples=len(samples), gaps_over_350ms=sum(g > 350 for g in gaps),
                         largest_gaps_ms=sorted(gaps, reverse=True)[:3]))
        if voltage:
            from .nvapi_voltage import summarize_voltage
            rows[-1]['voltage'] = summarize_voltage(samples)
    return rows


def host(args):
    if os.name != 'nt': raise ValueError('Host sampler must run on Windows')
    from concurrent.futures import ThreadPoolExecutor
    from .backends import NVML
    from .observe_core import GPU_UUIDS
    out = Path(args.out)
    devices, frames, errors = [], [], []
    ready, stop = threading.Event(), threading.Event()
    thread = None
    window_start, window_end = None, None
    voltage_reader = None
    def frame(pool):
        readings = list(pool.map(lambda d: d.telemetry(), devices))
        frames.append(readings)  # Preserve partial/error evidence, including NVAPI failure.
        if voltage_reader is not None:
            for reading in readings:
                reading['voltage'] = voltage_reader.read(reading['uuid'])
                v = reading['voltage']
                if v['status'] != 0 or not v['reported_core_voltage_v']:
                    raise RuntimeError('NVAPI voltage failed for '+reading['uuid']+': '+str(v['status']))
    def sample():
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                deadline = time.perf_counter()+900
                while True:
                    begin = time.perf_counter()
                    frame(pool)
                    ready.set()
                    if stop.is_set():
                        # Stop can arrive during an in-flight query. Take a new
                        # complete frame AFTER observing stop to bracket the end.
                        frame(pool)
                        break
                    if time.perf_counter() >= deadline:
                        raise TimeoutError('Host sampling watchdog expired')
                    stop.wait(max(0, .1-(time.perf_counter()-begin)))
        except BaseException as exc:
            errors.append(type(exc).__name__+': '+str(exc)); ready.set()
    try:
        for uuid in GPU_UUIDS: devices.append(NVML(uuid))
        if getattr(args, 'voltage', False):
            from .nvapi_voltage import VoltageReader
            voltage_reader = VoltageReader(GPU_UUIDS)
            save(out/'voltage-bindings.json', voltage_reader.bindings)
        thread = threading.Thread(target=sample, daemon=True)
        thread.start()
        if not ready.wait(15) or errors:
            raise RuntimeError('Host sampler failed to become ready: '+str(errors))
        window_start = time.perf_counter_ns()
        print(json.dumps(dict(status='host_ready', pid=os.getpid(),
                              host_window_start_ns=window_start)), flush=True)
        if input() != 'finish': raise ValueError('Expected finish command')
        window_end = time.perf_counter_ns()
    finally:
        stop.set()
        if thread: thread.join(timeout=15)
        if thread and thread.is_alive(): errors.append('Host sampler did not stop')
        if not thread or not thread.is_alive():
            for device in devices:
                try: device.close()
                except Exception as exc: errors.append(str(exc))
            save(out/'windows-samples.json', dict(frames=frames, errors=errors,
                clock_domain='Windows_perf_counter_ns', energy_window_aligned=False,
                host_window_start_ns=window_start, host_window_end_ns=window_end,
                host_window_scope='encloses_WSL_load_includes_ready_and_finish_coordination'))
        else:
            save(out/'windows-sampling-failure.json', dict(errors=errors))
    rows = summarize(frames, GPU_UUIDS, getattr(args, 'voltage', False))
    result = dict(status='host_sampling_complete' if not errors else 'failed',
                  errors=errors, by_uuid=rows, energy_window_aligned=False)
    save(out/'windows-summary.json', result)
    print(json.dumps(result), flush=True)
    if errors: raise RuntimeError('Host sampling errors')


async def workload(args, out, result):
    import asyncio
    import subprocess
    from . import eager, service_client as client_api, service_model as service
    identity = eager.checked(args.identity)
    result['identity_fingerprint'] = identity['fingerprint']
    root = Path(__file__).resolve().parents[1]
    winpath = lambda p: subprocess.check_output(['wslpath', '-w', str(p)], text=True).strip()
    env = dict(os.environ, PYTHONPATH=winpath(root))
    env.pop('PYTHONHOME', None)
    host_process = None
    with (out/'windows-stderr.log').open('wb') as stderr:
        try:
            async with client_api.BenchClient(identity['base_url'], service.model_id(identity)) as client:
                seed = getattr(args, 'seed', 1109100)
                await service.warmup(client, 512, seed+1000000, out.name+'-warm')
                requests = service.prepared(client, 512, seed, 20, out.name)
                isolation = await eager.isolate_observation_cache(identity, requests[0])
                if isolation['method'] == 'unique_request_cache_salt':
                    import secrets
                    for request in requests: request['cache_salt'] = secrets.token_hex(32)
                save(out/'requests.json', requests)
                if getattr(args, 'verify_metrics', False):
                    result['metrics_before'] = await asyncio.to_thread(service.metrics, identity)
                voltage_flags = ['--voltage'] if getattr(args, 'voltage', False) else []
                host_process = await asyncio.create_subprocess_exec(args.windows_python, '-B', '-m',
                    'rtx5090.host_load_check', '--host', '--out', winpath(out),
                    *voltage_flags,
                    stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=stderr, env=env)
                ready = json.loads(await asyncio.wait_for(host_process.stdout.readline(), 30))
                if ready.get('status') != 'host_ready': raise RuntimeError('Windows sampler not ready')
                result['host_ready'] = ready
                service.emit(dict(status='host_ready_starting_20_WSL_requests'))
                result['wsl_load_begin_ns'] = time.monotonic_ns()
                for request in requests:
                    result['replies'].append(await client.run(request))
                result['wsl_load_end_ns'] = time.monotonic_ns()
        finally:
            if host_process is not None:
                try:
                    if host_process.returncode is None:
                        host_process.stdin.write(b'finish\n')
                        await host_process.stdin.drain()
                        host_process.stdin.close()
                    result['host_stdout_tail'] = (await asyncio.wait_for(host_process.stdout.read(), 30)).decode(errors='replace')
                    result['host_exit_code'] = await asyncio.wait_for(host_process.wait(), 5)
                except Exception as exc:
                    result['host_shutdown_error'] = service.safe_error(exc)
                    if host_process.returncode is None:
                        host_process.kill()
                        await host_process.wait()
    eager.checked(args.identity)
    if getattr(args, 'verify_metrics', False):
        from .service_core import cache_miss_status
        result['metrics_after'] = await asyncio.to_thread(service.metrics, identity)
        result['cache_status'], result['cache_errors'] = cache_miss_status(
            result['metrics_before'], result['metrics_after'], 20, service.prefix_policy(identity))
    host_report = json.loads((out/'windows-summary.json').read_text(encoding='utf-8'))
    result['by_uuid'] = host_report['by_uuid']
    if result.get('host_exit_code') != 0 or host_report['status'] != 'host_sampling_complete':
        raise RuntimeError('Windows sampler failed; inspect windows-stderr.log')
    result['status'] = ('loaded_host_sampling_passed' if all(
        row['gaps_over_350ms'] == 0 for row in result['by_uuid']) else 'loaded_host_sampling_gap')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', action='store_true')
    p.add_argument('--voltage', action='store_true', help='Require Windows-local read-only NVAPI voltage')
    p.add_argument('--identity')
    p.add_argument('--out', required=True)
    p.add_argument('--windows-python', default='/mnt/c/ProgramData/Anaconda3/python.exe')
    args = p.parse_args()
    if args.host:
        host(args)
        return
    import asyncio
    import fcntl
    from . import eager, service_model as service
    from .service_core import read
    if not args.identity: p.error('--identity is required in WSL mode')
    identity = read(args.identity)
    with (Path(identity['runtime'])/'client.lock').open('a') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out = service.fresh(args.out)
        result = dict(status='failed', clock_writes=False, energy_window_aligned=False,
                      energy_validated=False, voltage_requested=args.voltage,
                      equation_2_parameters_validated=False,
                      commit=eager.git('rev-parse', 'HEAD'), replies=[])
        try:
            asyncio.run(workload(args, out, result))
        except BaseException as exc:
            result['error'] = service.safe_error(exc)
            raise
        finally:
            save(out/'diagnostic.json', result)
            service.emit({k:v for k,v in result.items() if k not in ('replies', 'host_stdout_tail')})


if __name__ == '__main__':
    main()
