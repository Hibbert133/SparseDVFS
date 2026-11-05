"""Optimized resident-service tooling. Never creates/restarts a vLLM engine.

Target commands are executed manually by the user; fit/validate/report are CPU-only.
"""
import argparse
import asyncio
from collections import Counter
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import sys
import time
import uuid as uuid_module

from . import service_core as core
from . import service_client as client_api
from . import service_telemetry as telemetry
from .observe_core import digest, GPU_UUIDS


def safe_error(exc):
    value = f'{type(exc).__name__}: {exc}'
    for key in ('SPARSEDVFS_BRIDGE_TOKEN', 'OPENAI_API_KEY'):
        if os.environ.get(key): value = value.replace(os.environ[key], '[REDACTED]')
    return value


def fresh(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def emit(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


def checked_identity(path, require_ready=True):
    old = core.read(path)
    live = client_api.inspect_service(old['base_url'], old.get('launch_dir'))
    if live['fingerprint'] != old['fingerprint']:
        raise ValueError('Live service boot/config differs from inspected identity')
    if require_ready and not live['identity_ready']:
        raise ValueError('Actual dtype/weight loader/service boot/GPU mapping not established; inspect startup log')
    return live


def model_id(identity):
    entries = identity['models']['data']
    if len(entries) != 1: raise ValueError('Require exactly one served model')
    return entries[0]['id']


async def stop_profile(identity):
    if all(identity['profiler_routes'].values()):
        await asyncio.to_thread(client_api.http, identity['base_url'], '/stop_profile', {}, 600)
        if identity.get('launch_dir'):
            log = (Path(identity['launch_dir'])/'startup.log').read_text(encoding='utf-8', errors='replace')
            # vLLM catches worker profiler start/stop exceptions internally, so
            # HTTP 200 alone is insufficient when its log records a failure.
            if re.search(r'Failed to (?:start|stop) profiler', log, re.I):
                raise RuntimeError('Native profiler failure in service log; verify stopped state before measurement')


def metrics(identity):
    return core.metric_snapshot(client_api.http(identity['base_url'], '/metrics', text=True))


def prefix_policy(identity):
    value = identity.get('startup_identity', {}).get('prefix_caching')
    return {'True': True, 'False': False, 'true': True, 'false': False}.get(value)


def prepared(client, length, seed, count, prefix):
    requests = [client.prepare(length, seed+i, f'{prefix}-{i}') for i in range(count)]
    if len({r['prompt_hash'] for r in requests}) != count:
        raise ValueError('Duplicate prompts')
    return requests


async def warmup(client, length, seed, prefix):
    # Separate IDs/prompts from all measured lists. The first request is the ready test.
    seed = int(digest([seed, prefix])[:15], 16)
    for request in prepared(client, length, seed, 2, prefix): await client.run(request)


def log_cursor(identity):
    if not identity.get('launch_dir'): return None
    path = Path(identity['launch_dir'])/'startup.log'
    return path, path.stat().st_size


def new_compilation(cursor):
    if cursor is None: return 'unknown_no_log'
    path, offset = cursor
    with path.open('rb') as stream:
        stream.seek(offset)
        text = stream.read().decode('utf-8', errors='replace')
    if re.search(r'capturing cuda|compil(?:ing|ation took)|torch.compile takes|graph capturing', text, re.I):
        return 'compilation_or_capture_log_in_window'
    if re.search(r'Failed to (?:start|stop) profiler', text, re.I):
        return 'native_profiler_failure_in_window'
    return None


async def measured_window(client, identity, devices, requests, out, name,
                          sources=None, interval_s=.1, target=None, memory_reference=None):
    sampler = telemetry.Sampler(devices, interval_s)
    result = dict(status='invalid', quality_errors=[], requests=len(requests),
                  request_ids=[r['request_id'] for r in requests],
                  prompt_hashes=[r['prompt_hash'] for r in requests],
                  length=requests[0]['length'], replies=[])
    core.write_new(out/(name+'-requests.json'), requests)
    begin = end = None
    try:
        before = await asyncio.to_thread(metrics, identity)
        cursor = log_cursor(identity)
        await asyncio.to_thread(sampler.start)
        begin = time.monotonic_ns()
        for request in requests: result['replies'].append(await client.run(request))
        end = time.monotonic_ns()
    except BaseException as exc:
        result['error'] = safe_error(exc)
        raise
    finally:
        if sampler.thread is not None:
            await asyncio.to_thread(sampler.finish)
        core.write_new(out/(name+'-telemetry.json'), dict(frames=sampler.frames,
            errors=sampler.errors, begin_ns=begin, end_ns=end, clock_domain='client_WSL_monotonic_ns'))
        if 'error' in result: core.write_new(out/(name+'-window.json'), result)
    try:
        after = await asyncio.to_thread(metrics, identity)
    except BaseException as exc:
        result['error'] = safe_error(exc)
        core.write_new(out/(name+'-window.json'), result)
        raise
    result.update(begin_ns=begin, end_ns=end, window_s=(end-begin)/1e9,
                  metrics_before=before, metrics_after=after)
    result['cache_status'], cache_errors = core.cache_miss_status(before, after, len(requests), prefix_policy(identity))
    result['quality_errors'].extend(cache_errors+sampler.errors)
    if issue := new_compilation(cursor): result['quality_errors'].append(issue)
    in_window = [[s for s in frame if begin <= s['mapped']['clock_mhz']['t_ns'] <= end]
                 for frame in sampler.frames]
    # At boundaries each card's field can lie on a different side; require paired
    # frames only in the interior, and retain raw boundary samples in the artifact.
    interior = [f for f in in_window if len(f) == 2]
    if not interior: result['quality_errors'].append('No interior dual-GPU telemetry')
    result['quality_errors'].extend(telemetry.clock_errors(interior, target, memory_reference))
    try:
        energy = telemetry.energy_window(sampler.frames, begin, end, sources, interval_s)
        result.update(energy_j=energy['energy_j'], power_w=energy['power_w'], energy=energy,
                      energy_by_uuid_j={u: v['energy_j'] for u, v in energy['by_uuid'].items()},
                      energy_relative_uncertainty=energy['relative_uncertainty'],
                      temperature_range_c={u: v['temperature_range_c'] for u, v in energy['by_uuid'].items()})
        result['quality_errors'].extend(energy['quality_errors'])
    except Exception as exc:
        result['quality_errors'].append(safe_error(exc))
    result.update(ttft_s=statistics.mean(r['ttft_s'] for r in result['replies']),
                  tpot_s=statistics.mean(r['tpot_s'] for r in result['replies']))
    result['clock_ranges'] = {}
    for u in GPU_UUIDS:
        samples = [s['raw'] for f in interior for s in f if s['uuid'] == u]
        values = [s['clock_mhz'] for s in samples if s.get('clock_mhz') is not None]
        memory = [s['memory_clock_mhz'] for s in samples if s.get('memory_clock_mhz') is not None]
        result['clock_ranges'][u] = dict(core=[min(values), max(values)] if values else None,
            memory=sorted(set(memory)), memory_mode=Counter(memory).most_common(1)[0][0] if memory else None)
    result['status'] = 'valid' if not result['quality_errors'] else 'invalid'
    core.write_new(out/(name+'-window.json'), result)
    return result


async def load_frames(client, devices, seed, prefix, artifact=None):
    sampler = telemetry.Sampler(devices)
    seed = int(digest([seed, prefix])[:15], 16)
    try:
        await asyncio.to_thread(sampler.start)
        for request in prepared(client, 512, seed, 3, prefix): await client.run(request)
    finally:
        await asyncio.to_thread(sampler.finish)
        if artifact:
            core.write_new(artifact, dict(frames=sampler.frames, errors=sampler.errors))
    if sampler.errors: raise RuntimeError(str(sampler.errors))
    return sampler.frames


async def settle(client, devices, target, seed, prefix):
    await asyncio.sleep(.5)
    deadline, consecutive, records, count = time.monotonic()+5, 0, [], 0
    task = None
    try:
        while time.monotonic() < deadline:
            if task is None or task.done():
                if task: await task
                task = asyncio.create_task(client.run(client.prepare(512, seed+count, f'{prefix}-{count}')))
                count += 1
            snapshots = await asyncio.gather(*(asyncio.to_thread(telemetry.sample_device, d) for d in devices))
            records.append(snapshots)
            good = all(abs(s['raw'].get('clock_mhz', -100000)-target) <= 15 for s in snapshots)
            consecutive = consecutive+1 if good else 0
            if consecutive >= 3: return records
            await asyncio.sleep(.1)
        raise RuntimeError('Both clocks did not settle within 5s; no measurement retry')
    finally:
        if task: await task  # finish the finite unmeasured load; never cancel a live request mid-work


async def restore_checked(controller, client, devices, baseline, seed, prefix, artifact=None):
    controller.reset()
    after = await load_frames(client, devices, seed, prefix, artifact)
    evidence = telemetry.restoration_evidence(baseline, after, controller.last_target)
    controller.restore_result.update(evidence=evidence, verified=evidence['verified'])
    if not evidence['verified']:
        raise RuntimeError('Reset returned but physical restoration remains unverified; stop clock work')
    return after


async def observe(args):
    identity = checked_identity(args.identity, require_ready=False)
    if not all(identity['profiler_routes'].values()) or not identity['profile_overrides_supported']:
        raise ValueError('Resident profiler/iteration override API unavailable; do not switch eager or restart here')
    out = fresh(args.out)
    trace_dir = Path(identity['launch']['profiler_config']['torch_profiler_dir'])
    report = dict(schema=core.SCHEMA, status='failed', identity_fingerprint=identity['fingerprint'],
                  windows=[], scope='sampled_prefill_early_late_decode_not_entire_request_inventory')
    try:
        async with client_api.BenchClient(identity['base_url'], model_id(identity)) as client:
            await stop_profile(identity)
            for li, length in enumerate(core.LENGTHS):
                await warmup(client, length, 700000+li*100, f'warm-observe-{out.name}-{length}')
                # Native WorkerProfiler.step starts delayed capture ON the selected
                # forward. First forward is 1, so decode 1 starts at delay 2.
                for name, delay, limit in (('prefill', 0, 1), ('decode_early', 2, 3), ('decode_tail', 62, 3)):
                    prefix = f'svc{uuid_module.uuid4().hex[:12]}_{length}_{name}'
                    request = client.prepare(length, int(digest([out.name, length, name])[:15], 16), prefix)
                    core.write_new(out/(prefix+'-request.json'), request)
                    before = await asyncio.to_thread(metrics, identity)
                    entry = dict(prefix=prefix, length=length, requested_phase=name,
                                 delay_iterations=delay, max_iterations=limit, ranks={})
                    report['windows'].append(entry)
                    try:
                        await asyncio.to_thread(client_api.http, identity['base_url'], '/start_profile',
                            dict(profile_prefix=prefix, delay_iterations=delay, max_iterations=limit), 600)
                        entry['reply'] = await client.run(request)
                    finally:
                        await stop_profile(identity)
                    after = await asyncio.to_thread(metrics, identity)
                    entry['cache_status'], issues = core.cache_miss_status(before, after, 1, prefix_policy(identity))
                    files = [p for p in trace_dir.rglob(prefix+'*') if p.is_file()
                             and (p.name.endswith('.trace.json') or p.name.endswith('.trace.json.gz'))]
                    by_rank = {}
                    for file in files:
                        match = re.search(r'(?:^|_)tp([01])(?:_|\.|$)', file.name)
                        if not match: continue
                        rank = int(match.group(1))
                        if rank in by_rank: raise ValueError('Multiple trace files for one rank/window')
                        by_rank[rank] = file
                    if set(by_rank) != {0, 1}: raise ValueError('Both rank traces must be exported before proceeding')
                    for rank, file in by_rank.items():
                        trace = core.read_trace(file)
                        parsed = core.correlate_trace(trace, limit)
                        del trace
                        if name.startswith('decode') and any(re.search(r'context_[1-9]', n)
                                                            for n in parsed['annotation_names']):
                            parsed['issues'].append('Decode window includes context work; scheduling differs')
                            parsed['status'] = 'incomplete'
                        if name == 'prefill' and not any(re.search(r'context_[1-9]', n)
                                                        for n in parsed['annotation_names']):
                            parsed['issues'].append('Prefill not observed')
                            parsed['status'] = 'incomplete'
                        entry['ranks'][str(rank)] = dict(trace=str(file), sha256=client_api.sha(file),
                                                       summary={k: v for k, v in parsed.items() if k != 'signatures'})
                        core.write_new(out/f'{prefix}-rank-{rank}.json', parsed)
                        if parsed['status'] != 'boundary_checked':
                            issues.extend(f'rank {rank}: {issue}' for issue in parsed['issues'])
                    if issues: raise ValueError(str(issues))
                    emit(dict(window=prefix, status='boundary_checked', ranks=2))
            report['status'] = 'observed_sampled_windows'
            report['operator_models_ready'] = False
    except BaseException as exc:
        report['error'] = safe_error(exc)
        raise
    finally:
        core.write_new(out/'observation.json', report)
    return report


async def pilot(args):
    identity = checked_identity(args.identity)
    config = core.read(args.telemetry)
    out = fresh(args.out)
    result = dict(schema=core.SCHEMA, status='failed', identity_fingerprint=identity['fingerprint'],
                  telemetry_config_sha256=digest(config), requests_per_window=args.requests,
                  sample_interval_s=.1)
    devices = []
    try:
        devices = telemetry.open_devices(config)
        async with client_api.BenchClient(identity['base_url'], model_id(identity)) as client:
            await stop_profile(identity)
            await warmup(client, 512, 900000, 'pilot-warm-'+out.name)
            requests = prepared(client, 512, int(digest(['pilot', out.name])[:15], 16),
                                args.requests, 'pilot-'+out.name)
            row = await measured_window(client, identity, devices, requests, out, 'pilot')
            result.update(window=row, client_source_sha256=client.source_sha256,
                          tokenizer_fingerprint=client.tokenizer_fingerprint)
            if row.get('energy'):
                result['sources'] = {u: r['source'] for u, r in row['energy']['by_uuid'].items()}
                result['memory_reference'] = {u: r['memory_mode'] for u, r in row['clock_ranges'].items()}
                result['suggested_requests_if_budget_failed'] = max(args.requests,
                    math.ceil(args.requests*row['energy_relative_uncertainty']/.03*1.25))
            result['status'] = 'ready' if row['status'] == 'valid' else 'not_ready'
    except BaseException as exc:
        result['error'] = safe_error(exc)
        raise
    finally:
        for d in devices:
            try: d.close()
            except Exception as exc:
                result.update(status='failed', close_error=safe_error(exc))
        core.write_new(out/'pilot.json', result)
    return result


async def clock_check(args):
    if not args.apply or not args.ack_default_auto:
        raise ValueError('Clock acceptance requires --apply and --ack-default-auto')
    identity = checked_identity(args.identity)
    config = core.read(args.telemetry)
    if any(r['backend'] != 'bridge' for r in config):
        raise ValueError('This Windows/WSL clock gate requires bridge disconnect recovery on both cards')
    out = fresh(args.out)
    result = dict(schema=core.SCHEMA, status='failed', identity_fingerprint=identity['fingerprint'],
                  telemetry_config_sha256=digest(config), cases=[], synthetic=False)
    devices = []
    controller = telemetry.DualClocks(devices, apply=True)
    baseline = []
    try:
        devices = telemetry.open_devices(config)
        controller.devices = devices
        async with client_api.BenchClient(identity['base_url'], model_id(identity)) as client:
            await stop_profile(identity)
            baseline = await load_frames(client, devices, 910000, 'clock-baseline-'+out.name)
            core.write_new(out/'baseline.json', baseline)
            try:
                for i, target in enumerate((2527, 2250)):
                    controller.set(target)
                    pre = await settle(client, devices, target, 911000+i*100, 'clock-settle-'+str(target))
                    frames = await load_frames(client, devices, 912000+i*100, 'clock-loaded-'+str(target))
                    core.write_new(out/f'clock-{target}.json', dict(precheck=pre, loaded=frames))
                    # Loaded points only for acceptance; formal windows check every interior readback.
                    loaded = [[s for s in f if (s['raw'].get('utilization_pct') or 0) > 0] for f in frames]
                    paired = [f for f in loaded if len(f) == 2]
                    errors = telemetry.clock_errors(paired, target)
                    if len(paired) < 3 or errors: raise ValueError('Loaded dual-card lock verification failed: '+str(errors))
                    result['cases'].append(dict(target_mhz=target, loaded_readback_verified=True))
            finally:
                if controller.restore_result['required']:
                    await restore_checked(controller, client, devices, baseline, 913000,
                                          'clock-normal-restore', out/'normal-restore.json')
                    result['normal_restoration'] = dict(controller.restore_result)
            # A controlled Python exception tests the same finally path as a client abort.
            try:
                controller.set(2250)
                await settle(client, devices, 2250, 914000, 'clock-abort')
                raise InterruptedError('intentional_restore_acceptance')
            except InterruptedError:
                result['controlled_exception_injected'] = True
            finally:
                await restore_checked(controller, client, devices, baseline, 915000,
                                      'clock-abort-restore', out/'exception-restore.json')
                result['exception_restoration'] = dict(controller.restore_result)
            if all(r['backend'] == 'bridge' for r in config):
                controller.set(2250)
                await settle(client, devices, 2250, 916000, 'clock-disconnect')
                for device in devices: device.close()
                devices = []
                # Bridge Session.close resets dirty clocks after EOF. Reconnect read-only.
                await asyncio.sleep(1)
                devices = telemetry.open_devices(config)
                frames = await load_frames(client, devices, 917000, 'clock-disconnect-restore')
                evidence = telemetry.restoration_evidence(baseline, frames, 2250)
                core.write_new(out/'disconnect-restore.json', frames)
                result['disconnect_restoration'] = evidence
                if not evidence['verified']: raise RuntimeError('Bridge disconnect restoration unverified')
                controller.dirty.clear()
                controller.restore_result = dict(required=True, reset_returned=False,
                    verified=True, evidence=evidence, basis='server_EOF_cleanup_loaded_readback')
            result['status'] = 'passed'
    except BaseException as exc:
        result['error'] = safe_error(exc)
        raise
    finally:
        if controller.dirty:
            try: controller.reset()
            except Exception as exc: result['restore_error'] = safe_error(exc)
        for device in devices:
            try: device.close()
            except Exception as exc: result['close_error'] = safe_error(exc)
        result['restoration'] = controller.restore_result
        core.write_new(out/'clock-check.json', result)
    return result


async def collect(args):
    identity = checked_identity(args.identity)
    config, pilot_data = core.read(args.telemetry), core.read(args.pilot)
    if (pilot_data.get('status') != 'ready' or pilot_data['identity_fingerprint'] != identity['fingerprint']
            or pilot_data['telemetry_config_sha256'] != digest(config)):
        raise ValueError('Require a successful same-service telemetry pilot')
    fixed = args.clock_mode == 'fixed'
    if fixed:
        if any(r['backend'] != 'bridge' for r in config):
            raise ValueError('Fixed clocks require bridge disconnect recovery on both cards')
        if not args.apply or not args.clock_check: raise ValueError('Fixed clocks require --apply and --clock-check')
        gate = core.read(args.clock_check)
        telemetry.require_clock_gate(gate, identity['fingerprint'], digest(config))
    elif args.apply: raise ValueError('--apply is not used for current-frequency observations')
    frozen = core.read(args.model) if args.model else None
    if args.split == 'validation' and frozen is None: raise ValueError('Freeze --model before validation collection')
    if args.split == 'train' and frozen is not None: raise ValueError('Training must not consume a fitted model')
    protocol = dict(lengths=list(core.LENGTHS), output_tokens=64, concurrency=1,
        requests_per_window=pilot_data['requests_per_window'], windows_per_cell=3,
        frequencies=['2527', '2250'] if fixed else ['current'],
        sample_interval_s=pilot_data['sample_interval_s'], sources=pilot_data['sources'],
        memory_reference=pilot_data['memory_reference'],
        tokenizer_fingerprint=pilot_data['tokenizer_fingerprint'],
        client_source_sha256=pilot_data['client_source_sha256'],
        telemetry_config_sha256=digest(config), precheck='0.5s_then_5s_three_reads_extra_unmeasured_requests',
        latency_semantics='installed_vllm_benchmark_HTTP_stream_TTFT_TPOT;window_includes_client_gaps',
        energy_semantics='both_GPU_boards_same_client_window_no_idle_subtraction',
        primary_acceptance=core.LIMITS)
    if frozen and (frozen['protocol'] != protocol or frozen['training_seed'] == args.seed
                   or frozen['identity_fingerprint'] != identity['fingerprint']):
        raise ValueError('Validation protocol/identity must match frozen model with a distinct seed')
    out = fresh(args.out)
    data = dict(schema=core.SCHEMA, status='failed', split=args.split, seed=args.seed,
        synthetic=False, identity_ready=identity['identity_ready'], identity_fingerprint=identity['fingerprint'],
        identity=identity, protocol=protocol, windows=[], holdout_1024='sealed_not_collected',
        frozen_model_sha256=digest(frozen) if frozen else None,
        restoration=dict(required=fixed, verified=False))
    devices = []
    controller = telemetry.DualClocks(devices, apply=args.apply)
    try:
        devices = telemetry.open_devices(config)
        controller.devices = devices
        async with client_api.BenchClient(identity['base_url'], model_id(identity)) as client:
            if (client.source_sha256 != protocol['client_source_sha256'] or
                    client.tokenizer_fingerprint != protocol['tokenizer_fingerprint']):
                raise ValueError('Benchmark source or tokenizer changed since pilot')
            await stop_profile(identity)
            # Build ALL lists before fitting labels, with seeds hashed by split/cell.
            plan = []
            for fi, freq in enumerate(protocol['frequencies']):
                for repeat in range(3):
                    for li in range(3):
                        length = core.LENGTHS[(li+repeat)%3]
                        name = f'{args.split}-{freq}-{length}-{repeat}'
                        seed = int(digest([args.seed, args.split, freq, length, repeat])[:15], 16)
                        requests = prepared(client, length, seed, protocol['requests_per_window'], name+'-'+out.name)
                        if frozen and set(r['prompt_hash'] for r in requests) & set(frozen['training_prompt_hashes']):
                            raise ValueError('Training prompt repeated in validation')
                        plan.append(dict(frequency=freq, length=length, repeat=repeat, name=name, requests=requests))
            core.write_new(out/'request-plan.json', plan)
            baseline = await load_frames(client, devices, 920000+args.seed, 'collect-baseline-'+out.name) if fixed else []
            if fixed: core.write_new(out/'restore-baseline.json', baseline)
            active_freq = None
            try:
                for i, entry in enumerate(plan):
                    freq, length, name = entry['frequency'], entry['length'], entry['name']
                    if fixed and freq != active_freq:
                        controller.set(int(freq)); active_freq = freq
                    await warmup(client, length, 930000+args.seed+i*20, 'warm-'+name+'-'+out.name)
                    if fixed:
                        pre = await settle(client, devices, int(freq), 940000+args.seed+i*20, 'settle-'+name)
                        core.write_new(out/(name+'-precheck.json'), pre)
                    row = await measured_window(client, identity, devices, entry['requests'], out, name,
                        sources=protocol['sources'], interval_s=protocol['sample_interval_s'],
                        target=int(freq) if fixed else None, memory_reference=protocol['memory_reference'])
                    row['frequency'] = freq
                    data['windows'].append(row)
                    emit(dict(window=name, status=row['status'], errors=row['quality_errors'],
                              energy_j=row.get('energy_j'), window_s=row.get('window_s')))
                    if row['status'] != 'valid': raise ValueError('Invalid window preserved; stop rather than retry/omit')
            finally:
                if fixed and controller.restore_result['required']:
                    await restore_checked(controller, client, devices, baseline, 950000+args.seed,
                                          'collect-restore-'+out.name, out/'restore-after.json')
                data['restoration'] = controller.restore_result if fixed else dict(required=False, verified=False)
            data['temperature_range_c'] = {u: [
                min(r['temperature_range_c'][u][0] for r in data['windows']),
                max(r['temperature_range_c'][u][1] for r in data['windows'])] for u in GPU_UUIDS}
            live = checked_identity(args.identity)
            if live['fingerprint'] != data['identity_fingerprint']: raise ValueError('Service identity changed during collection')
            data['status'] = 'collected'
    except BaseException as exc:
        data['error'] = safe_error(exc)
        raise
    finally:
        if controller.dirty:
            try: controller.reset()
            except Exception as exc: data['restore_error'] = safe_error(exc)
        if fixed: data['restoration'] = controller.restore_result
        for device in devices:
            try: device.close()
            except Exception as exc: data['close_error'] = safe_error(exc)
        core.write_new(out/'measurements.json', data)
    return data


def compact(value):
    result = {k: value[k] for k in ('schema', 'status', 'error', 'restore_error', 'identity_ready',
        'fingerprint', 'profiler_routes', 'profile_overrides_supported', 'unknown', 'parameters',
        'summary', 'by_frequency', 'max_energy_relative_uncertainty',
        'scope_errors', 'restoration', 'sources', 'requests_per_window',
        'training_windows', 'independent_validation',
        'scope', 'temperature_range_c', 'observed_clock_ranges_mhz', 'identity_unknown',
        'vllm_version', 'torch_version', 'cuda', 'triton', 'profiler_config',
        'suggested_requests_if_budget_failed', 'normal_restoration', 'exception_restoration',
        'disconnect_restoration') if k in value}
    if 'windows' in value: result['windows'] = len(value['windows'])
    if 'window' in value:
        result['pilot_quality_errors'] = value['window']['quality_errors']
        result['pilot_energy_relative_uncertainty'] = value['window'].get('energy_relative_uncertainty')
    if 'startup_identity' in value:
        result['startup_identity'] = {k: v for k, v in value['startup_identity'].items()
                                      if k != 'engine_config_line'}
    return result or value


def text_report(args):
    value = core.read(args.file)
    if args.section == 'summary': return compact(value)
    if args.section == 'cells': return value.get('cells', {'error': 'No validation cells'})
    if args.section == 'config':
        return {k: value.get(k) for k in ('base_url', 'service_process', 'startup_identity',
            'gpu_index_uuids', 'gpu_hardware', 'rank_uuid_basis', 'unknown', 'protocol', 'scope',
            'temperature_range_c', 'observed_clock_ranges_mhz')}
    windows = value.get('windows', [])
    if not isinstance(windows, list): raise ValueError('Expected observation windows')
    if args.section == 'coverage':
        return [dict(length=w['length'], phase=w.get('requested_phase'), rank=r,
                     **{k: s['summary'].get(k) for k in ('status', 'annotated_iterations',
                         'iterations_with_gpu', 'gpu_activities', 'attribution_fraction',
                         'graph_replay_observed', 'issues')})
                for w in windows for r, s in w.get('ranks', {}).items()]
    if args.length is None or args.rank is None:
        raise ValueError('Small kernel reports require --length 128/512/2048 and --rank 0/1')
    rows = []
    for w in windows:
        if w['length'] != args.length: continue
        path = Path(args.file).parent/f'{w["prefix"]}-rank-{args.rank}.json'
        report = core.read(path)
        rows.append(dict(length=w['length'], rank=args.rank, requested_phase=w['requested_phase'],
                         top_sampled_kernels=report['signatures'][:args.top],
                         full_operator_coverage_verified=False, usable_as_fitting_labels=False))
    return rows


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    pre = sub.add_parser('preflight'); pre.add_argument('--out', required=True)
    pre.add_argument('--launch', action='store_true', help=argparse.SUPPRESS)
    ins = sub.add_parser('inspect'); ins.add_argument('--base-url', default=client_api.BASE_URL)
    ins.add_argument('--launch-dir'); ins.add_argument('--out', required=True)
    for name in ('observe', 'pilot', 'clock-check', 'collect'):
        cmd = sub.add_parser(name)
        cmd.add_argument('--identity', required=True); cmd.add_argument('--out', required=True)
        if name != 'observe': cmd.add_argument('--telemetry', required=True)
        if name == 'pilot': cmd.add_argument('--requests', type=int, default=20)
        if name in ('clock-check', 'collect'): cmd.add_argument('--apply', action='store_true')
        if name == 'clock-check': cmd.add_argument('--ack-default-auto', action='store_true')
        if name == 'collect':
            cmd.add_argument('--pilot', required=True); cmd.add_argument('--split', choices=['train', 'validation'], required=True)
            cmd.add_argument('--seed', type=int, required=True); cmd.add_argument('--model')
            cmd.add_argument('--clock-mode', choices=['current', 'fixed'], default='current')
            cmd.add_argument('--clock-check')
    fit = sub.add_parser('fit'); fit.add_argument('--data', required=True); fit.add_argument('--out', required=True)
    val = sub.add_parser('validate'); val.add_argument('--data', required=True)
    val.add_argument('--model', required=True); val.add_argument('--out', required=True)
    rep = sub.add_parser('report'); rep.add_argument('--file', required=True)
    rep.add_argument('--section', choices=['summary', 'config', 'coverage', 'kernels', 'cells'], default='summary')
    rep.add_argument('--rank', type=int, choices=[0, 1]); rep.add_argument('--length', type=int, choices=core.LENGTHS)
    rep.add_argument('--top', type=int, choices=range(1, 11), default=3)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    previous_term = signal.getsignal(signal.SIGTERM)
    def terminate(signum, frame):
        raise KeyboardInterrupt('SIGTERM: restore owned clocks')
    signal.signal(signal.SIGTERM, terminate)
    try:
        if getattr(args, 'requests', 20) < 2: raise ValueError('At least two requests per pilot window')
        if args.command == 'preflight': result = client_api.preflight(args.out, args.launch)
        elif args.command == 'inspect':
            result = client_api.inspect_service(args.base_url, args.launch_dir)
            core.write_new(fresh(args.out)/'identity.json', result)
        elif args.command in ('observe', 'pilot', 'collect', 'clock-check'):
            function = {'observe': observe, 'pilot': pilot, 'collect': collect, 'clock-check': clock_check}[args.command]
            result = asyncio.run(function(args))
        elif args.command == 'fit':
            result = core.fit(core.read(args.data)); core.write_new(Path(args.out)/'model.json', result)
        elif args.command == 'validate':
            result = core.validate(core.read(args.data), core.read(args.model))
            core.write_new(Path(args.out)/'validation.json', result)
        else:
            emit(text_report(args))
            return 0
        emit(compact(result))
        return 0 if result.get('status') not in ('failed', 'not_ready', 'not_passed', 'needs_identity_evidence') else 2
    except KeyboardInterrupt:
        emit(dict(status='interrupted', message='Evidence retained; inspect restoration before further work'))
        return 130
    except Exception as exc:
        emit(dict(status='failed', error=safe_error(exc)))
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous_term)


if __name__ == '__main__':
    sys.exit(main())
