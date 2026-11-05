"""User-operated eager experiment: preflight, serve, inspect, observe, clock gate,
paired collect, empirical fit/validate and guarded serial deployment.

Never stops a service, changes dependencies, or labels window fits Eq1/Eq2.
"""
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from . import eager_core as core, service_client as client_api, service_model as service
from . import service_telemetry as telemetry
from .service_core import read, write_new
from .observe_core import GPU_UUIDS, digest
from .eager_ipc import rpc, LocalServer, Rendezvous, PairSwitch, LookaheadSwitch
from .performance import safe_error

ROOT = Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True,
                          check=True, timeout=20).stdout.strip()


def close_devices(devices, result):
    errors = []
    for device in devices:
        try: device.close()
        except BaseException as exc: errors.append(safe_error(exc))
    if errors: result.update(status='failed', close_errors=errors)


def preflight_record(model):
    from .native import check_vllm_sources, triton_record
    from vllm.engine.arg_utils import EngineArgs
    import torch
    if os.name != 'posix': raise ValueError('Run preflight in target WSL Conda, not development Windows')
    if git('branch', '--show-current') != '5090_eager':
        raise ValueError('Select the separate 5090_eager branch before preflight')
    if git('status', '--porcelain', '--untracked-files=no'):
        raise ValueError('Tracked working tree changes present; preserve/review them before preflight')
    required = {'dtype', 'load_format', 'worker_cls', 'enforce_eager', 'tensor_parallel_size',
                'language_model_only', 'max_num_seqs', 'max_model_len'}
    if not required <= set(EngineArgs.__dataclass_fields__): raise ValueError('Installed EngineArgs contract changed')
    if torch.version.cuda != '12.9': raise ValueError('Require existing PyTorch cu129; do not mix CUDA13')
    if any(os.environ.get(k) is not None for k in ('NCCL_P2P_DISABLE', 'NCCL_SHM_DISABLE',
            'NCCL_IB_DISABLE', 'NCCL_DEBUG', 'NCCL_DEBUG_SUBSYS')):
        raise ValueError('Use launcher to preserve successful NCCL environment')
    if os.environ.get('NCCL_CUMEM_HOST_ENABLE') != '0': raise ValueError('NCCL_CUMEM_HOST_ENABLE must be 0')
    if os.environ.get('SPARSEDVFS_TRITON_PATH') != client_api.TRITON:
        raise ValueError('Persistent Triton override is missing')
    smi = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,driver_version',
                         '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=10)
    rows = [list(map(str.strip, s.split(','))) for s in smi.stdout.splitlines() if s.strip()]
    if not set(GPU_UUIDS) <= {r[1] for r in rows}: raise ValueError('Expected GPU UUIDs not found')
    # Hash only source/contracts; never print environment wholesale or bridge credentials.
    sources = check_vllm_sources()
    return dict(schema=core.SCHEMA, status='preflight_ready_not_loaded_or_clock_verified',
        commit=git('rev-parse', 'HEAD'), branch=git('branch', '--show-current'),
        tools={p.name: client_api.sha(p) for p in Path(__file__).parent.glob('eager*.py')},
        packages={n: importlib.metadata.version(n) for n in ('vllm', 'torch', 'transformers')},
        python=sys.version, cuda=torch.version.cuda, triton=triton_record(client_api.TRITON),
        vllm_sources=sources, weights=core.weights(model), gpu_inventory=rows,
        capabilities=core.CAPABILITIES, native_libraries={p.name: client_api.sha(p) for p in (ROOT/'build').glob('libfuse*.so')},
        native_abi_and_cuda_linkage='not_loaded_or_certified_by_this_preflight',
        environment={k: os.environ.get(k) for k in ('SPARSEDVFS_TRITON_PATH', 'TRITON_CACHE_DIR',
            'NCCL_CUMEM_HOST_ENABLE', 'VLLM_USE_V2_MODEL_RUNNER')})


def serve(args):
    run = Path(args.run).resolve()
    old = read(run/'preflight.json')
    actual = preflight_record(args.model)
    if old != actual: raise ValueError('Preflight/environment changed; recheck BEFORE stopping service')
    if (run/'launch.json').exists(): raise FileExistsError('Launch already attempted; preserve evidence, no automatic restart')
    session = uuid.uuid4().hex
    runtime = Path('/tmp')/('sdvfs-'+session)
    runtime.mkdir(mode=0o700)
    os.environ.update(SPARSEDVFS_EAGER_RUN=str(run), SPARSEDVFS_EAGER_RUNTIME=str(runtime),
                      SPARSEDVFS_EAGER_SESSION=session, CUDA_VISIBLE_DEVICES=','.join(GPU_UUIDS),
                      VLLM_USE_V2_MODEL_RUNNER='0', VLLM_WORKER_MULTIPROC_METHOD='spawn')
    argv = ['vllm', 'serve', args.model, '--host', '0.0.0.0', '--port', '8000',
        '--tensor-parallel-size', '2', '--dtype', 'bfloat16', '--load-format', 'auto',
        '--enforce-eager', '--max-model-len', '4096', '--max-num-seqs', '1',
        '--gpu-memory-utilization', '0.90', '--reasoning-parser', 'qwen3',
        '--language-model-only', '--trust-remote-code', '--worker-cls', 'rtx5090.eager_worker.EagerWorker']
    write_new(run/'launch.json', dict(schema=core.SCHEMA, session=session, runtime=str(runtime),
        preflight_sha256=digest(old), process=client_api.process_record(os.getpid()), argv=argv))
    os.execvp(argv[0], argv)


def rank_commands(identity, command):
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(rpc, Path(identity['runtime'])/f'rank-{r}.sock',
            dict(command, session=identity['session']), 30) for r in (0, 1)]
        values, failures = [], []
        for f in futures:
            try: values.append(f.result())
            except BaseException as exc: failures.append(safe_error(exc))
    if failures:
        if command.get('op') == 'arm' and command.get('mode') != 'off':
            # A one-sided arm must not accidentally instrument a later request.
            for rank in (0, 1):
                try:
                    rpc(Path(identity['runtime'])/f'rank-{rank}.sock',
                        dict(op='arm', mode='off', session=identity['session']), 5)
                except BaseException as exc: failures.append('disarm: '+safe_error(exc))
        raise RuntimeError('Two-rank RPC failed: '+str(failures))
    return values


def inspect(run, base):
    run = Path(run).resolve()
    launch = read(run/'launch.json')
    current = client_api.process_record(launch['process']['pid'])
    if current['start_ticks'] != launch['process']['start_ticks'] or 'serve' not in current['argv']:
        raise ValueError('Server PID changed/reused')
    preliminary = dict(session=launch['session'], runtime=launch['runtime'])
    states = rank_commands(preliminary, dict(op='status'))
    if any(s['poisoned'] or s['running'] or s['pending'] or s['mode'] != 'off' for s in states):
        raise ValueError('Workers are not healthy and idle')
    ranks = core.validate_ranks([s['identity'] for s in states])
    for r in ranks:
        if client_api.process_record(r['pid'])['start_ticks'] != r['process']['start_ticks']:
            raise ValueError('Worker PID was reused')
    models = client_api.http(base, '/v1/models')
    if len(models['data']) != 1 or models['data'][0].get('root', models['data'][0]['id']) != ranks[0]['weights']['path']:
        raise ValueError('HTTP model differs from actual workers')
    # Check the selected endpoint's port; model/process identity is checked above.
    # The remote hostname's listener ownership is not inferred from an HTTP reply.
    from urllib.parse import urlparse
    port = urlparse(base).port or 80
    if port != 8000: raise ValueError('This launcher is pinned to port 8000')
    log = (run/'startup.log').read_text(encoding='utf-8', errors='replace')
    startup = client_api.startup_identity(log)
    if not startup['weights_loaded_log'] or startup['enforce_eager'] not in ('True', 'true'):
        raise ValueError('Startup weight loading/eager evidence absent')
    value = dict(schema=core.SCHEMA, session=launch['session'], runtime=launch['runtime'],
        launch_dir=str(run), base_url=base, models=models, ranks=ranks, identity_ready=True,
        startup_identity=startup, profiler_routes={'/start_profile': False, '/stop_profile': False},
        capabilities=core.CAPABILITIES,
        configuration='real_BF16_TP2_eager_512_64', service_process=current)
    value['fingerprint'] = digest(dict(session=value['session'], ranks=ranks,
        base_url=base, server_start=current['start_ticks'], preflight=launch['preflight_sha256']))
    return value


def checked(path):
    old = read(path)
    live = inspect(old['launch_dir'], old['base_url'])
    if live['fingerprint'] != old['fingerprint']: raise ValueError('Live eager identity changed')
    return live


async def clear_cache(identity):
    if service.prefix_policy(identity) is False: return
    paths = (await asyncio.to_thread(client_api.http, identity['base_url'], '/openapi.json'))['paths']
    if 'post' not in paths.get('/reset_prefix_cache', {}):
        raise RuntimeError('Paired prompts need supported cache reset; do not silently change service configuration')
    answer = await asyncio.to_thread(client_api.http, identity['base_url'], '/reset_prefix_cache', {})
    if answer is False: raise RuntimeError('Prefix cache reset failed')


async def isolate_observation_cache(identity, request):
    """Observation-only isolation; worker scheduler evidence still proves a miss.

    Do not relax reset requirements for paired energy/latency windows.
    """
    if service.prefix_policy(identity) is False:
        return dict(method='prefix_cache_disabled', miss_verified=False)
    api = await asyncio.to_thread(client_api.http, identity['base_url'], '/openapi.json')
    paths = api.get('paths', {})
    if 'post' in paths.get('/reset_prefix_cache', {}):
        answer = await asyncio.to_thread(client_api.http, identity['base_url'], '/reset_prefix_cache', {})
        if answer is False: raise RuntimeError('Prefix cache reset failed')
        return dict(method='reset_prefix_cache', miss_verified=False)
    schema = paths.get('/v1/completions', {}).get('post', {}).get('requestBody', {}).get(
        'content', {}).get('application/json', {}).get('schema', {})
    seen = set()
    while '$ref' in schema:
        ref = schema['$ref']
        if not ref.startswith('#/') or ref in seen:
            raise ValueError('Cannot resolve live completion request schema')
        seen.add(ref)
        schema = api
        for key in ref[2:].split('/'):
            schema = schema.get(key.replace('~1', '/').replace('~0', '~'), {})
    field = schema.get('properties', {}).get('cache_salt', {})
    if not (field.get('type') == 'string' or any(
            option.get('type') == 'string' for option in field.get('anyOf', []))):
        raise RuntimeError('Observation requires reset_prefix_cache or advertised string cache_salt; no request armed')
    request['cache_salt'] = uuid.uuid4().hex
    return dict(method='unique_request_cache_salt', miss_verified=False,
                advertised_field=field, openapi_sha256=digest(api))


def coverage(directory, identity):
    rows = [read(Path(directory)/f'rank-{rank}-execution.json') for rank in (0, 1)]
    expected = ['prefill']+[f'decode.{i}' for i in range(63)]
    for rank, row in enumerate(rows):
        if row['status'] != 'complete' or row['rank'] != rank or row['identity_sha256'] != digest(identity['ranks'][rank]):
            raise ValueError('Wrong worker execution evidence')
        if row['steps'] is None:
            if row.get('contexts') != expected or row.get('unit_order_verified') is not True:
                raise ValueError('Incomplete minimal runtime coverage')
        elif [s['context'] for s in row['steps']] != expected or any(s['units'] != core.UNITS for s in row['steps']):
            raise ValueError('Incomplete 64-step/64-layer coverage')
    if rows[0]['steps_sha256'] != rows[1]['steps_sha256']:
        raise ValueError('Rank execution/scheduler sequences differ')
    return dict(passed=True, identity_fingerprint=identity['fingerprint'],
        execution_sha256=[digest(r) for r in rows], units=core.UNITS,
        scope='module_sequence_not_kernel_DAG_or_replay', contexts=expected)


async def observe(args):
    identity = checked(args.identity)
    out = service.fresh(args.out)
    result = dict(schema=core.SCHEMA, status='failed', identity_fingerprint=identity['fingerprint'])
    try:
        async with client_api.BenchClient(identity['base_url'], service.model_id(identity)) as client:
            await service.warmup(client, 512, 100, 'eager-observe-'+out.name)
            request = client.prepare(512, args.seed, 'eager-'+uuid.uuid4().hex)
            result['cache_isolation'] = await isolate_observation_cache(identity, request)
            result['client_provenance'] = dict(commit=git('rev-parse', 'HEAD'),
                eager_sha256=client_api.sha(__file__), service_client_sha256=client_api.sha(client_api.__file__))
            write_new(out/'request.json', request)
            rank_commands(identity, dict(op='arm', mode='observe' if args.trace else 'units',
                out=str(out), request_id=request['request_id'], prompt_hash=request['prompt_hash']))
            result['reply'] = await client.run(request)
            result['coverage'] = coverage(out, identity)
            result['cache_isolation'].update(miss_verified=True,
                evidence='both_rank_context_key_requires_admitted_zero_and_full_512_prefill')
            result['status'] = 'complete'
    except BaseException as exc:
        result.update(status='failed', error=safe_error(exc)); raise
    finally:
        write_new(out/'observation.json', result)
    return result


async def switch_check(args):
    """Extra bidirectional timing, with load and API/physical brackets kept apart."""
    identity = checked(args.identity)
    config = read(args.telemetry)
    core.require_gate(read(args.gate), identity['fingerprint'], digest(config))
    if not args.apply: raise ValueError('Switch calibration requires --apply')
    out = service.fresh(args.out)
    devices = telemetry.open_devices(config)
    controller = telemetry.DualClocks(devices, apply=True)
    result = dict(schema=core.SCHEMA, status='failed', identity_fingerprint=identity['fingerprint'],
                  telemetry_config_sha256=digest(config), transitions=[])
    try:
        async with client_api.BenchClient(identity['base_url'], service.model_id(identity)) as client:
            baseline = await service.load_frames(client, devices, 42000, out.name, out/'baseline.json')
            try:
                previous = None
                for i, target in enumerate([2250, 2527]*5+[2250]):
                    begin = time.monotonic_ns()
                    controller.set(target)
                    returned = time.monotonic_ns()
                    samples = await service.settle(client, devices, target, 43000+i*100, out.name+str(i))
                    end = time.monotonic_ns()
                    row = dict(from_mhz=previous, to_mhz=target, begin_ns=begin, api_return_ns=returned,
                        confirmed_ns=end, samples=samples,
                        physical_transition_interval_ns=[begin, end],
                        includes_load_start_and_100ms_observation=True, exact_switch_latency=False)
                    result['transitions'].append(row)
                    write_new(out/f'transition-{i}.json', row)
                    previous = target
            finally:
                if controller.restore_result['required']:
                    await service.restore_checked(controller, client, devices, baseline, 45000, out.name, out/'restore.json')
            result.update(status='complete', fine_boundary_precision_verified=False,
                conservative_bound_verified=True,
                conservative_switch_s=max((r['confirmed_ns']-r['begin_ns'])/1e9
                    for r in result['transitions'] if r['from_mhz'] is not None),
                lookahead_authorized=False, reason='100ms observation plus settle delay gives a conservative bound; no sub-ms claim')
    finally:
        if controller.dirty:
            try: controller.reset()
            except BaseException as exc: result['restore_error'] = safe_error(exc)
        result['restoration'] = controller.restore_result
        close_devices(devices, result)
        write_new(out/'switch-check.json', result)
    return result


async def collect(args):
    identity = checked(args.identity)
    config, gate, pilot = read(args.telemetry), read(args.gate), read(args.pilot)
    if not args.apply: raise ValueError('Two fixed-frequency conditions require --apply')
    core.require_gate(gate, identity['fingerprint'], digest(config))
    if pilot['status'] != 'ready' or pilot['identity_fingerprint'] != identity['fingerprint']:
        raise ValueError('A ready pilot from this eager service is required')
    count = pilot['requests_per_window']
    if count < 20: raise ValueError('Use at least 20 pilot requests')
    model = read(args.model) if args.model else None
    if args.split == 'validation' and model is None: raise ValueError('Freeze --model before validation')
    if model and (model['identity_fingerprint'] != identity['fingerprint'] or model['protocol']['requests'] != count):
        raise ValueError('Frozen model protocol/environment differs')
    out = service.fresh(args.out)
    result = dict(schema=core.SCHEMA, status='failed', synthetic=False, split=args.split,
        identity_fingerprint=identity['fingerprint'], seed=args.seed,
        protocol=dict(input=512, output=64, concurrency=1, windows=3, requests=count), windows=[])
    devices = telemetry.open_devices(config)
    controller = telemetry.DualClocks(devices, apply=True)
    try:
        async with client_api.BenchClient(identity['base_url'], service.model_id(identity)) as client:
            await service.warmup(client, 512, args.seed+1000000, out.name+'warm')
            baseline = await service.load_frames(client, devices, args.seed+2000000, out.name, out/'baseline.json')
            try:
                for repeat in range(3):
                    requests = service.prepared(client, 512, args.seed+repeat*count, count, f'{out.name}-{repeat}')
                    if model and set(model['train_prompt_hashes']) & {r['prompt_hash'] for r in requests}:
                        raise ValueError('Validation requests overlap training; do not tune on heldout')
                    conditions = ['auto', '2250', '2527']
                    conditions = conditions[repeat:]+conditions[:repeat]
                    for frequency in conditions:
                        if frequency == 'auto' and controller.dirty:
                            await service.restore_checked(controller, client, devices, baseline, args.seed+3000000+repeat,
                                out.name+f'-auto-{repeat}', out/f'auto-restore-{repeat}.json')
                        elif frequency != 'auto':
                            controller.set(int(frequency))
                            await service.settle(client, devices, int(frequency), args.seed+4000000+repeat, out.name+frequency)
                        await clear_cache(identity)
                        # Same tokens, distinct request IDs across conditions; reset is verified by metrics.
                        paired = [dict(r, request_id=r['request_id']+'-'+frequency) for r in requests]
                        row = await service.measured_window(client, identity, devices, paired, out,
                            f'{repeat}-{frequency}', pilot['sources'], pilot['sample_interval_s'],
                            None if frequency == 'auto' else int(frequency), pilot['memory_reference'])
                        row.update(frequency=frequency, repeat=repeat)
                        result['windows'].append(row)
                        service.emit(dict(window=f'{repeat}-{frequency}', status=row['status'],
                            energy_j=row.get('energy_j'), window_s=row['window_s'], errors=row['quality_errors']))
                        if row['status'] != 'valid': raise ValueError('Invalid window retained; stop rather than omit/refit')
            finally:
                if controller.dirty:
                    await service.restore_checked(controller, client, devices, baseline, args.seed+5000000,
                        out.name, out/'final-restore.json')
            result['status'] = 'collected'
    except BaseException as exc:
        result['error'] = safe_error(exc); raise
    finally:
        if controller.dirty:
            try: controller.reset()
            except BaseException as exc: result['restore_error'] = safe_error(exc)
        result['restoration'] = controller.restore_result
        close_devices(devices, result)
        write_new(out/'dataset.json', result)
    return result


async def serial(args):
    identity = checked(args.identity)
    config, gate, plan = read(args.telemetry), read(args.gate), read(args.plan)
    if args.serial_evidence:
        plan = dict(plan, mode='lookahead', serial_acceptance=read(args.serial_evidence))
    core.require_gate(gate, identity['fingerprint'], digest(config))
    core.validate_plan(plan, identity['fingerprint'])
    if not args.apply: raise ValueError('Serial control requires --apply')
    out = service.fresh(args.out)
    devices = telemetry.open_devices(config)
    controller = telemetry.DualClocks(devices, apply=True)
    action = PairSwitch(controller, plan, out/'boundaries.jsonl')
    queued = LookaheadSwitch(action, out/'lookahead.jsonl') if plan['mode'] == 'lookahead' else None
    if queued: action = queued
    gatekeeper = Rendezvous(action, identity['fingerprint'])
    address = Path(identity['runtime'])/('control-'+uuid.uuid4().hex[:8]+'.sock')
    server = LocalServer(address, gatekeeper.exchange, on_disconnect=gatekeeper.abort)
    result = dict(schema=core.SCHEMA, status='failed', identity_fingerprint=identity['fingerprint'],
        plan_sha256=digest(plan), mode=plan['mode'],
        scope='control_acceptance_not_energy_benefit_or_overlap_validation')
    try:
        async with client_api.BenchClient(identity['base_url'], service.model_id(identity)) as client:
            baseline = await service.load_frames(client, devices, args.seed+10000, out.name, out/'baseline.json')
            try:
                await clear_cache(identity)
                request = client.prepare(512, args.seed, 'serial-'+uuid.uuid4().hex)
                write_new(out/'request.json', request)
                rank_commands(identity, dict(op='arm', mode='serial', out=str(out),
                    request_id=request['request_id'], prompt_hash=request['prompt_hash'], plan=plan, gate=gate, apply=True,
                    fingerprint=identity['fingerprint'], telemetry_hash=digest(config), coordinator=str(address)))
                result['reply'] = await client.run(request)
                result['coverage'] = coverage(out, identity)
                if gatekeeper.error: raise RuntimeError(gatekeeper.error)
                result['status'] = 'control_completed'
            finally:
                gatekeeper.abort('Experiment completed or failed')
                if queued: queued.close()
                if controller.restore_result['required']:
                    # A poisoned worker may prevent this load. Preserve reset-returned
                    # separately and require manual recovery, never claim verification.
                    await service.restore_checked(controller, client, devices, baseline, args.seed+20000,
                        out.name, out/'restore.json')
    except BaseException as exc:
        result.update(status='failed', error=safe_error(exc)); raise
    finally:
        if queued: queued.close()
        server.close()
        if controller.dirty:
            try: controller.reset()
            except BaseException as exc: result['restore_error'] = safe_error(exc)
        result['restoration'] = controller.restore_result
        close_devices(devices, result)
        write_new(out/'serial.json', result)
    return result


async def compare(args):
    """Independent, paired AUTO/frozen fixed/frozen serial-policy windows."""
    from statistics import mean
    identity = checked(args.identity)
    config, gate, plan, frozen, pilot = (read(args.telemetry), read(args.gate), read(args.plan),
                                        read(args.model), read(args.pilot))
    if not args.apply: raise ValueError('Comparison requires explicit --apply')
    core.require_gate(gate, identity['fingerprint'], digest(config))
    core.validate_plan(plan, identity['fingerprint'])
    acceptance = read(args.serial_evidence)
    if (acceptance.get('status') != 'control_completed' or acceptance.get('plan_sha256') != digest(plan)
            or acceptance.get('identity_fingerprint') != identity['fingerprint']
            or acceptance.get('restoration', {}).get('verified') is not True):
        raise ValueError('Complete serial acceptance before formal comparison')
    if plan['mode'] != 'serial': raise ValueError('Lookahead needs a separate overlap validation experiment')
    best = frozen.get('best_fixed')
    if (best not in ('2250', '2527') or frozen['identity_fingerprint'] != identity['fingerprint']
            or pilot['identity_fingerprint'] != identity['fingerprint'] or pilot['status'] != 'ready'
            or frozen['protocol']['requests'] != pilot['requests_per_window']):
        raise ValueError('Missing feasible frozen fixed baseline or matching pilot')
    out = service.fresh(args.out)
    devices = telemetry.open_devices(config)
    controller = telemetry.DualClocks(devices, apply=True)
    action = PairSwitch(controller, plan, out/'boundaries.jsonl')
    address = Path(identity['runtime'])/('compare-'+uuid.uuid4().hex[:8]+'.sock')
    holder = {'gate': None}
    def dispatch(message):
        if holder['gate'] is None: raise RuntimeError('No policy request active')
        return holder['gate'].exchange(message)
    def disconnected(reason):
        if holder['gate']: holder['gate'].abort(reason)
    server = LocalServer(address, dispatch, on_disconnect=disconnected)
    result = dict(schema=core.SCHEMA, status='failed', synthetic=False, windows=[],
        identity_fingerprint=identity['fingerprint'], plan_sha256=digest(plan),
        baseline_model_sha256=digest(frozen), best_fixed=best, seed=args.seed)
    try:
        async with client_api.BenchClient(identity['base_url'], service.model_id(identity)) as client:
            baseline = await service.load_frames(client, devices, args.seed+1000000, out.name, out/'baseline.json')
            class PolicyClient:
                async def run(self, request):
                    directory = out/request['request_id']
                    directory.mkdir()
                    holder['gate'] = Rendezvous(action, identity['fingerprint'])
                    rank_commands(identity, dict(op='arm', mode='serial', out=str(directory),
                        request_id=request['request_id'], prompt_hash=request['prompt_hash'], plan=plan,
                        gate=gate, apply=True, fingerprint=identity['fingerprint'],
                        telemetry_hash=digest(config), coordinator=str(address)))
                    try:
                        reply = await client.run(request)
                        coverage(directory, identity)
                        if holder['gate'].error: raise RuntimeError(holder['gate'].error)
                        return reply
                    finally:
                        holder['gate'].abort('Request finished')
                        holder['gate'] = None
            wrapped = PolicyClient()
            try:
                count = pilot['requests_per_window']
                excluded = set(frozen['train_prompt_hashes']) | set(plan['excluded_prompt_hashes'])
                for repeat in range(3):
                    requests = service.prepared(client, 512, args.seed+repeat*count, count, f'cmp-{repeat}')
                    if excluded & {r['prompt_hash'] for r in requests}: raise ValueError('Comparison request leakage')
                    order = ['auto', 'fixed', 'policy']
                    order = order[repeat:]+order[:repeat]
                    for condition in order:
                        if controller.dirty:
                            await service.restore_checked(controller, client, devices, baseline,
                                args.seed+2000000+repeat, out.name+condition,
                                out/f'{repeat}-{condition}-restore.json')
                        if condition == 'fixed':
                            controller.set(int(best))
                            await service.settle(client, devices, int(best), args.seed+3000000+repeat, out.name+condition)
                        if condition == 'policy': action.current = None  # Do not reuse fixed-mode state.
                        await clear_cache(identity)
                        paired = [dict(r, request_id=r['request_id']+'-'+condition) for r in requests]
                        row = await service.measured_window(wrapped if condition == 'policy' else client,
                            identity, devices, paired, out, f'{repeat}-{condition}',
                            pilot['sources'], pilot['sample_interval_s'],
                            int(best) if condition == 'fixed' else None, pilot['memory_reference'])
                        row.update(condition=condition, repeat=repeat)
                        result['windows'].append(row)
                        if row['status'] != 'valid': raise ValueError('Invalid comparison window retained')
            finally:
                if controller.dirty:
                    await service.restore_checked(controller, client, devices, baseline, args.seed+4000000,
                        out.name, out/'final-restore.json')
            values = {c: {k: mean(w[k] for w in result['windows'] if w['condition'] == c)
                          for k in ('window_s', 'energy_j')} for c in ('auto', 'fixed', 'policy')}
            comparisons = {c: dict(energy_saving=1-v['energy_j']/values['auto']['energy_j'],
                latency_change=v['window_s']/values['auto']['window_s']-1) for c, v in values.items() if c != 'auto'}
            result.update(status='compared', observed=values, comparisons=comparisons,
                incremental_value=(comparisons['policy']['latency_change'] <= .05
                    and comparisons['fixed']['latency_change'] <= .05
                    and values['policy']['energy_j'] < values['fixed']['energy_j']),
                interpretation='observed_means_not_statistical_significance_or_full_paper_reproduction',
                policy_setup_and_boundary_overheads_included_in_window=True)
    except BaseException as exc:
        result.update(status='failed', error=safe_error(exc)); raise
    finally:
        if holder['gate']: holder['gate'].abort('Comparison ended')
        server.close()
        if controller.dirty:
            try: controller.reset()
            except BaseException as exc: result['restore_error'] = safe_error(exc)
        result['restoration'] = controller.restore_result
        close_devices(devices, result)
        write_new(out/'comparison.json', result)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('preflight', 'serve'):
        cmd = sub.add_parser(name); cmd.add_argument('--run', required=True)
        cmd.add_argument('--model', default=client_api.MODEL)
    cmd = sub.add_parser('inspect'); cmd.add_argument('--run', required=True)
    cmd.add_argument('--base-url', default=client_api.BASE_URL); cmd.add_argument('--out', required=True)
    for name in ('observe', 'pilot', 'clock-check', 'switch-check', 'collect', 'serial', 'compare'):
        cmd = sub.add_parser(name); cmd.add_argument('--identity', required=True); cmd.add_argument('--out', required=True)
        if name != 'observe': cmd.add_argument('--telemetry', required=True)
        if name == 'observe': cmd.add_argument('--trace', action='store_true')
        if name in ('observe', 'collect', 'serial', 'compare'): cmd.add_argument('--seed', type=int, required=True)
        if name == 'pilot': cmd.add_argument('--requests', type=int, default=20)
        if name in ('clock-check', 'switch-check', 'collect', 'serial', 'compare'): cmd.add_argument('--apply', action='store_true')
        if name == 'clock-check': cmd.add_argument('--ack-default-auto', action='store_true')
        if name in ('switch-check', 'collect', 'serial', 'compare'): cmd.add_argument('--gate', required=True)
        if name == 'collect':
            cmd.add_argument('--pilot', required=True); cmd.add_argument('--split', choices=['train', 'validation'], required=True)
            cmd.add_argument('--model')
        if name == 'serial':
            cmd.add_argument('--plan', required=True)
            cmd.add_argument('--serial-evidence', help='Enable lookahead only with matching serial and fine transition evidence')
        if name == 'compare':
            for field in ('plan', 'serial-evidence', 'model', 'pilot'): cmd.add_argument('--'+field, required=True)
    cmd = sub.add_parser('fit'); cmd.add_argument('--data', required=True); cmd.add_argument('--out', required=True)
    cmd = sub.add_parser('validate'); cmd.add_argument('--data', required=True)
    cmd.add_argument('--model', required=True); cmd.add_argument('--out', required=True)
    cmd = sub.add_parser('paper-fit'); cmd.add_argument('--data', required=True); cmd.add_argument('--out', required=True)
    cmd = sub.add_parser('paper-validate'); cmd.add_argument('--data', required=True)
    cmd.add_argument('--model', required=True); cmd.add_argument('--out', required=True)
    cmd = sub.add_parser('plan')
    for field in ('model', 'validation', 'coverage', 'switching', 'temperature', 'out'):
        cmd.add_argument('--'+field, required=True)
    cmd = sub.add_parser('report'); cmd.add_argument('--file', required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    def stop(signum, frame): raise KeyboardInterrupt('Stop requested; run restoration finally blocks')
    if hasattr(signal, 'SIGTERM'): signal.signal(signal.SIGTERM, stop)
    lease = None
    try:
        name = args.command
        if hasattr(args, 'identity'):
            if os.name != 'posix': raise ValueError('Live operations run only in target WSL')
            import fcntl
            identity = read(args.identity)
            lease = (Path(identity['runtime'])/'client.lock').open('a')
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if name == 'preflight':
            value = preflight_record(args.model); write_new(Path(args.run)/'preflight.json', value)
        elif name == 'serve': return serve(args)
        elif name == 'inspect':
            value = inspect(args.run, args.base_url); write_new(args.out, value)
        elif name == 'pilot': value = asyncio.run(service.pilot(args, identity=checked(args.identity)))
        elif name == 'clock-check': value = asyncio.run(service.clock_check(args, identity=checked(args.identity)))
        elif name in ('observe', 'switch-check', 'collect', 'serial', 'compare'):
            value = asyncio.run({'observe': observe, 'switch-check': switch_check, 'collect': collect,
                                 'serial': serial, 'compare': compare}[name](args))
        elif name == 'fit':
            value = core.freeze_window_model(read(args.data)); write_new(args.out, value)
        elif name == 'validate':
            value = core.validate_window_model(read(args.data), read(args.model)); write_new(args.out, value)
        elif name in ('paper-fit', 'paper-validate', 'plan'):
            from . import eager_model
            if name == 'paper-fit': value = eager_model.fit(read(args.data))
            elif name == 'paper-validate': value = eager_model.validate(read(args.data), read(args.model))
            else:
                certificate = read(args.coverage)
                certificate = certificate.get('coverage', certificate)
                value = eager_model.plan(read(args.model), read(args.validation), certificate,
                    read(args.switching), read(args.temperature))
            write_new(args.out, value)
        else: value = read(args.file)
        service.emit({k: v for k, v in value.items() if k not in (
            'ranks', 'windows', 'weights', 'transitions', 'tools', 'operators', 'workloads', 'contexts',
            'train_prompt_hashes', 'training_prompts')})
        return 1 if value.get('status') in ('failed', 'not_ready') or value.get('passed') is False else 0
    except BaseException as exc:
        service.emit(dict(status='failed', error=safe_error(exc),
            instruction='Preserve output; inspect restoration before any further clock operation'))
        return 1
    finally:
        if lease: lease.close()


if __name__ == '__main__': sys.exit(main())
