"""Target-side orchestration. Importing this module never changes the host."""
import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from sparsedvfs.hardware import probe, read, integer, write_json, load_json, make_grid
from sparsedvfs.graph_partitioner import build_plan
from .contracts import IMAGE, IMAGE_TAG, IMAGE_DIGEST, REVISION, server_args, validate_plan, PROTOCOL, digest_json
from .http_client import completion
from .telemetry import Window

PACKAGE = Path(__file__).resolve().parents[1]
DERIVED_TAG = 'sparsedvfs-thor-v5:0.19.0-r38.2'


def run(argv, **kwargs):
    print('+ ' + ' '.join(map(str, argv)), flush=True)
    return subprocess.run(list(map(str, argv)), check=True, **kwargs)


def capture(argv):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        return dict(returncode=result.returncode, stdout=result.stdout.strip(), stderr=result.stderr.strip())
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'error': str(exc)}


def target_only():
    if platform.system() != 'Linux' or platform.machine() not in ('aarch64', 'arm64'):
        raise RuntimeError('Run this command on the Thor ARM64 Linux host; no current-host installation is performed')


def compatibility():
    bsp = read('/etc/nv_tegra_release', '')
    memory = read('/proc/meminfo', '')
    match = re.search(r'MemTotal:\s+(\d+)\s+kB', memory)
    return dict(bsp_r38_2=bool(re.search(r'R38\b', bsp) and re.search(r'REVISION:\s*2\.', bsp)),
                memory_at_least_110_gib=bool(match and int(match.group(1)) >= 110*1024*1024),
                model=read('/proc/device-tree/model', '').replace('\x00', ''))


def scenario(args):
    root = args.results / f'scenario_{args.input_tokens}_{args.output_tokens}'
    return root/'operator' if getattr(args, 'granularity', 'module') == 'operator' else root


def in_results(path, args):
    return '/results/' + Path(path).resolve().relative_to(args.results.resolve()).as_posix()


def paths(args):
    args.results, args.model_dir = Path(args.results).resolve(), Path(args.model_dir).resolve()
    args.results.mkdir(parents=True, exist_ok=True)
    scenario(args).mkdir(parents=True, exist_ok=True)


def doctor(args):
    report = dict(machine=platform.machine(), platform=platform.platform(),
                  model=read('/proc/device-tree/model', '').replace('\x00', ''),
                  meminfo=read('/proc/meminfo'), bsp=read('/etc/nv_tegra_release'),
                  os_release=read('/etc/os-release'),
                  disk_free_bytes=shutil.disk_usage(args.results).free,
                  compatibility=compatibility(),
                  docker=capture(['docker', 'version']),
                  docker_info=capture(['docker', 'info', '--format', '{{json .}}']),
                  nvidia_runtime=capture(['nvidia-container-cli', '--version']),
                  cuda=capture(['nvcc', '--version']), nvpmodel=capture(['nvpmodel', '-q']))
    try:
        report['hardware'] = probe()
    except Exception as exc:
        report['frequency_probe_error'] = str(exc)
    report['actions'] = [
        'Check T5000/128GB and Jetson Linux R38.2 against the locked image; no flashing/nvpmodel change is automatic.',
        'If GPC nodes are missing before CUDA initialization, run prepare then smoke, then doctor again.',
        'Install a BSP-compatible Docker Engine/NVIDIA Container Toolkit manually only if absent; see Quickstart links.']
    write_json(args.results / 'doctor.json', report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def software(args):
    saved = load_json(args.results / 'prepare.json')
    if saved.get('adapter_version') != 5:
        raise RuntimeError('This is not a v5 prepared image; keep old results separate and run v5 prepare on Thor')
    if saved['base_image'] != IMAGE or saved['model_revision'] != REVISION:
        raise RuntimeError('Preparation metadata does not match the locked software')
    return saved['derived_image_id']


def docker_cmd(args, *, apply=False, model=True, mounts=True, name=None):
    command = ['docker', 'run', '--rm', '--runtime=nvidia', '--network=host', '--user=0:0',
               '--shm-size=16g', '--ulimit=memlock=-1', '--ulimit=stack=67108864',
               '--env=VLLM_USE_V2_MODEL_RUNNER=0', '--env=HF_HUB_OFFLINE=1',
               '--env=SPARSEDVFS_IMAGE_DIGEST=' + IMAGE_DIGEST,
               '--mount', f'type=bind,src={args.results},dst=/results']
    if name:
        command += ['--name', name]
    if model:
        command += ['--mount', f'type=bind,src={args.model_dir},dst=/model,readonly']
    bindings_path = getattr(args, 'recovery_bindings', args.results / 'mounts.json')
    if mounts and bindings_path.exists():
        bindings = load_json(bindings_path)['bindings']
        for b in bindings:
            rw = apply and b['control']
            command += ['--mount', f"type=bind,src={b['host']},dst={b['container']}" + ('' if rw else ',readonly')]
    if apply:
        command += ['--env=SPARSEDVFS_APPLY=1']
        if os.geteuid() != 0:
            raise RuntimeError('Use sudo on Thor for --apply (BPMP discovery and shared lock)')
        info = capture(['docker', 'info', '--format', '{{json .SecurityOptions}}'])
        if 'rootless' in info.get('stdout', ''):
            raise RuntimeError('Direct frequency writes require rootful Docker')
        lock = Path('/run/lock/sparsedvfs-thor.lock')
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.touch(exist_ok=True)
        command += ['--mount', f'type=bind,src={lock},dst={lock}']
    return command


def prepare(args):
    target_only()
    checks = compatibility()
    if not checks['bsp_r38_2'] or not checks['memory_at_least_110_gib']:
        raise RuntimeError('This lock targets R38.2 and T5000/128GB. Doctor found a different BSP/memory; review compatibility before preparing; no host change is automatic.')
    if not shutil.which('docker'):
        raise RuntimeError('Docker is absent. Follow the official installation links in THOR_VLLM_QUICKSTART_zh.md; host setup is manual.')
    run(['docker', 'pull', '--platform=linux/arm64', IMAGE])
    run(['docker', 'build', '--platform=linux/arm64', '-f', PACKAGE / 'container/Dockerfile',
         '-t', DERIVED_TAG, PACKAGE])
    image_id = subprocess.check_output(['docker', 'image', 'inspect', '--format={{.Id}}', DERIVED_TAG], text=True).strip()
    digests = subprocess.check_output(['docker', 'image', 'inspect', '--format={{json .RepoDigests}}', IMAGE], text=True).strip()
    if IMAGE_DIGEST not in digests:
        raise RuntimeError('Pulled image digest mismatch')
    write_json(args.results / 'prepare.json', dict(adapter_version=5, base_image=IMAGE, source_tag=IMAGE_TAG,
               derived_image_id=image_id, model_revision=REVISION,
               derived_inspect=json.loads(subprocess.check_output(['docker', 'inspect', image_id], text=True))))
    run(docker_cmd(args, model=False, mounts=False) + [image_id, 'python3', '-B', '-m',
                                                     'sparsedvfs_vllm.runtime_info'])
    args.model_dir.mkdir(parents=True, exist_ok=True)
    cmd = docker_cmd(args, model=False, mounts=False)
    # Download is the only stage mounting model files writable; no host pip/cuda setup.
    cmd = [x for x in cmd if x != '--env=HF_HUB_OFFLINE=1']
    cmd += ['--mount', f'type=bind,src={args.model_dir},dst=/model', image_id,
            'python3', '-B', '-m', 'sparsedvfs_vllm.model_prepare']
    run(cmd)
    print('Prepared pinned image and hashed model. Run doctor/selftest/smoke next.')


def replace_paths(value, replacements):
    if isinstance(value, dict):
        return {k: replace_paths(v, replacements) for k, v in value.items()}
    if isinstance(value, list):
        return [replace_paths(v, replacements) for v in value]
    if isinstance(value, str):
        for source, dest in sorted(replacements, key=lambda pair: -len(pair[0])):
            if value == source or value.startswith(source + '/'):
                return dest + value[len(source):]
    return value


def hardware(args, *, grid=None):
    """Bind resolved control directories outside /sys; do not disable AppArmor."""
    evidence = load_json(args.emc_evidence) if args.emc_evidence else None
    prior = load_json(args.results / 'hardware_host.json') if (args.results / 'hardware_host.json').exists() else None
    if not evidence and prior:
        evidence = prior['emc'].get('evidence')
    rates = evidence['rates_hz'] if evidence else []
    hw = probe(emc_rates=rates)
    sensor_config = load_json(args.sensor_config) if getattr(args, 'sensor_config', None) else None
    if sensor_config:
        for node in sensor_config.get('read_only_observations', []):
            if not node.get('evidence') or not str(node.get('path', '')).startswith('/sys/'):
                raise ValueError('Observation sensors need a /sys path and documented unit/scale evidence')
            if node.get('kind') == 'voltage' and (not isinstance(node.get('scale_to_v'), (int,float)) or node['scale_to_v'] <= 0):
                raise ValueError('Voltage sensors require a verified positive scale_to_v')
        hw['read_only_observations'] = sensor_config.get('read_only_observations', [])
        hw['calibration_scope'] = sensor_config.get('calibration_scope', {})
    elif prior:
        hw['read_only_observations'] = prior.get('read_only_observations', hw.get('read_only_observations', []))
        hw['calibration_scope'] = prior.get('calibration_scope', {})
    if evidence:
        if (not evidence.get('source') or evidence.get('device_model') != hw['device_model'] or
                evidence.get('jetson_release') != hw['jetson_release'] or
                not evidence.get('verified_discrete_opps')):
            raise ValueError('EMC evidence must identify this model/BSP and verified discrete OPPs')
        hw['emc']['rates_source'] = 'explicit_verified_evidence'
        hw['emc']['evidence'] = evidence
    if not hw['emc']['frequencies']:
        if grid == 'full':
            raise RuntimeError('Full three-domain profiling needs >=2 verified discrete EMC rates; no table was found')
        current = integer(hw['emc']['current_path'])
        lo, hi = hw['emc']['min_rate'], hw['emc']['max_rate']
        if not current or lo is None or hi is None or not lo <= current <= hi:
            raise RuntimeError('Cannot establish a valid current EMC rate even for quick profiling')
        hw['emc']['frequencies'] = [current]
        hw['emc']['rates_source'] = 'quick_current_readback_not_discrete_table'
    if grid == 'quick':
        current = integer(hw['emc']['current_path'])
        if not current or current < 0 or (hw['emc']['min_rate'] is not None and current < hw['emc']['min_rate']) or (hw['emc']['max_rate'] is not None and current > hw['emc']['max_rate']):
            raise RuntimeError('Current EMC readback is not a positive rate within its legal bounds')
        if current not in hw['emc']['frequencies'] and hw['emc']['rates_source'] != 'quick_current_readback_not_discrete_table':
            raise RuntimeError('Current EMC rate is absent from the verified legal table')
        hw['emc']['frequencies'] = [current]
    if grid == 'full' and len(hw['emc']['frequencies']) < 2:
        raise RuntimeError('Full profiling needs at least two verified EMC frequencies')
    if any(f <= 0 for f in hw['cpu_common_khz'] + hw['gpu']['frequencies'] + hw['emc']['frequencies']):
        raise ValueError('CPU/GPC/EMC profiling requires strictly positive discrete frequencies')
    if grid and (len(hw['cpu_common_khz']) < 2 or len(hw['gpu']['frequencies']) < 2):
        raise RuntimeError('At least two legal CPU and GPC OPPs are required')
    if args.power_source:
        selected = load_json(args.power_source)
        if not selected.get('scope') or selected['scope'] == 'unclassified':
            raise ValueError('An explicit power source must document its measurement scope')
        hw['power'] = selected
        hw['power_selection_source'] = 'explicit_verified'
    elif prior and prior.get('power_selection_source') == 'explicit_verified':
        hw['power'] = prior['power']
        hw['power_selection_source'] = 'explicit_verified'
    bindings, replacements = [], []
    def bind(source, dest, control):
        resolved = str(Path(source).resolve())
        bindings.append(dict(host=resolved, container=dest, control=control))
        replacements.extend([(str(source), dest), (resolved, dest)])
    for cpu in hw['cpu']:
        bind(cpu['path'], '/host-dvfs/cpu/' + Path(cpu['path']).name, True)
    bind(hw['gpu']['path'], '/host-dvfs/gpu', True)
    bind(hw['emc']['path'], '/host-dvfs/emc', True)
    bind(str(Path(hw['emc']['halt_path']).parent), '/host-dvfs/bwmgr', True)
    sensor_dirs = set()
    for p in hw['power_sources'] + ([hw['power']] if hw['power'] else []):
        for key in ('path', 'voltage_path', 'current_path'):
            if p.get(key):
                sensor_dirs.add(str(Path(p[key]).parent))
    for p in hw['thermal']:
        sensor_dirs.add(str(Path(p['path']).parent))
    for p in hw.get('read_only_observations', []):
        sensor_dirs.add(str(Path(p['path']).parent))
    for i, folder in enumerate(sorted(sensor_dirs)):
        bind(folder, f'/host-sensors/{i}', False)
    write_json(args.results / 'hardware_host.json', hw)
    write_json(args.results / 'hardware.json', replace_paths(hw, replacements))
    write_json(args.results / 'mounts.json', dict(bindings=bindings, replacements=replacements))
    return hw


def inside(args, action, extra=(), *, apply=False, mounts=True):
    cmd = docker_cmd(args, apply=apply, mounts=mounts) + [software(args), 'python3', '-B', '-m',
          'sparsedvfs_vllm.inside', action, '--run-dir', in_results(scenario(args), args),
          '--input-tokens', str(args.input_tokens), '--output-tokens', str(args.output_tokens)]
    cmd += ['--granularity', args.granularity, '--backend', args.backend,
            '--switch-repeat', str(args.switch_repeat)]
    coverage = scenario(args)/'coverage.json'
    if args.granularity == 'operator' and coverage.exists() and action != 'trace':
        cmd += ['--coverage', in_results(coverage, args)]
    if mounts:
        cmd += ['--hardware', '/results/hardware.json']
    run(cmd + list(extra))


def wait_health(proc, port, timeout=1800):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError('vLLM server exited before health check; inspect server.log and OOM report')
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(1)
    raise RuntimeError('Server readiness timeout; inspect server.log')


def start_server(args, out, *, mode='native', apply=False, production=False, control=False):
    name = 'sparsedvfs-v5-' + str(time.time_ns())
    cmd = docker_cmd(args, apply=apply, name=name, mounts=not production)
    if mode != 'native':
        data = load_json(scenario(args) / 'workload.json')
        from .inside import public_workload
        config = dict(mode=mode, enabled=True, apply=apply, hardware='/results/hardware.json',
                      granularity=args.granularity,
                      coverage=in_results(scenario(args)/'coverage.json', args) if args.granularity == 'operator' else None,
                      out_dir=in_results(out / 'worker', args), workload=public_workload(data),
                      plan=in_results(scenario(args) / 'plan.json', args),
                      fixed_triplet=load_json(scenario(args) / 'profile.json')['reference']
                      if mode == 'fixed' else None)
        config['journal_during_request'] = not control
        if control:
            gate = out / 'control.json'
            write_json(gate, dict(enabled=False, run_id='warmup'))
            config['control_file'] = in_results(gate, args)
        config_file = out / 'worker_config.json'
        write_json(config_file, config)
        cmd += ['--env=SPARSEDVFS_WORKER_CONFIG=' + in_results(config_file, args)]
    cmd += [software(args), 'vllm'] + server_args(custom=mode != 'native', production=production, port=args.port)
    write_json(out / 'launch.json', dict(command=cmd, mode=mode, production=production, apply=apply,
                                        image_id=software(args), bsp=read('/etc/nv_tegra_release')))
    log = (out / 'server.log').open('w', encoding='utf-8')
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    return proc, name, log


def stop_server(proc, name, log, args=None, out=None):
    handshake = 0
    if args is not None and out is not None and (out / 'worker_config.json').exists():
        try:
            handshake = subprocess.run(['docker', 'exec', name, 'python3', '-B', '-m', 'sparsedvfs_vllm.control',
                          '--worker-dir', in_results(out / 'worker', args)], timeout=90).returncode
        except subprocess.TimeoutExpired:
            handshake = 1
    # Ask Docker to signal PID 1; wait for worker/controller restoration.
    result = subprocess.run(['docker', 'stop', '--time=90', name], capture_output=True, text=True)
    try:
        proc.wait(timeout=100)
    except subprocess.TimeoutExpired:
        proc.terminate()
        proc.wait(timeout=10)
    finally:
        log.close()
    return handshake or result.returncode


def atomic_gate(path, enabled, run_id):
    temp = path.with_suffix('.tmp')
    write_json(temp, dict(enabled=enabled, run_id=run_id))
    temp.replace(path)


def service_experiment(args, *, smoke=False):
    hw = hardware(args) if not smoke else None
    inside(args, 'workload', mounts=False)
    data = load_json(scenario(args) / 'workload.json')
    out = scenario(args) / (('smoke_' if smoke else 'bench_') + str(time.time_ns()))
    out.mkdir()
    modes = ['native'] if smoke else ['native', 'instrumented', 'fixed', 'serial', 'lookahead']
    if not smoke:
        if not args.apply:
            raise ValueError('Five-way hardware comparison requires --apply; use trace/selftest for no-write verification')
        plan = load_json(scenario(args) / 'plan.json')
        validate_plan(plan)
        if plan.get('profile_sha256') != digest_json(load_json(scenario(args) / 'profile.json')):
            raise ValueError('Profile changed after plan generation; run plan again')
        trace = load_json(scenario(args) / 'trace.json')
        if trace['worker_trace']['identity'] != plan['identity']:
            raise ValueError('Trace and plan identities differ; rerun trace/profile for this workload/image')
    results = []
    try:
        for mode in modes:
            mode_dir = out / mode
            mode_dir.mkdir()
            proc, name, log = start_server(args, mode_dir, mode=mode,
                                          apply=args.apply and mode in ('fixed', 'serial', 'lookahead'), control=True)
            try:
                wait_health(proc, args.port)
                if hw is None:
                    try:
                        hw = hardware(args)
                    except Exception:
                        from sparsedvfs.hardware import _power_sources
                        _, source = _power_sources(Path('/sys'))
                        hw = dict(cpu=[], gpu={'current_path': '/nonexistent'}, emc={},
                                  power=source, thermal=[])
                for _ in range(2):
                    completion(f'http://127.0.0.1:{args.port}', data)
                if mode != 'native':
                    run(['docker', 'exec', name, 'python3', '-B', '-m', 'sparsedvfs_vllm.control',
                         '--worker-dir', in_results(mode_dir / 'worker', args), '--action', 'prepare'])
                for index in range(1 if smoke else args.repeat):
                    if mode != 'native':
                        atomic_gate(mode_dir / 'control.json', True, f'{mode}_{index}')
                    window = Window(hw).start()
                    try:
                        client = completion(f'http://127.0.0.1:{args.port}', data)
                    finally:
                        measured = window.stop()
                    row = dict(mode=mode, repeat=index, client=client, measurement=measured,
                               workload={k: v for k, v in data.items() if k != 'input_token_ids'},
                               image_id=software(args), model_revision=REVISION,
                               run_dir=str(mode_dir), production=False)
                    results.append(row)
                    write_json(mode_dir / f'request_{index}.json', row)
            finally:
                closed = stop_server(proc, name, log, args, mode_dir)
                if closed:
                    raise RuntimeError(f'{mode}: graceful shutdown failed; inspect restore_status.json/snapshot')
            if mode in ('fixed', 'serial', 'lookahead'):
                status = mode_dir / 'worker/controller/restore_status.json'
                if not status.exists() or not load_json(status)['restored']:
                    raise RuntimeError(f'{mode}: missing/failed restoration readback; run restore before continuing')
        hashes = {r['client']['output_sha256'] for r in results}
        summary = dict(comparable=len(hashes) == 1, results=results, run_dir=str(out),
                       native_configuration='eager identical limits without the custom worker')
        write_json(out / 'results.json', summary)
        write_json(scenario(args) / ('smoke_latest.json' if smoke else 'bench_latest.json'), summary)
        if len(hashes) != 1:
            raise RuntimeError('Outputs differ across modes/repeats; results are not comparable')
    except BaseException as exc:
        write_json(out / 'failure.json', dict(error=repr(exc), partial_results=results,
                   memory=read('/proc/meminfo'), action='Stopped; inspect server.log; no model/precision fallback'))
        raise


def plan_command(args):
    profile = load_json(scenario(args) / 'profile.json')
    if profile.get('identity', {}).get('backend') != PROTOCOL:
        raise ValueError('Refusing a v2/Transformers profile')
    if profile.get('schema') != 5 or profile.get('granularity') != args.granularity:
        raise ValueError('Regenerate a v5 profile at the selected granularity')
    if args.selector is None or args.max_slowdown is None or args.similarity_tolerance is None:
        raise ValueError('Supply --selector, --max-slowdown and --similarity-tolerance explicitly; the paper supplies no 5% defaults')
    calibration = load_json(args.calibration) if args.calibration else None
    if args.selector == 'paper' and (not calibration or calibration.get('schema') != 4):
        raise ValueError('--selector paper requires measured schema-4 --calibration; null templates and legacy per-OPP alpha/k1/k2 are not deployable')
    if args.selector == 'paper' and (not calibration.get('v5_validation', {}).get('deployable') or
                                    calibration.get('profile_sha256') != digest_json(profile)):
        raise ValueError('Strict paper deployment requires real fitted, held-out validated coefficients for this profile')
    if calibration and any(spec['s_comp'] != 0 or spec['s_mem'] != 0 for spec in calibration['operators'].values()):
        raise ValueError('Dense BF16 v4 has no verified skip/compression path; require S_comp=S_mem=0')
    plan = build_plan(profile, profile['switch_us'], factor=args.factor, tolerance=args.similarity_tolerance,
                      max_slowdown=args.max_slowdown, selector=args.selector,
                      calibration=calibration, memory_policy='paper' if args.selector == 'paper' else 'calibrated')
    plan['schema'], plan['granularity'] = 5, args.granularity
    if args.granularity == 'operator':
        plan['coverage_certificate'] = profile['coverage_certificate']
        plan['opp_coverage_certificates'] = profile['opp_coverage_certificates']
    plan['runtime_granularity'] = ('certified_dispatcher_units_with_atomic_fusion' if args.granularity == 'operator'
                                   else 'vllm_embedding_decoder_norm_compute_logits_sample')
    if args.granularity == 'operator':
        plan['graph_scope'] = 'certified dispatcher units serialized on one observed main stream; fused units atomic'
        plan['limitations'] = [item for item in plan['limitations'] if not item.startswith('decoder modules')]
        plan['limitations'].append('Dispatcher/fused units are not a recovered CUDA kernel DAG; hidden future stream changes are unproven')
    plan['fidelity'] = dict(equations='schema4 calibrated Eq1/Eq2' if args.selector == 'paper' else 'empirical mechanism path',
                           algorithm1='shared literal OR/max/tail core with caller similarity',
                           graph=plan['runtime_granularity'],
                           native='explicit stream/sysfs reference, not author register queue',
                           paper_experiments_reproduced=False)
    plan['profile_run_dir'] = profile['run_dir']
    plan['profile_file'] = in_results(scenario(args) / 'profile.json', args)
    plan['profile_sha256'] = digest_json(profile)
    if calibration:
        plan['calibration'] = calibration
    plan['emc_grid'] = profile['grid']['kind']
    validate_plan(plan)
    write_json(scenario(args) / 'plan.json', plan)
    print(f"Wrote {len(plan['blocks'])} blocks; "
          f"{sum(not b['amortized_strict_gt'] for b in plan['blocks'])} fail strict N*Tswitch amortization.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['doctor', 'prepare', 'selftest', 'smoke', 'trace', 'profile',
                                     'plan', 'bench', 'report', 'serve', 'restore', 'offline', 'calibrate'])
    p.add_argument('--granularity', choices=['module', 'operator'], default='module')
    p.add_argument('--backend', choices=['torch'], default='torch')
    p.add_argument('--switch-repeat', type=int, default=10)
    p.add_argument('--sensor-config')
    p.add_argument('--stage', choices=['collect', 'fit', 'validate'])
    p.add_argument('--observations')
    p.add_argument('--workloads')
    p.add_argument('--output')
    p.add_argument('--max-relative-error', type=float)
    p.add_argument('--results', default=str(PACKAGE / 'runs'))
    p.add_argument('--model-dir', default=str(PACKAGE / 'models/Qwen3.5-27B'))
    p.add_argument('--input-tokens', type=int, default=512)
    p.add_argument('--output-tokens', type=int, default=128)
    p.add_argument('--grid', choices=['quick', 'full'], default='quick')
    p.add_argument('--points', type=int, default=2)
    p.add_argument('--repeat', type=int, default=5)
    p.add_argument('--emc-evidence')
    p.add_argument('--power-source')
    p.add_argument('--factor', type=float, default=5)
    p.add_argument('--selector', choices=['measured', 'paper'], help='Required for plan; measured is an empirical mechanism path')
    p.add_argument('--max-slowdown', type=float, help='Required for plan; user-chosen per-unit latency allowance, not an author default')
    p.add_argument('--similarity-tolerance', type=float, help='Required for plan; relative Linf choice, 0 means exact equality')
    p.add_argument('--calibration')
    p.add_argument('--apply', action='store_true')
    p.add_argument('--mode', choices=['native', 'instrumented', 'fixed', 'serial', 'lookahead'], default='lookahead')
    p.add_argument('--profile', choices=['repro', 'production-native'], default='repro')
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--snapshot')
    args = p.parse_args()
    if args.input_tokens < 1 or args.output_tokens < 2 or args.input_tokens + args.output_tokens > 4096:
        p.error('Require input>=1, output>=2, input+output<=4096')
    if args.repeat < 1 or args.points < 2 or args.switch_repeat < 1:
        p.error('Require repeat>=1, points>=2')
    if args.command == 'trace' and args.apply:
        p.error('trace is a no-write diagnostic; do not pass --apply')
    paths(args)
    if args.command == 'calibrate' and args.stage in ('fit', 'validate'):
        from .calibration import command
        command(args, scenario(args))
        return
    if args.command == 'doctor':
        doctor(args)
        return
    if args.command == 'plan':
        plan_command(args)
        return
    if args.command == 'report':
        from .report import report
        report(scenario(args))
        return
    target_only()
    if args.command == 'prepare':
        prepare(args)
    elif args.command == 'selftest':
        with (args.results / 'selftest.log').open('w', encoding='utf-8') as log:
            run(docker_cmd(args, model=False, mounts=False) + [software(args), 'bash', 'scripts/selftest_v5.sh'],
                stdout=log, stderr=subprocess.STDOUT)
        print((args.results / 'selftest.log').read_text())
    elif args.command == 'smoke':
        service_experiment(args, smoke=True)
    elif args.command in ('trace', 'offline'):
        hardware(args)
        if args.command == 'trace' and args.granularity == 'operator':
            old = scenario(args)/'coverage.json'
            if old.exists():
                old.replace(old.with_name('coverage_obsolete_'+str(time.time_ns())+'.json'))
        extra = []
        if args.command == 'offline':
            if args.mode == 'native':
                raise ValueError('Offline modes are instrumented/fixed/serial/lookahead; smoke/bench provide the true native worker baseline')
            mode = args.mode
            extra += ['--mode', mode, '--plan', in_results(scenario(args) / 'plan.json', args)]
            if mode == 'fixed':
                extra += ['--grid-file', in_results(scenario(args) / 'grid.json', args)]
            if args.apply:
                extra += ['--apply']
        inside(args, args.command, extra, apply=args.apply)
    elif args.command == 'profile' or (args.command == 'calibrate' and args.stage == 'collect'):
        if not args.apply:
            raise ValueError('profile requires --apply; trace/selftest are the no-write checks')
        if args.granularity == 'operator' and not (scenario(args)/'coverage.json').exists():
            raise ValueError('Run trace --granularity operator before applied profiling')
        hw = hardware(args, grid=args.grid)
        grid = make_grid(hw, args.points)
        grid['kind'] = args.grid
        write_json(scenario(args) / 'grid.json', grid)
        inside(args, 'profile', ['--grid-file', in_results(scenario(args) / 'grid.json', args),
                               '--repeat', str(args.repeat), '--apply'], apply=True)
        if args.command == 'calibrate':
            from .calibration import command
            command(args, scenario(args))
    elif args.command == 'calibrate':
        raise ValueError('calibrate requires --stage collect|fit|validate')
    elif args.command == 'bench':
        service_experiment(args)
    elif args.command == 'serve':
        production = args.profile == 'production-native'
        mode = 'native' if production else args.mode
        if production and args.apply:
            raise ValueError('production-native does not use DVFS')
        if not production:
            hardware(args)
        inside(args, 'workload', mounts=False)
        out = scenario(args) / ('production_native_' if production else 'serve_')
        out = out.with_name(out.name + str(time.time_ns()))
        out.mkdir()
        proc, name, log = start_server(args, out, mode=mode, apply=args.apply, production=production)
        try:
            wait_health(proc, args.port)
            print(f'Ready on port {args.port}; Ctrl+C requests restoration. Logs: {out}', flush=True)
            proc.wait()
        finally:
            closed = stop_server(proc, name, log, args, out)
            if closed:
                raise RuntimeError('Service shutdown failed; inspect saved snapshot and run explicit restore')
    elif args.command == 'restore':
        if not args.snapshot:
            p.error('restore requires --snapshot')
        args.recovery_bindings = Path(args.snapshot).resolve().with_name('mounts.json')
        if not args.recovery_bindings.exists():
            raise ValueError('Matching mounts.json is missing beside the saved clock snapshot')
        run(docker_cmd(args, apply=args.apply, model=False) + [software(args), 'python3', '-B', '-m',
            'sparsedvfs_vllm.restore', '--snapshot', in_results(args.snapshot, args)] + (['--apply'] if args.apply else []))


if __name__ == '__main__':
    main()
