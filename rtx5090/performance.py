"""Standalone Eq. (1): isolated BF16 operators, no vLLM/FUSE/power fit.

Paper: arXiv:2603.21908v1, section 4.1.1. Logical tensor traffic and the
capacity/overhead estimators below are declared port choices, not author data.
Torch is imported only by doctor/collect; fit/validate use the standard library.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import sys
import time

from sparsedvfs.paper_core import timeline_seconds
from .backends import backend
from .native import DEFAULT_GPU, assert_cuda_uuid


SCHEMA = 1
AUTHOR_COMMIT = 'c524c51aac552fd16bec5743b1fb95309b46347b'
TRAFFIC = 'logical_tensor_bytes_not_measured_DRAM'
TIMING = 'synchronized_host_wall_s; separate_CUDA_event_capacity_pass'
POLICY = 'one_global_nonnegative_median_training_residual'
CLOCK_CHECK_POLICY = 'phase_readbacks_v2_optional_bounded_precheck'
PRECHECK_PHASES = {'before_wall', 'before_cuda'}
CHECK_PHASES = PRECHECK_PHASES | {'after_wall', 'after_cuda'}
STABLE_READS = 3
POLL_SECONDS = 0.1
MIB = 1024 * 1024
PARAMETER_KEYS = {'p_peak_flops_by_gpu_mhz', 'b_mem_logical_Bps',
                  't_overhead_s', 's_comp', 's_mem'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode('utf-8')).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    """Replace only this run's own artifact; callers refuse existing runs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def fresh(out, *names):
    out = Path(out)
    for name in names:
        for path in (out / name, out / (name + '.tmp')):
            if path.exists():
                raise FileExistsError(f'Evidence already exists: {path}; use a fresh --out')
    out.mkdir(parents=True, exist_ok=True)
    return out


def redact(message):
    token = os.environ.get('SPARSEDVFS_BRIDGE_TOKEN')
    return message.replace(token, '[REDACTED]') if token else message


def safe_error(exc):
    return redact(f'{type(exc).__name__}: {exc}')


def number(value, name, zero=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or (value < 0 if zero else value <= 0)):
        raise ValueError(f'{name} must be a finite {"nonnegative" if zero else "positive"} number')
    return value


def cases():
    result = []
    groups = [('capacity', (8192,), (256,)),
              ('calibration', (1024, 2048, 4096), (8, 64)),
              ('validation', (1536, 3072, 6144), (16, 128))]
    for role, matrices, buffers in groups:
        for n in matrices:
            result.append(dict(case_id=f'gemm_{n}', role=role, operator='gemm',
                               shape=[n, n, n], dtype='bfloat16', element_bytes=2,
                               work_flop=2*n*n*n, data_byte=2*(n*n+n*n+n*n),
                               s_comp=0.0, s_mem=0.0, traffic_semantics=TRAFFIC))
        for size in buffers:
            count = size*MIB//2
            result.append(dict(case_id=f'add_{size}mib', role=role, operator='add',
                               shape=[count], buffer_bytes=size*MIB,
                               dtype='bfloat16', element_bytes=2,
                               work_flop=count, data_byte=3*count*2,
                               s_comp=0.0, s_mem=0.0, traffic_semantics=TRAFFIC))
    return result


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    return {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
            for name in ('rtx5090/performance.py', 'rtx5090/backends.py',
                         'sparsedvfs/paper_core.py')}


def settings(args):
    if not re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', args.gpu_uuid):
        raise ValueError('Use one full GPU UUID')
    if not args.ack_default_auto:
        raise ValueError('Acknowledge initial automatic clocks with --ack-default-auto')
    rates = args.rates_mhz
    if (len(rates) < 2 or len(set(rates)) != len(rates) or
            any(type(rate) is not int or rate <= 0 for rate in rates)):
        raise ValueError('Use at least two distinct positive integer MHz values')
    if args.reference_mhz not in rates:
        raise ValueError('--reference-mhz must be in --rates-mhz')
    if args.warmup < 1 or args.repeat < 1:
        raise ValueError('--warmup and --repeat must be positive')
    number(args.clock_stability_timeout_s, '--clock-stability-timeout-s', zero=True)
    return dict(backend=args.backend, uuid=args.gpu_uuid, rates_mhz=list(rates),
                default_auto_ack=True, host=args.bridge_host, port=args.bridge_port)


def protocol(args):
    return dict(rates_mhz=list(args.rates_mhz), reference_mhz=args.reference_mhz,
                warmup=args.warmup, repeat=args.repeat, settle_s=0.5,
                core_tolerance_mhz=15, traffic_semantics=TRAFFIC, timing=TIMING,
                dtype='bfloat16', cpu_policy='unchanged_system_managed',
                memory_policy='unchanged_system_managed; require_one_observed_clock',
                scope='single_GPU_core_frequency_slice', seed=0,
                author_commit=AUTHOR_COMMIT,
                author_reference_defaults=(args.warmup == 5 and args.repeat == 5),
                clock_check_policy=CLOCK_CHECK_POLICY,
                clock_stability=dict(timeout_s=args.clock_stability_timeout_s,
                    required_consecutive_reads=STABLE_READS if args.clock_stability_timeout_s else 1,
                    interval_s=POLL_SECONDS if args.clock_stability_timeout_s else 0,
                    extra_operation_before_recheck=bool(args.clock_stability_timeout_s),
                    scope='pre_measurement_only; post_measurement_single_strict_read',
                    author_flow_deviation=bool(args.clock_stability_timeout_s)))


def capabilities(device, expected, rates):
    value = device.probe()
    if value.get('uuid', '').lower() != expected.lower():
        raise ValueError('Backend UUID differs from selected CUDA GPU')
    if device.synthetic or value.get('synthetic') is not False:
        raise ValueError('Synthetic backend cannot collect real Eq1 data')
    if not set(rates) <= set(value.get('rates_mhz', [])):
        raise ValueError('Requested clocks outside reported grid; no automatic substitution')
    return value


class ClockReadbackError(RuntimeError):
    def __init__(self, message, kind, snapshot):
        super().__init__(message)
        self.kind, self.snapshot = kind, snapshot


class ClockGuard:
    def __init__(self, device, uuid, memory_mhz=None, *, now_ns=time.monotonic_ns):
        self.device, self.uuid, self.memory_mhz = device, uuid, memory_mhz
        self.now_ns, self.last_snapshot = now_ns, None

    def observe(self, requested=None, *, phase='idle', last_sync_ns=None):
        # Caller and backend monotonic clocks are different in WSL/Windows.
        snapshot = dict(phase=phase, requested_mhz=requested, passed=False,
                        client_query_start_ns=self.now_ns(), client_last_sync_ns=last_sync_ns,
                        client_clock_domain='collector_monotonic',
                        backend_clock_domain='backend_monotonic_not_comparable_to_client')
        self.last_snapshot = snapshot
        if last_sync_ns is not None:
            snapshot['sync_to_query_start_s'] = (snapshot['client_query_start_ns']-last_sync_ns)/1e9
        kind = 'telemetry_error'
        try:
            try:
                reading = self.device.telemetry()
            finally:
                snapshot['client_query_end_ns'] = self.now_ns()
                snapshot['query_roundtrip_s'] = (snapshot['client_query_end_ns']-
                                                snapshot['client_query_start_ns'])/1e9
            # Only diagnostic fields are copied, never credentials or arbitrary replies.
            for key in ('uuid', 'synthetic', 'timestamp_ns', 'clock_mhz', 'sm_clock_mhz',
                        'memory_clock_mhz', 'temperature_c', 'power_mw', 'power_limit_mw',
                        'performance_state', 'utilization_pct', 'clock_event_reasons',
                        'clock_event_reasons_api', 'field_query_times_ns'):
                if key in reading:
                    snapshot[key] = reading[key]
            snapshot['errors'] = {str(k): redact(str(v)) for k, v in reading.get('errors', {}).items()}
            kind = 'identity_error'
            if reading.get('uuid', '').lower() != self.uuid.lower() or reading.get('synthetic') is not False:
                raise ValueError('Telemetry UUID/synthetic status differs from real selected GPU')
            kind = 'missing_readback'
            core = number(reading.get('clock_mhz'), 'GPU clock readback')
            memory = reading.get('memory_clock_mhz')
            if memory is None:
                raise RuntimeError('Memory clock unavailable; update Windows code and restart the bridge')
            number(memory, 'memory clock readback')
            if requested is not None:
                kind = 'memory_changed'
                if self.memory_mhz is None:
                    self.memory_mhz = memory
                elif self.memory_mhz != memory:
                    raise RuntimeError(f'Memory clock changed: {self.memory_mhz} -> {memory} MHz')
                kind = 'core_mismatch'
                if abs(core-requested) > 15:
                    raise RuntimeError(f'Clock mismatch: requested {requested}, observed {core} MHz')
            snapshot['passed'] = True
        except Exception as exc:
            snapshot.update(error=safe_error(exc), error_kind=kind)
            raise ClockReadbackError(str(exc), kind, snapshot) from exc
        return snapshot


class ClockChecks:
    """Preserve every read, including failures; retries never repair a postcheck.

    A positive budget is an explicit port choice. Extra same-shape operations
    are outside both timing passes. Individual CUDA/driver calls cannot be
    preempted, but late results cannot make the check pass.
    """
    def __init__(self, guard, row, timeout_s=0, *, sleep=time.sleep, now_ns=time.monotonic_ns):
        self.guard, self.row, self.timeout_s = guard, row, timeout_s
        self.sleep, self.now_ns = sleep, now_ns

    def __call__(self, phase, last_sync_ns, probe):
        if phase not in CHECK_PHASES:
            raise ValueError('Unknown clock checkpoint')
        row, started = self.row, self.now_ns()
        row['active_phase'] = phase
        bounded = self.timeout_s > 0 and phase in PRECHECK_PHASES
        deadline = started + int(self.timeout_s*1e9) if bounded else None
        required = STABLE_READS if bounded else 1
        check = dict(status='checking', started_client_ns=started, bounded=bounded,
                     required_consecutive_reads=required, attempts=0, extra_operations=0)
        row.setdefault('clock_checks', {})[phase] = check
        observations = row.setdefault('clock_observations', [])
        streak = 0
        try:
            while True:
                if check['attempts']:
                    remaining = (deadline-self.now_ns())/1e9
                    if remaining <= 0:
                        raise TimeoutError(f'Clock stability timeout at {phase}: {self.timeout_s:g} s')
                    self.sleep(min(POLL_SECONDS, remaining))
                    if self.now_ns() >= deadline:
                        raise TimeoutError(f'Clock stability timeout at {phase}: {self.timeout_s:g} s')
                    check['extra_operations'] += 1
                    last_sync_ns = probe()
                    if self.now_ns() >= deadline:
                        raise TimeoutError(f'Clock stability timeout after extra operation at {phase}')
                error = None
                try:
                    snapshot = self.guard.observe(row['requested_mhz'], phase=phase,
                                                  last_sync_ns=last_sync_ns)
                except ClockReadbackError as exc:
                    snapshot, error = exc.snapshot, exc
                check['attempts'] += 1
                snapshot['attempt'] = check['attempts']
                snapshot['within_deadline'] = deadline is None or self.now_ns() < deadline
                streak = streak+1 if snapshot['passed'] else 0
                snapshot['consecutive_matching_reads'] = streak
                observations.append(snapshot)
                row['snapshots'][phase] = snapshot  # Also retained on the throwing path.
                if error is not None and (not bounded or error.kind != 'core_mismatch'):
                    raise error
                if not snapshot['within_deadline']:
                    raise TimeoutError(f'Clock stability timeout during query at {phase}')
                if streak >= required:
                    check['status'] = 'passed'
                    return snapshot
        except BaseException as exc:
            check.update(status='failed', error=safe_error(exc))
            raise
        finally:
            check['ended_client_ns'] = self.now_ns()
            check['elapsed_s'] = (check['ended_client_ns']-started)/1e9


class TorchRuntime:
    """Single visible CUDA GPU; all allocations are outside timed regions."""
    real_cuda = True

    def __init__(self, uuid):
        if 'torch' in sys.modules and sys.modules['torch'].cuda.is_initialized():
            raise RuntimeError('Start a fresh process before selecting CUDA_VISIBLE_DEVICES')
        os.environ['CUDA_VISIBLE_DEVICES'] = uuid
        import torch
        self.torch = torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable in this interpreter')
        self.uuid = assert_cuda_uuid(torch, uuid)
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError('Selected device does not support BF16')
        self.properties = torch.cuda.get_device_properties(0)

    def environment(self):
        packages = {}
        for name in ('torch', 'triton', 'vllm', 'transformers'):
            try:
                packages[name] = dict(distribution_version=importlib.metadata.version(name))
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        packages['torch'].update(imported_version=self.torch.__version__,
                                 path=self.torch.__file__)
        if os.environ.get('SPARSEDVFS_TRITON_PATH'):
            module = importlib.import_module('triton')
            origin = Path(module.__file__).resolve()
            expected = Path(os.environ['SPARSEDVFS_TRITON_PATH']).resolve()/'triton/__init__.py'
            if origin != expected or module.__version__ != '3.6.0':
                raise RuntimeError('Actual Triton differs from the explicit existing 3.6.0 path')
            packages['triton'] = dict(packages['triton'] or {}, imported_version=module.__version__,
                                     path=str(origin), entry_sha256=hashlib.sha256(origin.read_bytes()).hexdigest())
        free, total = self.torch.cuda.mem_get_info()
        return dict(python=sys.version, executable=sys.executable, gpu_uuid=self.uuid,
                    gpu_name=self.properties.name, torch_cuda=self.torch.version.cuda,
                    memory_free_bytes=free, memory_total_bytes=total, packages=packages,
                    sources_sha256=source_hashes(),
                    env={key: os.environ.get(key) for key in
                         ('CONDA_PREFIX', 'PYTHONPATH', 'TRITON_CACHE_DIR',
                          'SPARSEDVFS_TRITON_PATH', 'CUDA_VISIBLE_DEVICES')})

    def measure(self, row, warmup, repeat, observe):
        torch, spec = self.torch, row['case']
        def phase(name):
            row.update(active_phase=name, sample_index=None)
        def probe():
            operation()
            torch.cuda.synchronize()
            return time.monotonic_ns()
        phase('allocating')
        with torch.inference_mode():
            generator = torch.Generator(device='cuda').manual_seed(0)
            if spec['operator'] == 'gemm':
                m, n, k = spec['shape']
                a = torch.randn((m, k), dtype=torch.bfloat16, device='cuda', generator=generator)
                b = torch.randn((k, n), dtype=torch.bfloat16, device='cuda', generator=generator)
                output = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
                def operation(): torch.mm(a, b, out=output)
            else:
                a = torch.randn(spec['shape'], dtype=torch.bfloat16, device='cuda', generator=generator)
                b = torch.randn(spec['shape'], dtype=torch.bfloat16, device='cuda', generator=generator)
                output = torch.empty_like(a)
                def operation(): torch.add(a, b, out=output)
            # Primary author-style host timing. No CUDA Event calls inside it.
            phase('warmup_wall')
            for _ in range(warmup): operation()
            torch.cuda.synchronize()
            last_sync = time.monotonic_ns()
            observe('before_wall', last_sync, probe)
            phase('sampling_wall')
            for index in range(repeat):
                row['sample_index'] = index
                start = time.perf_counter()
                operation()
                torch.cuda.synchronize()
                row['wall_samples_s'].append(time.perf_counter()-start)
                last_sync = time.monotonic_ns()
            phase('after_wall')
            observe('after_wall', last_sync, probe)
            # Independent CUDA Event pass for effective capacity estimates.
            events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                      for _ in range(repeat)]
            phase('warmup_cuda')
            for _ in range(warmup): operation()
            torch.cuda.synchronize()
            last_sync = time.monotonic_ns()
            observe('before_cuda', last_sync, probe)
            phase('sampling_cuda')
            for index, (start, end) in enumerate(events):
                row['sample_index'] = index
                start.record()
                operation()
                end.record()
                end.synchronize()
                row['cuda_event_samples_s'].append(start.elapsed_time(end)/1000.0)
                last_sync = time.monotonic_ns()
            phase('after_cuda')
            observe('after_cuda', last_sync, probe)
            phase('complete')


def doctor(args, *, device_factory=backend, runtime_factory=TorchRuntime):
    config = settings(args)
    if args.apply:
        raise ValueError('doctor is read-only; --apply belongs only to collect')
    out = fresh(args.out, 'doctor.json')
    report = dict(schema=SCHEMA, status='failed', clock_writes_performed=False,
                  config=config, protocol=protocol(args))
    device = None
    try:
        runtime = runtime_factory(args.gpu_uuid)
        report['environment'] = runtime.environment()
        device = device_factory(config)
        report['capabilities'] = capabilities(device, args.gpu_uuid, args.rates_mhz)
        report['idle_readback'] = ClockGuard(device, args.gpu_uuid).observe()
        report['status'] = 'ready'
    except BaseException as exc:
        report['error'] = safe_error(exc)
        if isinstance(exc, ClockReadbackError):
            report['idle_readback'] = exc.snapshot
        raise
    finally:
        try:
            if device is not None:
                try:
                    device.close()
                except BaseException as exc:
                    report.update(status='failed', close_error=safe_error(exc))
                    raise
        finally:
            save(out/'doctor.json', report)
    return report


def collect(args, *, device_factory=backend, runtime_factory=TorchRuntime, sleep=time.sleep,
            progress=None, now_ns=time.monotonic_ns):
    if not args.apply:
        raise ValueError('Real clock writes require explicit --apply')
    config = settings(args)
    out = fresh(args.out, 'measurements.json', 'model.json', 'validation.json', 'predictions.csv')
    data = dict(schema=SCHEMA, kind='eq1_isolated_operator_measurements', status='collecting',
                created_utc=datetime.now(timezone.utc).isoformat(), config=config,
                protocol=protocol(args), cases=cases(), records=[], synthetic=False,
                real_cuda_executed=False, rate_transitions=[],
                active_context=dict(phase='initializing'),
                restoration=dict(attempted=False, reset_returned=False,
                                                          physical_restore_verified=False))
    device = None
    device_lock = None
    dirty = False
    failure = None
    old_handler = None
    handler_installed = False
    row = None
    try:
        def terminate(signum, frame):
            raise KeyboardInterrupt('SIGTERM during Eq1 collection')
        if hasattr(signal, 'SIGTERM'):
            try:
                old_handler = signal.signal(signal.SIGTERM, terminate)
                handler_installed = True
            except ValueError:
                pass  # Tests may invoke collection from a non-main thread.
        runtime = runtime_factory(args.gpu_uuid)
        data['environment'] = runtime.environment()
        if data['environment'].get('memory_free_bytes', 0) < 2*1024**3:
            raise RuntimeError('Less than 2 GiB free; stop your own inference service first')
        if (out/'doctor.json').exists():
            prior = load(out/'doctor.json')
            if (prior.get('status') != 'ready' or prior['config'] != config or
                    prior['environment']['sources_sha256'] != data['environment']['sources_sha256']):
                raise ValueError('Doctor/config/source mismatch; use a fresh output and rerun doctor')
        if config['backend'] == 'nvml':
            from .bridge import DeviceLock
            device_lock = DeviceLock(args.gpu_uuid)
        device = device_factory(config)
        data['capabilities'] = capabilities(device, args.gpu_uuid, args.rates_mhz)
        data['synthetic'] = not runtime.real_cuda
        guard = ClockGuard(device, args.gpu_uuid, now_ns=now_ns)
        for rate in args.rates_mhz:
            row = None
            data['active_context'] = dict(phase='setting_frequency', requested_mhz=rate)
            transition = dict(requested_mhz=rate, set_started_client_ns=now_ns(), set_returned=False)
            data['rate_transitions'].append(transition)
            dirty = True  # A failing write may still have partially changed state.
            device.set(rate)
            transition.update(set_returned=True, set_ended_client_ns=now_ns())
            data['active_context']['phase'] = 'settle_wait'
            transition['settle_started_client_ns'] = now_ns()
            sleep(0.5)
            transition['settle_ended_client_ns'] = now_ns()
            for spec in data['cases']:
                row = dict(record_id=f'{rate}/{spec["case_id"]}', case=dict(spec),
                           requested_mhz=rate, status='in_progress', wall_samples_s=[],
                           cuda_event_samples_s=[], snapshots={}, clock_observations=[],
                           clock_checks={}, active_phase='allocating')
                data['records'].append(row)
                data['active_context'] = dict(phase='measuring', requested_mhz=rate,
                                              record_id=row['record_id'])
                runtime.measure(row, args.warmup, args.repeat,
                                ClockChecks(guard, row, args.clock_stability_timeout_s,
                                            sleep=sleep, now_ns=now_ns))
                for values in (row['wall_samples_s'], row['cuda_event_samples_s']):
                    if len(values) != args.repeat: raise ValueError('Incomplete timing repetitions')
                    for value in values: number(value, 'measured duration')
                row['status'] = 'complete'
                data['real_cuda_executed'] = runtime.real_cuda
                data['memory_clock_mhz'] = guard.memory_mhz
                save(out/'measurements.json', data)
                if progress is not None:
                    progress(row, len(data['records']), len(data['cases'])*len(args.rates_mhz))
    except BaseException as exc:
        failure = exc
        data['error'] = safe_error(exc)
        context = dict(data['active_context'])
        if row is not None:
            row['status'] = 'failed'
            context.update(phase=row['active_phase'], sample_index=row.get('sample_index'),
                           wall_samples_completed=len(row['wall_samples_s']),
                           cuda_event_samples_completed=len(row['cuda_event_samples_s']),
                           last_readback=row['clock_observations'][-1] if row['clock_observations'] else None)
        data['failure_context'] = context
    finally:
        if device is not None and dirty:
            data['restoration']['attempted'] = True
            try:
                device.reset()
                data['restoration']['reset_returned'] = True
            except BaseException as exc:
                data['restoration']['error'] = safe_error(exc)
                failure = RuntimeError('GPU reset failed; stop and restore selected GPU manually')
        if device is not None:
            try:
                device.close()
            except BaseException as exc:
                data['close_error'] = safe_error(exc)
                failure = failure or exc
        if device_lock is not None:
            try:
                device_lock.close()
            except BaseException as exc:
                data['lock_close_error'] = safe_error(exc)
                failure = failure or exc
        if handler_installed:
            signal.signal(signal.SIGTERM, old_handler)
        data['status'] = 'failed' if failure is not None else 'collected'
        if failure is None:
            data['active_context'] = dict(phase='complete')
        save(out/'measurements.json', data)
    if failure is not None:
        raise failure
    return data


def checked_records(data, *, allow_synthetic=False):
    if (data.get('schema') != SCHEMA or data.get('kind') != 'eq1_isolated_operator_measurements' or
            data.get('status') != 'collected' or not data.get('restoration', {}).get('reset_returned')):
        raise ValueError('Require a complete collection with successful reset')
    if not allow_synthetic and (data.get('synthetic') is not False or data.get('real_cuda_executed') is not True):
        raise ValueError('Synthetic/unverified data cannot produce a real performance model')
    number(data.get('memory_clock_mhz'), 'recorded memory clock')
    if data['environment'].get('sources_sha256') != source_hashes():
        raise ValueError('Performance/core source changed after collection; use the matching code')
    p = data['protocol']
    if p.get('traffic_semantics') != TRAFFIC or p.get('timing') != TIMING or p.get('dtype') != 'bfloat16':
        raise ValueError('Measurement semantics differ from this Eq1 protocol')
    if p.get('clock_check_policy') != CLOCK_CHECK_POLICY or p.get('core_tolerance_mhz') != 15:
        raise ValueError('Clock check policy differs from this Eq1 protocol')
    stability = p.get('clock_stability', {})
    budget = number(stability.get('timeout_s'), 'clock stability timeout', zero=True)
    required = STABLE_READS if budget else 1
    if stability.get('required_consecutive_reads') != required:
        raise ValueError('Clock stability read count differs from this Eq1 protocol')
    rates = p['rates_mhz']
    if len(rates) < 2 or len(set(rates)) != len(rates) or p['reference_mhz'] not in rates:
        raise ValueError('Incomplete frequency grid')
    if data.get('cases') != cases():
        raise ValueError('Case shapes/roles/FLOPs/bytes differ from the declared protocol')
    expected = {f'{rate}/{spec["case_id"]}': (rate, spec) for rate in rates for spec in cases()}
    rows = {}
    for row in data['records']:
        key = row['record_id']
        if key not in expected or key in rows:
            raise ValueError('Unknown/duplicate observation')
        rate, spec = expected[key]
        if row['case'] != spec or row['requested_mhz'] != rate or row.get('status') != 'complete':
            raise ValueError('Observation identity/role differs')
        for field in ('wall_samples_s', 'cuda_event_samples_s'):
            if len(row[field]) != p['repeat']: raise ValueError('Missing raw samples')
            for value in row[field]: number(value, field)
        if set(row['snapshots']) != {'before_wall', 'after_wall', 'before_cuda', 'after_cuda'}:
            raise ValueError('Missing timing-pass readbacks')
        if (set(row.get('clock_checks', {})) != CHECK_PHASES or
                any(check.get('status') != 'passed' for check in row['clock_checks'].values())):
            raise ValueError('Incomplete/failed clock checkpoints')
        for phase, check in row['clock_checks'].items():
            reads = [r for r in row.get('clock_observations', []) if r.get('phase') == phase]
            count = required if phase in PRECHECK_PHASES else 1
            if (len(reads) != check.get('attempts') or len(reads) < count or
                    row['snapshots'][phase] != reads[-1] or
                    reads[-1].get('consecutive_matching_reads') != count or
                    any(r.get('passed') is not True or r.get('within_deadline') is not True
                        for r in reads[-count:])):
                raise ValueError('Clock stability evidence is incomplete')
        for reading in row['snapshots'].values():
            if (reading['uuid'].lower() != data['config']['uuid'].lower() or
                    reading.get('passed') is not True or reading.get('within_deadline') is not True or
                    abs(number(reading['clock_mhz'], 'GPU readback')-rate) > 15 or
                    reading['memory_clock_mhz'] != data['memory_clock_mhz']):
                raise ValueError('Stored clock readbacks do not match the measurement contract')
        rows[key] = row
    if set(rows) != set(expected): raise ValueError('Missing cases or frequency observations')
    return rows


def mean(row, field):
    return statistics.mean(row[field])


def fit_model(data, *, allow_synthetic=False):
    rows = checked_records(data, allow_synthetic=allow_synthetic)
    p = data['protocol']
    peak = {}
    for rate in p['rates_mhz']:
        row = rows[f'{rate}/gemm_8192']
        peak[str(rate)] = row['case']['work_flop']/mean(row, 'cuda_event_samples_s')
    reference = rows[f'{p["reference_mhz"]}/add_256mib']
    bandwidth = reference['case']['data_byte']/mean(reference, 'cuda_event_samples_s')
    residuals = []
    capacity_ids = [f'{rate}/gemm_8192' for rate in p['rates_mhz']]
    capacity_ids.append(reference['record_id'])
    for row in rows.values():
        if row['case']['role'] != 'calibration': continue
        spec = row['case']
        roofline = timeline_seconds(spec['work_flop'], spec['data_byte'], 0, 0,
                                    peak[str(row['requested_mhz'])], bandwidth, 0)
        residuals.append(dict(record_id=row['record_id'],
                              residual_s=mean(row, 'wall_samples_s')-roofline))
    median = statistics.median(r['residual_s'] for r in residuals)
    parameters = dict(p_peak_flops_by_gpu_mhz=peak, b_mem_logical_Bps=bandwidth,
                      t_overhead_s=max(0.0, median), s_comp=0.0, s_mem=0.0)
    return dict(schema=SCHEMA, kind='eq1_gpu_slice_performance_model',
                source_observations_sha256=digest(data), environment=data['environment'],
                synthetic=data['synthetic'], scope=p['scope'], traffic_semantics=TRAFFIC,
                timing=TIMING, memory_clock_mhz=data['memory_clock_mhz'],
                parameters=parameters, capacity_record_ids=capacity_ids,
                calibration_record_ids=[r['record_id'] for r in residuals],
                overhead_estimation=dict(policy=POLICY, residuals=residuals,
                                         unclamped_median_s=median,
                                         negative_residual_count=sum(r['residual_s'] < 0 for r in residuals)),
                units=dict(work='FLOP', data='byte', compute_capacity='FLOP/s',
                           logical_bandwidth='byte/s', time='s'),
                full_paper_reproduced=False,
                choices=['static logical input/output bytes, not hardware DRAM traffic',
                         'BF16 GEMM effective capacity, not universal theoretical peak',
                         'one reference logical bandwidth; no per-GPU-clock B_mem',
                         'one fixed nonnegative training-residual overhead, not author coefficients'])


def validate_model(data, model, *, allow_synthetic=False):
    rows = checked_records(data, allow_synthetic=allow_synthetic)
    if (model.get('schema') != SCHEMA or model.get('kind') != 'eq1_gpu_slice_performance_model' or
            model.get('source_observations_sha256') != digest(data) or
            model.get('environment') != data['environment'] or
            model.get('synthetic') != data['synthetic'] or
            model.get('memory_clock_mhz') != data['memory_clock_mhz'] or
            model.get('traffic_semantics') != TRAFFIC or model.get('timing') != TIMING):
        raise ValueError('Frozen model differs from its observations/semantics')
    params = model['parameters']
    if set(params) != PARAMETER_KEYS or params['s_comp'] != 0 or params['s_mem'] != 0:
        raise ValueError('Only one shared bandwidth/overhead and dense effective sparsity are supported')
    peak = params['p_peak_flops_by_gpu_mhz']
    if set(peak) != {str(rate) for rate in data['protocol']['rates_mhz']}:
        raise ValueError('Model frequency grid differs')
    for value in peak.values(): number(value, 'P_peak')
    bandwidth = number(params['b_mem_logical_Bps'], 'B_mem')
    overhead = number(params['t_overhead_s'], 'T_overhead', zero=True)
    expected_calibration = {key for key, row in rows.items() if row['case']['role'] == 'calibration'}
    if set(model['calibration_record_ids']) != expected_calibration:
        raise ValueError('Model fit must use only the declared calibration shapes')
    predictions = []
    for row in rows.values():
        if row['case']['role'] != 'validation': continue
        spec = row['case']
        predicted = timeline_seconds(spec['work_flop'], spec['data_byte'], 0, 0,
                                     peak[str(row['requested_mhz'])], bandwidth, overhead)
        observed = mean(row, 'wall_samples_s')
        error = abs(predicted-observed)/observed
        predictions.append(dict(case_id=spec['case_id'], gpu_mhz=row['requested_mhz'],
                                work_flop=spec['work_flop'], logical_data_byte=spec['data_byte'],
                                compute_s=spec['work_flop']/peak[str(row['requested_mhz'])],
                                memory_s=spec['data_byte']/bandwidth, overhead_s=overhead,
                                predicted_s=predicted, measured_wall_s=observed,
                                measured_cuda_s=mean(row, 'cuda_event_samples_s'),
                                wall_stdev_s=statistics.stdev(row['wall_samples_s']) if len(row['wall_samples_s']) > 1 else 0,
                                relative_error=error, within_10_percent=error <= .1))
    errors = [row['relative_error'] for row in predictions]
    diagnostic = []
    for rate in data['protocol']['rates_mhz']:
        row = rows[f'{rate}/add_256mib']
        observed_bw = row['case']['data_byte']/mean(row, 'cuda_event_samples_s')
        diagnostic.append(dict(gpu_mhz=rate, logical_bandwidth_Bps=observed_bw,
                               relative_to_reference=observed_bw/bandwidth-1,
                               used_for_B_mem=(rate == data['protocol']['reference_mhz'])))
    return dict(schema=SCHEMA, status='synthetic_test_only' if data['synthetic'] else 'evaluated',
                model_sha256=digest(model), source_observations_sha256=digest(data),
                predictions=predictions,
                summary=dict(cases=len(predictions), mean_relative_error=statistics.mean(errors),
                             max_relative_error=max(errors),
                             fraction_within_10_percent=statistics.mean(e <= .1 for e in errors)),
                bandwidth_diagnostic=diagnostic, full_paper_reproduced=False,
                claim='Eq1 checked only for this GPU slice, logical traffic and timing convention',
                note='No accuracy pass threshold is imposed; errors expose model limitations, not a savings result')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('doctor', 'collect', 'fit', 'validate'))
    p.add_argument('--out', required=True)
    p.add_argument('--gpu-uuid', default=DEFAULT_GPU)
    p.add_argument('--backend', choices=('nvml', 'bridge'), default='bridge')
    p.add_argument('--bridge-host', default='127.0.0.1')
    p.add_argument('--bridge-port', type=int, default=50900)
    p.add_argument('--rates-mhz', type=int, nargs='+', default=[2250, 2527])
    p.add_argument('--reference-mhz', type=int, default=2527)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--repeat', type=int, default=5)
    p.add_argument('--clock-stability-timeout-s', type=float, default=0,
                   help='Opt-in precheck budget; 0 keeps one strict read. Positive: '
                        'require 3 consecutive reads with extra untimed same-shape operations.')
    p.add_argument('--ack-default-auto', action='store_true')
    p.add_argument('--apply', action='store_true')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command != 'collect' and args.apply:
            raise ValueError('--apply belongs only to collect')
        if args.command == 'doctor':
            report = doctor(args)
            print(json.dumps(dict(status=report['status'], gpu_uuid=args.gpu_uuid,
                                  idle_readback=report['idle_readback']), indent=2))
        elif args.command == 'collect':
            def progress(row, current, total):
                print(f'[{current}/{total}] {row["record_id"]} complete', file=sys.stderr, flush=True)
            data = collect(args, progress=progress)
            print(json.dumps(dict(status=data['status'], records=len(data['records']),
                                  memory_clock_mhz=data['memory_clock_mhz'], restoration=data['restoration']), indent=2))
        elif args.command == 'fit':
            out = fresh(args.out, 'model.json')
            model = fit_model(load(out/'measurements.json'))
            save(out/'model.json', model)
            print(json.dumps(model['parameters'], indent=2))
        else:
            out = fresh(args.out, 'validation.json', 'predictions.csv')
            model = load(out/'model.json')
            report = validate_model(load(out/'measurements.json'), model)
            save(out/'validation.json', report)
            with (out/'predictions.csv').open('w', encoding='utf-8', newline='') as target:
                writer = csv.DictWriter(target, fieldnames=list(report['predictions'][0]))
                writer.writeheader()
                writer.writerows(report['predictions'])
            print('case | MHz | predicted_ms | measured_ms | relative_error')
            for row in report['predictions']:
                print(f'{row["case_id"]} | {row["gpu_mhz"]} | {row["predicted_s"]*1000:.6f} | '
                      f'{row["measured_wall_s"]*1000:.6f} | {row["relative_error"]:.3%}')
            print(json.dumps(report['summary'], indent=2))
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print(safe_error(exc), file=sys.stderr)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1


if __name__ == '__main__':
    raise SystemExit(main())
