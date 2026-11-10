"""Read-only dual-UUID sampling, bounded clock mapping, explicit clock writes."""
from concurrent.futures import ThreadPoolExecutor
import math
import os
import statistics
import threading
import time

from .backends import Bridge, NVML
from .observe_core import GPU_UUIDS
from .service_core import positive


def open_devices(config):
    if len(config) != 2 or {r['uuid'] for r in config} != set(GPU_UUIDS):
        raise ValueError('Require exactly the two declared GPU UUIDs')
    devices = []
    try:
        for uuid in GPU_UUIDS:
            row = next(r for r in config if r['uuid'] == uuid)
            if set(row)-{'backend', 'uuid', 'host', 'port'}:
                raise ValueError('Unknown telemetry setting; credentials belong only in environment')
            if row['backend'] == 'nvml': device = NVML(uuid)
            elif row['backend'] == 'bridge':
                token = os.environ.get('SPARSEDVFS_BRIDGE_TOKEN')
                if not token: raise ValueError('Missing SPARSEDVFS_BRIDGE_TOKEN')
                device = Bridge(row.get('host', '127.0.0.1'), row.get('port', 50900), token, uuid)
            else: raise ValueError('Only real NVML/bridge backends are allowed')
            devices.append(device)
            probe = device.telemetry()
            if probe.get('uuid') != uuid or probe.get('synthetic') is not False:
                raise ValueError('Telemetry UUID or real-hardware check failed')
        return devices
    except BaseException:
        for device in devices:
            try: device.close()
            except Exception: pass
        raise


def sample_device(device, now=time.monotonic_ns):
    begin = now()
    raw = device.telemetry()
    end = now()
    if raw.get('uuid') != device.uuid or raw.get('synthetic') is not False:
        raise ValueError('Wrong UUID or synthetic telemetry')
    times = raw.get('field_query_times_ns', {})
    if not times:
        raise ValueError('Telemetry backend must provide per-field query timestamps')
    remote_start = raw['timestamp_ns']
    remote_end = max(t['end'] for t in times.values())
    # The remote query interval is enclosed by the local request interval.
    # Offset is an interval, NOT an equality between Windows and WSL clocks.
    resolution_s = raw.get('timestamp_clock', {}).get('resolution_s', 0)
    positive(resolution_s, 'remote timestamp resolution', zero=True)
    resolution_ns = math.ceil(resolution_s*1e9)
    # Include documented timer quantization, never an arbitrary error allowance.
    low, high = begin-remote_start-resolution_ns, end-remote_end+resolution_ns
    if high < low:
        raise ValueError('Remote elapsed time exceeds client bracket; incompatible clock units')
    offset, uncertainty = (low+high)/2, (high-low)/2
    mapped = {}
    for key, timespan in times.items():
        if not remote_start <= timespan['start'] <= timespan['end'] <= remote_end:
            raise ValueError('Invalid remote field timestamps')
        mapped[key] = dict(t_ns=(timespan['start']+timespan['end'])/2+offset,
                           uncertainty_ns=uncertainty+(timespan['end']-timespan['start'])/2)
    return dict(uuid=device.uuid, begin_ns=begin, end_ns=end, raw=raw,
                clock_offset_interval_ns=[low, high], mapped=mapped)


class Sampler:
    def __init__(self, devices, interval_s=.1):
        self.devices, self.interval_s = devices, positive(interval_s, 'sample interval')
        self.frames, self.errors = [], []
        self.stop_event, self.ready = threading.Event(), threading.Event()
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        if not self.ready.wait(15):
            self.stop_event.set()
            raise TimeoutError('No initial dual-card telemetry frame')
        if self.errors: raise RuntimeError(self.errors[0])
        return self

    def _run(self):
        with ThreadPoolExecutor(max_workers=2) as pool:
            try:
                while not self.stop_event.is_set():
                    started = time.monotonic()
                    frame = list(pool.map(sample_device, self.devices))
                    self.frames.append(frame)
                    self.ready.set()
                    self.stop_event.wait(max(0, self.interval_s-(time.monotonic()-started)))
            except Exception as exc:
                self.errors.append(str(exc)); self.ready.set()

    def finish(self):
        # Acquire one complete post-window frame before stopping.
        before = len(self.frames)
        deadline = time.monotonic()+15
        while len(self.frames) <= before and not self.errors and time.monotonic() < deadline:
            time.sleep(.01)
        self.stop_event.set()
        if self.thread: self.thread.join(timeout=15)
        if self.thread and self.thread.is_alive(): self.errors.append('Sampler did not stop')
        if len(self.frames) <= before: self.errors.append('Missing post-window frame')
        return self.frames


def field_series(samples, field):
    result = []
    for sample in samples:
        v = sample['raw'].get(field)
        if v is None: raise ValueError(f'{sample["uuid"]}: unavailable {field}: {sample["raw"].get("errors")}')
        positive(v, field, zero=True)
        m = sample['mapped'][field]
        result.append((m['t_ns']/1e9, float(v), m['uncertainty_ns']/1e9))
    result.sort()
    if any(b[0] <= a[0] for a, b in zip(result, result[1:])):
        raise ValueError('Duplicate/backwards mapped timestamps')
    return result


def interpolate(series, t):
    if len(series) < 2 or t < series[0][0] or t > series[-1][0]:
        raise ValueError('Measurement boundary is not bracketed by telemetry')
    for a, b in zip(series, series[1:]):
        if a[0] <= t <= b[0]:
            return a[1]+(b[1]-a[1])*(t-a[0])/(b[0]-a[0])
    raise ValueError('No interpolation bracket')


def energy_window(frames, begin_ns, end_ns, sources=None, interval_s=.1):
    t0, t1 = begin_ns/1e9, end_ns/1e9
    duration = positive(t1-t0, 'window duration')
    result, problems = {}, []
    for uuid in GPU_UUIDS:
        samples = [s for frame in frames for s in frame if s['uuid'] == uuid]
        if len(samples) < 4: raise ValueError('Insufficient dual-card telemetry')
        # A source is frozen at pilot time; intermittent counter failure is not a fallback.
        has_energy = [s['raw'].get('energy_mj') is not None for s in samples]
        source = sources[uuid] if sources else ('counter' if all(has_energy) else 'power')
        if source not in ('counter', 'power'): raise ValueError('Unknown energy source')
        if any(has_energy) and not all(has_energy): raise ValueError('Intermittent energy counter failure')
        field = 'energy_mj' if source == 'counter' else 'power_mw'
        series = field_series(samples, field)
        gaps = [b[0]-a[0] for a, b in zip(series, series[1:])]
        if max(gaps) > max(.35, interval_s*3.5): problems.append(f'{uuid}: telemetry sample gap')
        time_uncertainty = max(p[2] for p in series)
        changes = [(b[0], abs(b[1]-a[1])) for a, b in zip(series, series[1:]) if b[1] != a[1]]
        if len(changes) < 3: raise ValueError('Sensor update/resolution cannot be established from <3 changes')
        resolution = max([changes[0][0]-series[0][0], series[-1][0]-changes[-1][0]]+
                         [b[0]-a[0] for a, b in zip(changes, changes[1:])])
        lo, hi = interpolate(series, t0), interpolate(series, t1)
        if source == 'counter':
            if any(b[1] < a[1] for a, b in zip(series, series[1:])):
                raise ValueError('Cumulative energy reset/wrap; do not splice counters')
            joules = positive((hi-lo)/1000, 'counter energy')
            rate_bound = max((b[1]-a[1])/(b[0]-a[0])/1000 for a, b in zip(series, series[1:]))
            error_j = 2*max(v for _, v in changes)/1000 + 2*time_uncertainty*rate_bound
        else:
            points = [(t0, lo)]+[(t, v) for t, v, _ in series if t0 < t < t1]+[(t1, hi)]
            joules = positive(sum((a[1]+b[1])*.5*(b[0]-a[0])/1000
                                  for a, b in zip(points, points[1:])), 'integrated energy')
            rate_bound = max(v for _, v, _ in series)/1000
            error_j = (2*time_uncertainty+2*resolution)*rate_bound
        if error_j/joules > .03: problems.append(f'{uuid}: temporal/quantization budget exceeds 3%')
        temp = field_series(samples, 'temperature_c')
        result[uuid] = dict(energy_j=joules, source=source, error_budget_j=error_j,
            relative_uncertainty=error_j/joules, max_query_uncertainty_s=time_uncertainty,
            observed_value_change_gap_s=resolution, max_sample_gap_s=max(gaps),
            temperature_range_c=[min(p[1] for p in temp), max(p[1] for p in temp)],
            resolution_basis='observed_changes_conservative_timing_budget_not_sensor_accuracy_certificate')
    total = sum(r['energy_j'] for r in result.values())
    return dict(energy_j=total, power_w=total/duration, by_uuid=result,
                relative_uncertainty=sum(r['error_budget_j'] for r in result.values())/total,
                quality_errors=problems, scope='two_GPU_boards_including_wait_idle_no_CPU_or_wall_power')


def clock_errors(frames, target=None, memory_reference=None):
    errors = []
    for frame in frames:
        if {s['uuid'] for s in frame} != set(GPU_UUIDS): errors.append('Missing GPU in telemetry frame')
        for s in frame:
            r = s['raw']; uuid = s['uuid']
            for field in ('clock_mhz', 'memory_clock_mhz', 'temperature_c'):
                if r.get(field) is None: errors.append(f'{uuid}: unavailable {field}')
            if target is not None and (r.get('clock_mhz') is None or abs(r['clock_mhz']-target) > 15):
                errors.append(f'{uuid}: actual clock outside {target} +/- 15 MHz')
            if memory_reference and r.get('memory_clock_mhz') != memory_reference[uuid]:
                errors.append(f'{uuid}: memory clock changed')
    return sorted(set(errors))


class DualClocks:
    """Own both clocks as one transaction; reset every touched device on any failure."""
    def __init__(self, devices, apply=False, allowed_rates=(2250, 2527)):
        self.devices, self.allow_apply = devices, apply
        self.allowed_rates = frozenset(allowed_rates)
        if not self.allowed_rates or any(type(f) is not int or f <= 0 for f in self.allowed_rates):
            raise ValueError('Explicit positive integer clock allowlist required')
        self.dirty = []
        self.last_target = None
        self.restore_result = dict(required=False, reset_returned=False, verified=False)

    def set(self, mhz):
        if not self.allow_apply: raise ValueError('Real writes require --apply')
        if mhz not in self.allowed_rates: raise ValueError('Clock outside explicit allowed rates')
        self.last_target = mhz
        self.restore_result = dict(required=True, reset_returned=False, verified=False)
        for device in self.devices:
            if device not in self.dirty: self.dirty.append(device)
            device.set(mhz)

    def reset(self):
        errors = {}
        for device in self.dirty:
            try: device.reset()
            except Exception as exc: errors[device.uuid] = str(exc)
        self.restore_result.update(reset_returned=not errors, verified=False, errors=errors)
        if not errors: self.dirty.clear()
        if errors: raise RuntimeError('GPU restoration failed; stop: '+str(errors))
        return self.restore_result


def require_clock_gate(gate, identity_fingerprint, config_sha256):
    checks = (gate.get('schema') == 'optimized-service-v1', gate.get('status') == 'passed',
              gate.get('synthetic') is False,
              gate.get('identity_fingerprint') == identity_fingerprint,
              gate.get('telemetry_config_sha256') == config_sha256,
              gate.get('controlled_exception_injected') is True)
    if not all(checks): raise ValueError('Dual-card clock gate identity/status is not verified')
    cases = gate.get('cases', [])
    if ({r.get('target_mhz') for r in cases if r.get('loaded_readback_verified') is True}
            != {2250, 2527}):
        raise ValueError('Both frequency pairs need loaded readback evidence')
    for name in ('normal_restoration', 'exception_restoration', 'disconnect_restoration'):
        restoration = gate.get(name, {})
        evidence = restoration.get('evidence', restoration)
        if (restoration.get('verified') is not True or evidence.get('verified') is not True
                or set(evidence.get('by_uuid', {})) != set(GPU_UUIDS)
                or any(v.get('verified') is not True for v in evidence['by_uuid'].values())):
            raise ValueError(f'{name}: dual-card physical restoration unverified')
        if name != 'disconnect_restoration' and restoration.get('reset_returned') is not True:
            raise ValueError(f'{name}: reset failed')


def restoration_evidence(baseline_frames, after_frames, last_target):
    """Conservative witness: loaded clock above the former upper lock bound.

    Idle downclock alone is NOT evidence of unlocking. Lack of this witness is
    unverified, not permission to assume success. This does not restore an
    arbitrary preexisting lock; the caller must acknowledge baseline auto mode.
    """
    result = {}
    for uuid in GPU_UUIDS:
        def loaded(frames):
            return [s['raw']['clock_mhz'] for f in frames for s in f if s['uuid'] == uuid
                    and (s['raw'].get('utilization_pct') or 0) > 0 and s['raw'].get('clock_mhz')]
        before, after = loaded(baseline_frames), loaded(after_frames)
        ok = (len(before) >= 3 and len(after) >= 3 and
              sum(v > last_target+15 for v in after) >= 3 and
              any(min(before)-15 <= v <= max(before)+15 for v in after))
        result[uuid] = dict(verified=bool(ok), baseline_loaded_range=[min(before), max(before)] if before else None,
            restored_loaded_range=[min(after), max(after)] if after else None,
            basis='loaded_readbacks_above_previous_lock_ceiling_and_within_baseline_range')
    return dict(verified=all(r['verified'] for r in result.values()), by_uuid=result)
