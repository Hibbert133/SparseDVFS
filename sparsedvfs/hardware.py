"""Read-only discovery; explicit restoration is the only Python sysfs writer."""
import json
import math
import platform
import statistics
import threading
import time
from pathlib import Path


def read(path, default=None):
    try:
        return Path(path).read_text().strip()
    except (OSError, ValueError):
        return default


def integer(path, default=None):
    value = read(path)
    return int(value) if value is not None else default


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _domain(path, cpu=False):
    path = Path(path)
    lo = 'scaling_min_freq' if cpu else 'min_freq'
    hi = 'scaling_max_freq' if cpu else 'max_freq'
    avail = 'scaling_available_frequencies' if cpu else 'available_frequencies'
    current = 'scaling_cur_freq' if cpu else 'cur_freq'
    if any(read(path / name) is None for name in (lo, hi, avail)):
        raise RuntimeError(f'Incomplete frequency interface: {path}; provide a board-specific config.')
    minimum, maximum = integer(path / lo), integer(path / hi)
    frequencies = sorted(set(int(x) for x in read(path / avail).split()))
    frequencies = [f for f in frequencies if f >= minimum and (maximum == 0 or f <= maximum)]
    if not frequencies:
        raise RuntimeError(f'No discrete OPPs within current limits: {path}')
    return {'path': str(path), 'unit': 'kHz' if cpu else 'Hz', 'frequencies': frequencies,
            'hardware_current_path': str(path/'cpuinfo_cur_freq') if cpu and (path/'cpuinfo_cur_freq').exists() else None,
            'current_path': str(path / current), 'original_min': minimum, 'original_max': maximum,
            'governor': read(path / ('scaling_governor' if cpu else 'governor'))}


def _power_sources(root):
    sources = []
    for hw in sorted((root / 'class/hwmon').glob('hwmon*')):
        name = read(hw / 'name', '').lower()
        for path in hw.glob('power*_input'):
            idx = path.name[len('power'):-len('_input')]
            label = read(hw / f'power{idx}_label', read(hw / f'in{idx}_label', ''))
            sources.append({'kind': 'direct', 'path': str(path), 'scale_to_w': 1e-6,
                            'raw_unit': 'uW', 'unit_source': 'Linux hwmon power*_input ABI',
                            'update_interval_ms': integer(hw/'update_interval'),
                            'sensor': name, 'label': label, 'scope': 'unclassified'})
        for path in hw.glob('in*_input'):
            idx = path.name[len('in'):-len('_input')]
            curr = hw / f'curr{idx}_input'
            if curr.exists():
                sources.append({'kind': 'vi', 'voltage_path': str(path), 'current_path': str(curr),
                                'scale_to_w': 1e-6, 'raw_unit': 'mV*mA',
                                'unit_source': 'Linux hwmon in*_input mV and curr*_input mA',
                                'update_interval_ms': integer(hw/'update_interval'), 'sensor': name,
                                'label': read(hw / f'in{idx}_label', ''), 'scope': 'unclassified'})
    # Thor's INA238 VIN covers module + carrier. Never sum it with INA3221 rails.
    selected = next((p for p in sources if 'ina238' in p['sensor'] and p['kind'] == 'direct'), None)
    if selected is None:
        selected = next((p for p in sources if 'ina238' in p['sensor'] and p['kind'] == 'vi'), None)
    if selected:
        selected = dict(selected, scope='system_module_and_carrier')
    return sources, selected


def probe(root='/sys', emc_rates=(), gpu_path=None, allow_no_emc=False):
    root = Path(root)
    cpus = [_domain(p, True) for p in sorted((root / 'devices/system/cpu/cpufreq').glob('policy*'))]
    if not cpus:
        raise RuntimeError('No CPU cpufreq policies found. Run probe on the Thor host.')
    gpu = Path(gpu_path) if gpu_path else root / 'class/devfreq/gpu-gpc-0'
    if not gpu.is_dir():
        raise RuntimeError('Thor GPC node absent. Initialize CUDA (--cuda-check), or pass --gpu-path explicitly; do not select gpu-nvd-0.')
    gpu = _domain(gpu)
    emc_path = root / 'kernel/debug/bpmp/debug/clk/emc'
    halt_path = root / 'kernel/debug/bpmp/debug/bwmgr/bwmgr_halt'
    if all(read(p) is not None for p in (emc_path / 'rate', emc_path / 'mrq_rate_locked', halt_path)):
        possible = read(emc_path / 'possible_rates', '')
        frequencies = sorted(set(int(x) for x in emc_rates))
        if not frequencies and possible:
            frequencies = sorted(set(int(x) for x in possible.split() if x.isdigit()))
        minimum, maximum = integer(emc_path / 'min_rate'), integer(emc_path / 'max_rate')
        if minimum is not None and maximum is not None:
            if any(f < minimum or f > maximum for f in frequencies):
                raise RuntimeError('Explicit EMC rate outside BPMP min_rate/max_rate')
        emc = {'backend': 'bpmp', 'path': str(emc_path), 'halt_path': str(halt_path),
               'frequencies': frequencies, 'unit': 'Hz', 'current_path': str(emc_path / 'rate'),
               'min_rate': minimum, 'max_rate': maximum,
               'rates_source': 'explicit_target_input' if emc_rates else 'possible_rates_or_unresolved'}
    elif allow_no_emc:
        emc = {'backend': 'off', 'frequencies': [0], 'unit': 'Hz', 'path': '', 'halt_path': ''}
    else:
        raise RuntimeError('BPMP EMC/bwmgr nodes unavailable or unreadable. Try probe as root on the host. --allow-no-emc explicitly selects a degraded two-domain demo.')
    sources, selected = _power_sources(root)
    temps = [{'path': str(p / 'temp'), 'name': read(p / 'type', p.name)}
             for p in sorted((root / 'class/thermal').glob('thermal_zone*')) if (p / 'temp').exists()]
    common_cpu = sorted(set.intersection(*(set(p['frequencies']) for p in cpus)))
    if not common_cpu:
        raise RuntimeError('CPU policies have no common OPP. Per-policy vector adaptation is required.')
    observations = []
    bpmp = root/'kernel/debug/bpmp/debug'
    for path in sorted((bpmp/'regulator').glob('*/voltage')):
        observations.append(dict(name=path.parent.name, kind='voltage', path=str(path),
                                 scale_to_v=None, unit='unresolved',
                                 evidence='BPMP voltage node; supply verified scale in --sensor-config'))
    for name in ('gpc0clk', 'gpc1clk', 'gpusysclk', 'emc'):
        path = bpmp/'clk'/name/'pto_counter'
        if path.exists():
            observations.append(dict(name=name, kind='hardware_frequency', path=str(path),
                                     unit='raw', evidence='BPMP pto_counter; raw format preserved'))
    return {'schema': 1, 'device_model': read('/proc/device-tree/model', '').replace('\x00', ''),
            'platform': platform.platform(), 'jetson_release': read('/etc/nv_tegra_release'),
            'cpu': cpus, 'cpu_common_khz': common_cpu, 'gpu': gpu, 'emc': emc,
            'power_sources': sources, 'power': selected, 'thermal': temps,
            'read_only_observations': observations,
            'degraded_no_emc': emc['backend'] == 'off', 'probe_is_read_only': True}


def make_grid(hardware, points=2):
    if points < 1:
        raise ValueError('points must be positive')
    def choose(values):
        values = sorted(set(values))
        if not values:
            raise ValueError('EMC OPPs unresolved; supply measured/supported --emc-rates-hz in probe')
        if points == 1:
            return [values[-1]]
        return sorted({values[round(i * (len(values) - 1) / (points - 1))] for i in range(points)})
    cpu, gpu, emc = (choose(hardware['cpu_common_khz']), choose(hardware['gpu']['frequencies']),
                     choose(hardware['emc']['frequencies']))
    return {'schema': 1, 'units': ['cpu_khz', 'gpu_hz', 'emc_hz'],
            'triplets': [[c, g, e] for c in cpu for g in gpu for e in emc],
            'reference': [max(cpu), max(gpu), max(emc)],
            'degraded_no_emc': hardware['degraded_no_emc']}


def snapshot(hardware):
    domains = []
    for d, cpu in [(d, True) for d in hardware['cpu']] + [(hardware['gpu'], False)]:
        path = Path(d['path'])
        lo, hi = ('scaling_min_freq', 'scaling_max_freq') if cpu else ('min_freq', 'max_freq')
        domains.append({'min_path': str(path / lo), 'max_path': str(path / hi),
                        'min': integer(path / lo), 'max': integer(path / hi)})
    emc = hardware['emc']
    bpmp = None
    if emc['backend'] == 'bpmp':
        p = Path(emc['path'])
        bpmp = {'rate_path': str(p / 'rate'), 'lock_path': str(p / 'mrq_rate_locked'),
                'halt_path': emc['halt_path'], 'rate': integer(p / 'rate'),
                'lock': integer(p / 'mrq_rate_locked'), 'halt': integer(emc['halt_path'])}
    return {'schema': 1, 'captured_monotonic_ns': time.monotonic_ns(), 'domains': domains, 'bpmp': bpmp}


def restore(saved):
    # Called only by the explicit restore CLI, never by probe/import.
    errors = []
    def put(path, value):
        if value is None:
            raise ValueError(f'Incomplete saved value for {path}')
        with open(path, 'w') as f:
            f.write(str(value))
    for d in saved['domains']:
        try:
            current_max = integer(d['max_path'])
            if current_max and d['min'] > current_max:
                put(d['max_path'], d['max']); put(d['min_path'], d['min'])
            else:
                put(d['min_path'], d['min']); put(d['max_path'], d['max'])
        except (OSError, ValueError, TypeError) as exc:
            errors.append(str(exc))
    b = saved.get('bpmp')
    if b:
        for path, value in [(b['lock_path'], 1), (b['rate_path'], b['rate']),
                            (b['lock_path'], b['lock']), (b['halt_path'], b['halt'])]:
            try:
                put(path, value)
            except (OSError, ValueError) as exc:
                errors.append(str(exc))
    if errors:
        raise RuntimeError('Restoration incomplete: ' + '; '.join(errors))


def frequencies(hardware):
    return {'cpu_khz': [integer(d['current_path']) for d in hardware['cpu']],
            'gpu_hz': integer(hardware['gpu']['current_path']),
            'emc_hz': integer(hardware['emc'].get('current_path', '/nonexistent'))}


def observation(hardware):
    """Separate software requests, optional hardware readings and voltage raw data."""
    values = []
    for node in hardware.get('read_only_observations', []):
        raw = read(node['path'])
        value = dict(node, raw=raw, value_v=None)
        if node.get('kind') == 'voltage' and node.get('scale_to_v') is not None and node.get('evidence'):
            try:
                value['value_v'] = float(raw)*float(node['scale_to_v'])
            except (TypeError, ValueError):
                pass
        values.append(value)
    return dict(monotonic_ns=time.monotonic_ns(), software=frequencies(hardware),
                cpuinfo_cur_khz=[integer(d['hardware_current_path']) if d.get('hardware_current_path') else None
                                  for d in hardware['cpu']],
                sensors=values, temperatures_c=temperatures(hardware),
                physical_pll_settled=False)


def wait_frequencies(hardware, triplet, timeout_s=0.5, tolerance=0.05, keepalive=None):
    started = time.monotonic_ns()
    matches = 0
    while (time.monotonic_ns() - started) / 1e9 < timeout_s:
        if keepalive:
            keepalive()
        actual = frequencies(hardware)
        targets = [(v, triplet[0]) for v in actual['cpu_khz']] + [(actual['gpu_hz'], triplet[1])]
        if triplet[2]:
            targets.append((actual['emc_hz'], triplet[2]))
        good = all(v is not None and abs(v - f) <= tolerance * f for v, f in targets)
        matches = matches + 1 if good else 0
        if matches >= 2:
            return {'observed_us': (time.monotonic_ns() - started) / 1000, 'actual': actual}
        time.sleep(0.0002)
    raise RuntimeError(f'Frequency readback did not settle to {triplet}; last readback {actual}')


def power_w(source):
    if source is None:
        return None
    try:
        if source['kind'] == 'direct':
            value = float(read(source['path'])) * source['scale_to_w']
        elif source['kind'] == 'vi':
            value = float(read(source['voltage_path'])) * float(read(source['current_path'])) * source['scale_to_w']
        else:
            raise ValueError('power.kind must be direct or vi; select one total rail')
        return value if math.isfinite(value) and value >= 0 else None
    except (OSError, TypeError, ValueError):
        return None


class PowerMonitor:
    def __init__(self, source, interval=0.02):
        self.source, self.interval = source, interval
        self.samples, self.errors = [], 0
        self.event = threading.Event()
        self.thread = None
        self.client = None
        self.sensor_interval_s = None
        if source and source.get('update_interval_ms'):
            self.sensor_interval_s = source['update_interval_ms']/1000
        self.backend_error = None

    def sample(self):
        t = time.monotonic_ns()
        if self.source and self.source.get('kind') == 'jtop':
            try:
                value = float(self.client.power['tot']['power']) / 1000
                if not math.isfinite(value) or value < 0:
                    value = None
            except Exception as exc:
                value = None
                self.backend_error = str(exc)
        else:
            value = power_w(self.source)
        if value is None:
            self.errors += 1
        else:
            self.samples.append((t, value))

    def start(self):
        if self.source and self.source.get('kind') == 'jtop':
            try:
                from jtop import jtop
                self.client = jtop()
                self.client.start()
                self.sensor_interval_s = float(self.client.interval)
            except Exception as exc:
                self.backend_error = str(exc)
                self.errors += 1
                # Never install, restart a service, or synthesize a power sample.
        self.sample()
        def loop():
            while not self.event.wait(self.interval):
                self.sample()
        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.event.set()
        if self.thread:
            self.thread.join()
        self.sample()
        if self.client is not None:
            try:
                self.client.close()
            except Exception as exc:
                self.errors += 1
                self.backend_error = str(exc)
            self.client = None

    def summarize(self, start_ns, end_ns):
        # Boundary interpolation avoids integrating model-load/warm-up time.
        samples = sorted(self.samples)
        duration = (end_ns - start_ns) / 1e9
        if self.sensor_interval_s and duration < 2 * self.sensor_interval_s:
            return {'energy_j': None, 'average_power_w': None, 'samples': len(samples),
                    'read_errors': self.errors, 'valid': False,
                    'reason': 'window shorter than two sensor/service intervals; increase generation length',
                    'sensor_interval_s': self.sensor_interval_s}
        coverage = bool(samples and samples[0][0] <= start_ns and samples[-1][0] >= end_ns)
        inside = [(t, p) for t, p in samples if start_ns < t < end_ns]
        if not coverage or self.errors or duration <= 0 or len(samples) < 3 or not inside:
            return {'energy_j': None, 'average_power_w': None, 'samples': len(samples),
                    'read_errors': self.errors, 'valid': False, 'reason': 'missing power, read error, or insufficient samples',
                    'backend_error': self.backend_error}
        def at(t):
            for (ta, pa), (tb, pb) in zip(samples, samples[1:]):
                if ta <= t <= tb:
                    return pa + (pb - pa) * ((t - ta) / (tb - ta)) if tb > ta else pb
            raise ValueError('uncovered boundary')
        clipped = [(start_ns, at(start_ns)), *inside, (end_ns, at(end_ns))]
        energy = sum((b[0] - a[0]) / 1e9 * (a[1] + b[1]) / 2 for a, b in zip(clipped, clipped[1:]))
        return {'energy_j': energy, 'average_power_w': energy / duration,
                'samples': len(samples), 'read_errors': self.errors, 'valid': True,
                'scope': self.source.get('scope', 'explicit_source'),
                'sensor_interval_s': self.sensor_interval_s}


def temperatures(hardware):
    values = {}
    for sensor in hardware.get('thermal', []):
        value = integer(sensor['path'])
        values[sensor['name']] = value / 1000 if value is not None else None
    return values
