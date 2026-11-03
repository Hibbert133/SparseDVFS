"""GPU board energy only. No simulated readings can pass real acceptance."""
import math
import random
import statistics
import threading
import time


def percentile(values, q):
    values = sorted(values)
    if not values: raise ValueError('No samples')
    pos = (len(values)-1)*q
    low = int(pos); high = min(low+1, len(values)-1)
    return values[low] + (values[high]-values[low])*(pos-low)


def summarize_energy(rows, uuid, evidence=None):
    if len(rows) < 2 or any(r.get('synthetic', True) or r['uuid'] != uuid for r in rows):
        raise ValueError('Energy needs real readings from the selected GPU')
    times = [r['timestamp_ns']/1e9 for r in rows]
    gaps = [b-a for a,b in zip(times, times[1:])]
    if min(gaps) <= 0: raise ValueError('Telemetry timestamps must increase on one device clock')
    seconds = times[-1]-times[0]
    if seconds < 30: raise ValueError('Energy window must be at least 30 seconds')
    counts = [r.get('energy_mj') for r in rows]
    integration_error = 0.0
    if all(v is not None for v in counts):
        if any(b < a for a,b in zip(counts, counts[1:])):
            raise ValueError('Energy counter reset/wrapped; discard window')
        joules = (counts[-1]-counts[0])/1000
        method = 'nvml_total_energy_counter'
    else:
        if not evidence or evidence.get('uuid') != uuid or evidence.get('validated') is not True:
            raise ValueError('No energy counter; power integration needs reviewed accuracy evidence for this UUID')
        integration_error = evidence.get('max_relative_error', 0)
        if not 0 < integration_error < .1 or not evidence.get('reference_method'):
            raise ValueError('Power accuracy evidence must state reference method and an error bound below 10%')
        if max(gaps) > evidence.get('max_sample_interval_s', 0):
            raise ValueError('Sampling gaps exceed power accuracy evidence')
        power = [r.get('power_mw') for r in rows]
        if any(v is None or v <= 0 for v in power): raise ValueError('Power telemetry unavailable')
        joules = sum((a+b)/2000*dt for a,b,dt in zip(power, power[1:], gaps))
        method = 'validated_power_trapezoid'
    if not math.isfinite(joules) or joules <= 0: raise ValueError('Nonpositive/invalid energy')
    # Transport/end-point uncertainty is a conservative acceptance margin, not a sensor calibration.
    endpoint = 2*max(r.get('roundtrip_s', 0) for r in rows)/seconds
    uncertainty = max(.005, integration_error + endpoint + .002/joules)
    return dict(energy_j=joules, seconds=seconds, method=method, scope='GPU_board',
                uncertainty_fraction=uncertainty, max_sampling_gap_s=max(gaps),
                clock_domain='telemetry_source_monotonic_only', physical_sensor_accuracy_verified=bool(evidence))


class Meter:
    def __init__(self, device, uuid, evidence=None, interval=.1):
        self.device, self.uuid, self.evidence, self.interval = device, uuid, evidence, interval
        self.rows = []; self.error = None; self.stop_event = threading.Event()
    def sample(self):
        start = time.monotonic()
        row = self.device.telemetry()
        row['roundtrip_s'] = time.monotonic()-start
        self.rows.append(row)
    def loop(self):
        while not self.stop_event.wait(self.interval):
            try: self.sample()
            except Exception as exc: self.error = exc; return
    def start(self):
        self.sample()
        self.thread = threading.Thread(target=self.loop, daemon=True); self.thread.start()
    def stop(self):
        self.stop_event.set(); self.thread.join(timeout=10)
        if self.thread.is_alive(): raise RuntimeError('Telemetry thread did not stop')
        if self.error: raise RuntimeError('Telemetry failed') from self.error
        self.sample()
        return summarize_energy(self.rows, self.uuid, self.evidence)


def acceptance(pairs):
    """Paired bootstrap, frozen candidate, no test-set selection."""
    if len(pairs) < 10: raise ValueError('At least 10 paired test windows required')
    ratios, delays, margins = [], [], []
    experiment_identity = None
    selection_sha = None
    for pair in pairs:
        auto, sparse = pair['auto'], pair['sparse']
        if pair['pair_id'] != auto['pair_id'] or pair['pair_id'] != sparse['pair_id']:
            raise ValueError('Pair ID mismatch')
        for row in (auto, sparse):
            if (not row.get('real_weights') or not row.get('real_cuda') or row.get('synthetic') or
                    row.get('split') != 'test' or not row.get('energy') or
                    not row.get('restoration_ok') or not row.get('output_checked')):
                raise ValueError('Unverified/synthetic/non-held-out result cannot support a real claim')
            if row['energy']['seconds'] < 30 or row['tokens'] <= 0 or row['mean_latency_s'] <= 0:
                raise ValueError('Invalid measurement window')
        if not sparse.get('frequency_applied') or auto.get('frequency_applied'):
            raise ValueError('Expected system-auto versus real controlled clocks')
        if not sparse.get('coverage_complete'):
            raise ValueError('Sparse execution boundaries were not validated')
        if not sparse.get('selection_sha256'):
            raise ValueError('Missing frozen selection')
        if selection_sha is None: selection_sha = sparse['selection_sha256']
        if selection_sha != sparse['selection_sha256']: raise ValueError('Candidate changed across test pairs')
        if auto['identity'] != sparse['identity'] or auto['outputs'] != sparse['outputs'] or auto['tokens'] != sparse['tokens']:
            raise ValueError('Mismatched environment/model/workload/output')
        if experiment_identity is None: experiment_identity = auto['identity']
        if experiment_identity != auto['identity']: raise ValueError('Experiment identity changed between pairs')
        e0, e1 = (row['energy']['energy_j']/row['tokens'] for row in (auto, sparse))
        if min(e0, e1) <= 0 or not all(math.isfinite(v) for v in (e0,e1)):
            raise ValueError('Invalid energy')
        ratios.append(e1/e0-1)
        delays.append(sparse['mean_latency_s']/auto['mean_latency_s']-1)
        margins.append(auto['energy']['uncertainty_fraction'] + sparse['energy']['uncertainty_fraction'])
    if len({p['pair_id'] for p in pairs}) != len(pairs): raise ValueError('Duplicate test pairs')
    rng = random.Random(5090)
    boot = [statistics.mean(rng.choices(ratios, k=len(ratios))) for _ in range(10000)]
    ci = [percentile(boot, .025), percentile(boot, .975)]
    latency = statistics.mean(p['sparse']['mean_latency_s'] for p in pairs) / statistics.mean(p['auto']['mean_latency_s'] for p in pairs)-1
    margin = max(margins)
    success = ci[1] < -margin and latency <= .05
    return dict(status='reproduced_for_this_workload' if success else 'not_reproduced_or_inconclusive',
                success=success, gpu_energy_change_mean=statistics.mean(ratios),
                gpu_energy_change_paired_bootstrap_ci95=ci, energy_uncertainty_margin=margin,
                mean_latency_change=latency, latency_limit=.05, pairs=len(pairs),
                p95_request_latency_s={k: percentile([r['elapsed_s'] for p in pairs for r in p[k]['requests']], .95)
                                       for k in ('auto','sparse')},
                energy_scope='GPU board, not whole system', paper_complete=False)
