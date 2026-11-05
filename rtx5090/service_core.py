"""Dependency-free contracts for optimized-service observations and models.

These are board-energy / HTTP-window empirical models, NOT SparseDVFS Eq1/Eq2.
No target identity, missing trace, sensor resolution, or validation is inferred.
"""
from bisect import bisect_right
from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path
import re
import statistics

from .observe_core import digest, GPU_UUIDS

SCHEMA = 'optimized-service-v1'
LENGTHS = (128, 512, 2048)
TARGETS = ('ttft_s', 'tpot_s', 'window_s', 'energy_j')
LIMITS = dict(wape=0.10, p90_relative_error=0.20)


def positive(x, name, zero=False):
    if (isinstance(x, bool) or not isinstance(x, (int, float))
            or not math.isfinite(x) or (x < 0 if zero else x <= 0)):
        raise ValueError(f'{name} must be finite and {"nonnegative" if zero else "positive"}')
    return x


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def read(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def quantile(values, q):
    a = sorted(values)
    if not a:
        raise ValueError('No samples')
    p = (len(a)-1)*q
    lo = int(p)
    return a[lo] + (a[min(lo+1, len(a)-1)]-a[lo])*(p-lo)


def errors(actual, predicted):
    if len(actual) != len(predicted) or not actual:
        raise ValueError('Expected nonempty paired predictions')
    rel = []
    for y, p in zip(actual, predicted):
        positive(y, 'label'); positive(p, 'prediction', zero=True)
        rel.append(abs(p-y)/y)
    return dict(n=len(rel), wape=sum(abs(p-y) for y, p in zip(actual, predicted))/sum(actual),
                p90_relative_error=quantile(rel, .9), max_relative_error=max(rel),
                within_10_percent=sum(x <= .1 for x in rel)/len(rel))


def nnls_line(xs, ys):
    """Exact two-variable NNLS by checking interior and both boundary faces."""
    if len(xs) != len(ys) or len(set(xs)) < 2:
        raise ValueError('Insufficient independent input lengths')
    for x in xs: positive(x, 'L/512')
    for y in ys: positive(y, 'target')
    n, sx, sy = len(xs), sum(xs), sum(ys)
    sxx, sxy = sum(x*x for x in xs), sum(x*y for x, y in zip(xs, ys))
    det = n*sxx-sx*sx
    a, b = (sy*sxx-sx*sxy)/det, (n*sxy-sx*sy)/det
    candidates = [(max(sy/n, 0), 0), (0, max(sxy/sxx, 0))]
    if a >= 0 and b >= 0: candidates.append((a, b))
    a, b = min(candidates, key=lambda p: sum((p[0]+p[1]*x-y)**2 for x, y in zip(xs, ys)))
    return dict(intercept=a, slope=b)


def validate_windows(data, split):
    if data.get('schema') != SCHEMA or data.get('split') != split:
        raise ValueError('Wrong schema or dataset split')
    if data.get('status') != 'collected' or data.get('synthetic') is not False:
        raise ValueError('Only completed real measurements may be fitted/validated')
    if data.get('restoration', {}).get('required') and not data['restoration'].get('verified'):
        raise ValueError('Physical clock restoration has not been verified')
    if not data.get('identity_ready'):
        raise ValueError('Service identity not verified')
    protocol = data['protocol']
    if (protocol['output_tokens'] != 64 or protocol['concurrency'] != 1
            or protocol['lengths'] != list(LENGTHS) or protocol['windows_per_cell'] != 3):
        raise ValueError('Unexpected preregistered workload')
    if protocol['frequencies'] not in (['current'], ['2527', '2250']):
        raise ValueError('Only current policy or the two accepted frequency pairs are modeled')
    rows = data['windows']
    cells = Counter()
    all_requests, all_prompts = set(), set()
    for row in rows:
        if row.get('status') != 'valid' or row.get('quality_errors'):
            raise ValueError('Invalid/unknown windows must not be silently omitted')
        if row['length'] not in LENGTHS or row['frequency'] not in protocol['frequencies']:
            raise ValueError('Undeclared workload or frequency')
        if row.get('cache_status') != 'verified_miss':
            raise ValueError('Complete prefill cache misses are unverified')
        if positive(row.get('energy_relative_uncertainty', 1), 'Energy uncertainty', zero=True) > .03:
            raise ValueError('Energy timing/quantization uncertainty exceeds 3%')
        if row['requests'] != protocol['requests_per_window']:
            raise ValueError('Window size changed')
        for key in TARGETS: positive(row[key], key)
        for u in GPU_UUIDS:
            clocks = row['clock_ranges'][u]['core']
            if len(clocks) != 2 or clocks[1] < clocks[0]: raise ValueError('Invalid clock range')
            for clock in clocks: positive(clock, 'observed GPU MHz')
            if row['frequency'] != 'current' and any(abs(c-int(row['frequency'])) > 15 for c in clocks):
                raise ValueError('Actual clock differs from the declared frequency pair')
        for energy in row['energy_by_uuid_j'].values(): positive(energy, 'Per-GPU energy')
        if row['energy_j'] != sum(row['energy_by_uuid_j'].values()):
            raise ValueError('Energy must sum the two UUIDs')
        if set(row['energy_by_uuid_j']) != set(GPU_UUIDS):
            raise ValueError('Both GPUs must be covered')
        for key, seen in (('request_ids', all_requests), ('prompt_hashes', all_prompts)):
            values = row[key]
            if len(values) != row['requests'] or len(set(values)) != len(values) or seen.intersection(values):
                raise ValueError(f'Duplicate or missing {key}')
            seen.update(values)
        cells[(row['frequency'], row['length'])] += 1
    expected = {(f, n): 3 for f in protocol['frequencies'] for n in LENGTHS}
    if dict(cells) != expected:
        raise ValueError('Require three independent windows per declared cell')
    return rows


def fit(data):
    rows = validate_windows(data, 'train')
    parameters = {}
    for freq in data['protocol']['frequencies']:
        subset = [r for r in rows if r['frequency'] == freq]
        parameters[freq] = {key: nnls_line([r['length']/512 for r in subset],
                                         [r[key] for r in subset]) for key in TARGETS}
    clock_ranges = {u: [min(r['clock_ranges'][u]['core'][0] for r in rows),
                        max(r['clock_ranges'][u]['core'][1] for r in rows)] for u in GPU_UUIDS}
    return dict(schema=SCHEMA, kind='service_window_empirical_linear',
                parameters=parameters, identity_fingerprint=data['identity_fingerprint'],
                protocol=data['protocol'], training_sha256=digest(data),
                training_prompt_hashes=sorted({p for r in rows for p in r['prompt_hashes']}),
                training_request_ids=sorted({p for r in rows for p in r['request_ids']}),
                training_seed=data['seed'], thresholds=LIMITS,
                training_windows=len(rows), independent_validation='pending',
                temperature_range_c=data['temperature_range_c'],
                observed_clock_ranges_mhz=clock_ranges,
                identity_unknown=data.get('identity', {}).get('unknown', []),
                ready_for_operator_models=False,
                units=dict(ttft_s='s', tpot_s='s/token', window_s='s', energy_j='J', power_w='W'),
                scope='same_service_boot_client_route_dtype_weights_cache_policy_observed_lengths_64_output_c1',
                full_paper_reproduced=False, operator_energy_measured=False,
                holdout_1024='sealed_not_collected')


def validate(data, model):
    rows = validate_windows(data, 'validation')
    if (model.get('kind') != 'service_window_empirical_linear' or model.get('schema') != SCHEMA
            or data.get('frozen_model_sha256') != digest(model)
            or data['identity_fingerprint'] != model['identity_fingerprint']
            or data['protocol'] != model['protocol'] or data['seed'] == model['training_seed']
            or model['thresholds'] != LIMITS):
        raise ValueError('Validation must match the frozen model/protocol and use a new seed')
    if any(set(r['prompt_hashes']) & set(model['training_prompt_hashes']) or
           set(r['request_ids']) & set(model['training_request_ids']) for r in rows):
        raise ValueError('Training/validation leakage')
    predictions = []
    scope_errors = []
    for uuid in GPU_UUIDS:
        lo, hi = model['temperature_range_c'][uuid]
        vlo, vhi = data['temperature_range_c'][uuid]
        if vlo < lo or vhi > hi:
            scope_errors.append(f'{uuid}: validation temperature outside training range')
        if 'current' in model['protocol']['frequencies']:
            clo, chi = model['observed_clock_ranges_mhz'][uuid]
            if any(r['clock_ranges'][uuid]['core'][0] < clo-15 or
                   r['clock_ranges'][uuid]['core'][1] > chi+15 for r in rows):
                scope_errors.append(f'{uuid}: current-policy frequency outside training range')
    for row in rows:
        ps = model['parameters'][row['frequency']]
        pred = {k: ps[k]['intercept']+ps[k]['slope']*row['length']/512 for k in TARGETS}
        pred['power_w'] = pred['energy_j']/positive(pred['window_s'], 'predicted window time')
        measured = {k: row[k] for k in TARGETS}
        measured['power_w'] = row['energy_j']/row['window_s']
        predictions.append(dict(frequency=row['frequency'], length=row['length'],
                                actual=measured, predicted=pred,
                                energy_relative_uncertainty=row['energy_relative_uncertainty']))
    def summarize(subset):
        return {k: errors([r['actual'][k] for r in subset], [r['predicted'][k] for r in subset])
                for k in (*TARGETS, 'power_w')}
    summary = summarize(predictions)
    by_frequency = {f: summarize([r for r in predictions if r['frequency'] == f])
                    for f in model['protocol']['frequencies']}
    cells = {f'{f}/{n}': summarize([r for r in predictions if r['frequency'] == f and r['length'] == n])
             for f in model['protocol']['frequencies'] for n in LENGTHS}
    passed = not scope_errors and all(stats[k]['wape'] <= .10 and
        stats[k]['p90_relative_error'] <= .20 for stats in by_frequency.values() for k in TARGETS)
    return dict(schema=SCHEMA, status='passed' if passed else 'not_passed',
                model_sha256=digest(model), validation_sha256=digest(data),
                summary=summary, by_frequency=by_frequency, cells=cells,
                scope_errors=scope_errors, predictions=predictions,
                max_energy_relative_uncertainty=max(r['energy_relative_uncertainty'] for r in rows),
                uncertainty_note='Only 3 independent windows/cell; no long-term or P99 claim',
                graph_partition_ready=False)


def metric_snapshot(text):
    """Keep counters by full label set; never combine different model/engine labels."""
    result = {}
    for line in text.splitlines():
        if not line or line.startswith('#'): continue
        match = re.match(r'([^\s]+)\s+([-+\d.eE]+)(?:\s+\S+)?$', line)
        if match:
            key, value = match.groups()
            if key.split('{', 1)[0].endswith('_created'): continue
            if any(s in key for s in ('prefix_cache_hits', 'prefix_cache_queries',
                                     'request_success_total', 'prompt_tokens_total',
                                     'generation_tokens_total')):
                result[key] = float(value)
    return result


def cache_miss_status(before, after, expected_requests, prefix_caching=None):
    hits = [k for k in before if 'prefix_cache_hits' in k]
    success = [k for k in before if 'request_success_total' in k]
    if (not hits and prefix_caching is not False) or not success or set(before) != set(after):
        return 'unknown', ['Missing or changed cache/request counter series']
    delta = {k: after[k]-before[k] for k in before}
    if any(v < 0 or not math.isfinite(v) for v in delta.values()):
        return 'unknown', ['Metric counter reset']
    if sum(delta[k] for k in success) != expected_requests:
        return 'unknown', ['Concurrent/extra/missing service requests in measured window']
    if any(delta[k] > 0 for k in hits):
        return 'cache_hit', ['Prefix cache hits in window; not complete prefill']
    return 'verified_miss', []


def correlate_trace(trace, expected_iterations=None):
    """One graph launch may own MANY kernels. Never infer ownership by GPU time.

    Boundary checks establish only sampled forward coverage, not full graph nodes.
    Shape/stride/capture mappings remain unknown without separate evidence.
    """
    events = trace['traceEvents'] if isinstance(trace, dict) else trace
    scopes = [e for e in events if e.get('ph') == 'X' and e.get('cat') == 'user_annotation'
              and re.match(r'execute_(?:context_|\d+_context_)', e.get('name', ''))]
    threads = defaultdict(list)
    for i, e in enumerate(scopes): threads[(e.get('pid'), e.get('tid'))].append((e['ts'], i, e))
    for entries in threads.values(): entries.sort(key=lambda x: x[0])
    starts = {k: [e[0] for e in v] for k, v in threads.items()}
    def cid(e):
        args = e.get('args', {})
        for k in ('correlation', 'correlation id', 'correlationId', 'Correlation ID'):
            if k in args: return str(args[k])
        return None
    launches = defaultdict(list)
    for e in events:
        cat, name = str(e.get('cat', '')).lower(), str(e.get('name', '')).lower()
        if not ('cuda_runtime' in cat or 'cuda_driver' in cat): continue
        if not any(x in name for x in ('launch', 'memcpy', 'memset')) or cid(e) is None: continue
        key = (e.get('pid'), e.get('tid'))
        index = bisect_right(starts.get(key, []), e.get('ts', -1))-1
        owner = None
        if index >= 0:
            _, i, scope = threads[key][index]
            if e['ts']+e.get('dur', 0) <= scope['ts']+scope.get('dur', 0): owner = i
        launches[cid(e)].append((owner, 'graphlaunch' in name, e.get('name')))
    inventory, per_scope = {}, Counter()
    total, known, count = 0.0, 0.0, 0
    for e in events:
        if str(e.get('cat', '')).lower() not in ('kernel', 'gpu_memcpy', 'gpu_memset'): continue
        dur = positive(e.get('dur'), 'GPU duration', zero=True)
        total += dur; count += 1
        options = launches.get(cid(e), [])
        owners = {v[0] for v in options}
        owner = next(iter(owners)) if len(owners) == 1 and None not in owners else None
        graph = bool(options) and any(v[1] for v in options)
        phase = 'unknown'
        if owner is not None:
            per_scope[owner] += 1; known += dur
            m = re.search(r'context_(\d+)', scopes[owner]['name'])
            phase = 'prefill' if m and int(m[1]) else 'decode'
        key = (e.get('name'), e.get('cat'), phase, graph)
        if key not in inventory:
            inventory[key] = dict(kernel=e.get('name'), category=e.get('cat'), phase=phase,
                graph_launch_correlated=graph, count=0, duration_us=0.0,
                shape=None, stride=None, capture_node_mapping='unknown', graph_ids=set())
        row = inventory[key]; row['count'] += 1; row['duration_us'] += dur
        for k, v in e.get('args', {}).items():
            if 'graph' in k.lower(): row['graph_ids'].add(f'{k}={v}')
    issues = []
    if not scopes: issues.append('No native forward annotations; iteration coverage unknown')
    if expected_iterations is not None and len(scopes) != expected_iterations:
        issues.append(f'Expected {expected_iterations} annotated forwards, observed {len(scopes)}')
    if any(not per_scope[i] for i in range(len(scopes))):
        issues.append('Forward annotation without correlated GPU activity, including possible tail loss')
    if not count: issues.append('No GPU activities')
    if isinstance(trace, dict):
        for k, v in trace.items():
            if k != 'traceEvents' and any(x in k.lower() for x in ('warning', 'error', 'dropped')) and v:
                issues.append(f'Trace diagnostic {k}: {str(v)[:500]}')
    rows = []
    for row in inventory.values():
        row['graph_ids'] = sorted(row['graph_ids']); rows.append(row)
    return dict(status='boundary_checked' if not issues else 'incomplete', issues=issues,
        annotated_iterations=len(scopes), annotation_names=[e['name'] for e in scopes],
        iterations_with_gpu=len(per_scope), gpu_activities=count, gpu_activity_sum_us=total,
        attributed_activity_sum_us=known, attribution_fraction=known/total if total else None,
        graph_replay_observed=any(r['graph_launch_correlated'] for r in rows),
        signatures=sorted(rows, key=lambda r: -r['duration_us']),
        full_operator_coverage_verified=False, usable_as_fitting_labels=False,
        warning='GPU activity sums may overlap; boundary checks do not prove every graph node was captured')


def read_trace(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as f: return json.load(f)
