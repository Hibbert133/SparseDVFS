"""Pure-Python dispatcher layout and profiler evidence contracts (no GPU imports)."""
import gzip
import json
import math
from collections import defaultdict
from .contracts import digest_json

MARKER = 'sdvfs.op|'


def context_key(context):
    return '{}:{}:{}'.format(context['phase'], context['num_computed_tokens'],
                            context['num_scheduled_tokens'])


def layout_entry(parent, ordinal, overload, metadata):
    return dict(name=f'{parent}::{overload}#{ordinal}', parent=parent, ordinal=ordinal,
                overload=overload, metadata=metadata, opaque_fused=not overload.startswith('aten.'))


def layout_hash(layout):
    return digest_json(layout)


def marker_name(request_id, step, ordinal):
    # Request IDs may contain arbitrary characters; hash instead of parsing them.
    return f'{MARKER}{digest_json(str(request_id))[:20]}|{step}|{ordinal}'


def validate_live(layout, ordinal, entry):
    if ordinal >= len(layout) or layout[ordinal] != entry:
        raise RuntimeError(f'Unknown/changed dispatcher signature at ordinal {ordinal}; stop and retrace')


def read_chrome(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as source:
        data = json.load(source)
    return data['traceEvents'] if isinstance(data, dict) else data


def _correlation(event):
    args = event.get('args', {})
    for key in ('correlation', 'correlation id', 'correlationId', 'Correlation ID'):
        if key in args:
            return str(args[key])
    return None


def correlate(events, expected_markers):
    """Join CPU CUDA launches to GPU activities by correlation ID.

    CPU containment is used only between the launch and a record_function scope
    on the SAME process/thread. GPU timestamps are never compared to host scope
    timestamps. Unknown trace dialects fail closed rather than infer coverage.
    """
    markers = [e for e in events if e.get('ph') == 'X' and
               str(e.get('name', '')).startswith(MARKER)]
    found = {e['name'] for e in markers}
    if found != set(expected_markers) or len(found) != len(markers):
        raise ValueError('Missing/duplicate profiler operator markers')
    by_thread = defaultdict(list)
    for event in markers:
        by_thread[(event.get('pid'), event.get('tid'))].append(event)
    launches, failures = {}, []
    for event in events:
        name, category = str(event.get('name', '')), str(event.get('cat', '')).lower()
        is_launch = ('cuda_runtime' in category or 'cuda_driver' in category) and any(
            word in name.lower() for word in ('launchkernel', 'launchcooperative', 'memcpy', 'memset'))
        if not is_launch:
            continue
        cid = _correlation(event)
        start, finish = event.get('ts', -1), event.get('ts', -1) + event.get('dur', 0)
        owners = [m for m in by_thread[(event.get('pid'), event.get('tid'))]
                  if m['ts'] <= start and finish <= m['ts'] + m.get('dur', 0)]
        if cid is None or len(owners) != 1 or cid in launches:
            failures.append('unattributed/ambiguous CUDA launch ' + name)
        else:
            launches[cid] = owners[0]['name']
    ownership, streams, devices = defaultdict(list), set(), set()
    gpu_count = 0
    for event in events:
        category = str(event.get('cat', '')).lower()
        if category not in ('kernel', 'gpu_memcpy', 'gpu_memset'):
            continue
        gpu_count += 1
        cid = _correlation(event)
        args = event.get('args', {})
        if cid not in launches or 'stream' not in args or 'device' not in args:
            failures.append('unattributed GPU activity ' + str(event.get('name')))
            continue
        duration = event.get('dur', -1)
        if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0:
            failures.append('invalid GPU activity duration')
            continue
        streams.add(str(args['stream']))
        devices.add(str(args['device']))
        ownership[launches[cid]].append(dict(name=event.get('name'), duration_us=duration,
                                            stream=str(args['stream']), device=str(args['device'])))
    if failures or gpu_count == 0 or len(streams) != 1 or len(devices) != 1:
        raise ValueError('CUDA coverage failed: ' + '; '.join(failures[:5]) +
                         f'; activities={gpu_count}, streams={streams}, devices={devices}')
    return dict(ownership=dict(ownership), gpu_activities=gpu_count,
                streams=sorted(streams), devices=sorted(devices), fully_attributed=True)


def build_certificate(trace, events, *, synthetic=False, minimum_requests=3):
    steps = trace['operator_steps']
    markers = [entry['marker'] for step in steps for entry in step['calls']]
    evidence = correlate(events, markers)
    requests = defaultdict(list)
    for step in steps:
        requests[step['request_id']].append(step)
    if len(requests) < minimum_requests:
        raise ValueError('Coverage requires three complete post-warmup requests')
    cases, request_cases = {}, []
    workload = trace['identity'].get('workload', {})
    expected_steps, prompt_tokens = workload.get('output_tokens'), workload.get('input_tokens')
    if not isinstance(expected_steps, int) or expected_steps < 2 or not isinstance(prompt_tokens, int):
        raise ValueError('Coverage identity needs the fixed input/output token counts')
    if not trace.get('output_sha256') or trace.get('native_output_sha256') != trace['output_sha256']:
        raise ValueError('Operator outputs must match the native smoke reference')
    for rid, rows in requests.items():
        if len(rows) != expected_steps or [r['step'] for r in rows] != list(range(expected_steps)):
            raise ValueError('Incomplete/out-of-order scheduler steps')
        keys = []
        for row in rows:
            expected_phase = 'prefill' if row['step'] == 0 else 'decode'
            expected_computed = 0 if row['step'] == 0 else prompt_tokens+row['step']-1
            expected_scheduled = prompt_tokens if row['step'] == 0 else 1
            if (row['phase'], row['num_computed_tokens'], row['num_scheduled_tokens']) != (
                    expected_phase, expected_computed, expected_scheduled):
                raise ValueError('Unexpected scheduler context in coverage evidence')
            key = context_key(row)
            keys.append(key)
            layout = [call['layout'] for call in row['calls']]
            names = [call['layout']['name'] for call in row['calls']
                     if call['marker'] in evidence['ownership']]
            if not layout or not names or len(names) != len(set(names)):
                raise ValueError('Empty/duplicate operator execution layout')
            if key in cases and (cases[key]['layout'] != layout or cases[key]['units'] != names):
                raise ValueError('Dispatcher layout/GPU ownership changed across requests')
            cases.setdefault(key, dict(layout=layout, units=names, layout_sha256=layout_hash(layout), requests=0))
            cases[key]['requests'] += 1
        request_cases.append(keys)
    if any(keys != request_cases[0] for keys in request_cases):
        raise ValueError('Request context coverage differs')
    if any(case['requests'] < minimum_requests for case in cases.values()):
        raise ValueError('A context has insufficient repeated coverage')
    result = dict(schema=5, granularity='operator', identity=trace['identity'],
                  synthetic=bool(synthetic), deployable=not synthetic, cases=cases,
                  request_count=len(requests), context_order=request_cases[0],
                  evidence=dict(fully_attributed=True, gpu_activities=evidence['gpu_activities'],
                                streams=evidence['streams'], devices=evidence['devices'],
                                method='same-thread launch containment plus correlation ID',
                                auxiliary_streams_supported=False),
                  output_sha256=trace['output_sha256'], native_output_sha256=trace['native_output_sha256'],
                  graph_scope='dispatcher units serialized on one observed GPU stream; fused units atomic')
    result['certificate_sha256'] = digest_json(result)
    return result


def validate_certificate(certificate, identity, *, allow_synthetic=False):
    c = dict(certificate)
    checksum = c.pop('certificate_sha256', None)
    if checksum != digest_json(c):
        raise ValueError('Coverage certificate checksum mismatch')
    if c.get('schema') != 5 or c.get('granularity') != 'operator' or c.get('identity') != identity:
        raise ValueError('Coverage certificate belongs to another version/granularity/identity')
    if (c.get('synthetic') or not c.get('deployable')) and not allow_synthetic:
        raise ValueError('Synthetic/incomplete coverage cannot authorize DVFS')
    e = c.get('evidence', {})
    if (not e.get('fully_attributed') or len(e.get('streams', [])) != 1 or
            len(e.get('devices', [])) != 1 or c.get('request_count', 0) < 3):
        raise ValueError('Coverage evidence is incomplete or uses multiple streams/devices')
    if not c.get('cases') or set(c['context_order']) != set(c['cases']):
        raise ValueError('Coverage context order is incomplete')
    if len(c['context_order']) != identity.get('workload', {}).get('output_tokens'):
        raise ValueError('Coverage omits scheduler steps')
    if not c.get('output_sha256') or c.get('native_output_sha256') != c['output_sha256']:
        raise ValueError('Native/dispatcher output comparison is missing')
    for case in c['cases'].values():
        if case['requests'] < 3 or case['layout_sha256'] != layout_hash(case['layout']):
            raise ValueError('Insufficient or changed context evidence')
        names = [entry['name'] for entry in case['layout']]
        if case['units'] != [name for name in names if name in case['units']]:
            raise ValueError('GPU units are not an ordered subset of dispatcher calls')
    return certificate
