"""Offline pair switching and block-size sensitivity. No hardware authorization.

Only stdlib: Windows Python 3.8 and WSL can read the same saved artifacts.
"""
import argparse
import hashlib
import json
import math
import sqlite3
from pathlib import Path
from statistics import median

UUIDS = ('GPU-22d1cd22-d198-45c4-2b54-974484026ab5',
         'GPU-e2f02606-bac6-52cd-f8a1-15a027db7da0')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def decode(raw, unit, unsigned32):
    if unsigned32 and isinstance(raw, int) and raw < 0:
        if raw < -(2 ** 31):
            raise ValueError('Value is outside signed int32')
        raw += 2 ** 32
    value = raw / (1e6 if unit == 'hz' else 1)
    if not math.isfinite(value) or not 0 <= value <= 5000:
        raise ValueError('Implausible frequency; check explicit unit/encoding')
    return value


def sustained(data, predicate, duration_ns):
    start = previous = None
    for t, value in data:
        if previous is not None and (t <= previous or t - previous > 3000000):
            start = None
        if predicate(value):
            if start is None:
                start = t
            if t - start >= duration_ns:
                return dict(start_ns=start, confirmed_ns=t)
        else:
            start = None
        previous = t
    return None


def pair_timing(begin, cards):
    """Both ranks release together: max absolute completion, not max duration."""
    if len(cards) != 2 or {c['uuid'] for c in cards} != set(UUIDS):
        raise ValueError('Require exactly both UUIDs')
    result = {}
    for field in ('midpoint', 'plateau'):
        for event in ('start_ns', 'confirmed_ns'):
            result[field + '_' + event.replace('_ns', '_ms')] = (
                (max(c[field][event] for c in cards) - begin) / 1e6
                if all(c[field] is not None for c in cards) else None)
    return result


def sensitivity(model, request_s=None):
    rows = []
    for switch_ms in (5, 10, 20, 50):
        # Eq.(3) uses strict >. These are hypotheses, not measured bounds.
        threshold = 5 * switch_ms / 1000
        calls = {}
        for key, op in model['operators'].items():
            t = op['latency_s']
            if not math.isfinite(t) or t <= 0:
                raise ValueError('Invalid frozen latency')
            calls[key] = math.floor(threshold / t) + 1
        rows.append(dict(assumed_pair_switch_ms=switch_ms,
                         strict_block_duration_gt_ms=threshold * 1000,
                         repeated_same_shape_calls_for_strict_eq3=calls,
                         hypothetical_5pct_switch_count_ceiling=(
                             math.floor(.05 * request_s / (switch_ms / 1000))
                             if request_s is not None else None)))
    return rows


def analyze(capture, stage, unit, unsigned32, request_s=None):
    report = read(capture / 'report.json')
    if (report['status'] != 'comparison_collected' or not report['restoration_verified']
            or report.get('control_backend') != 'persistent_windows_bridge'):
        raise ValueError('Require completed and restored persistent bridge collection')
    frozen = read(stage / 'stage-freeze.json')
    model = frozen['models']['linear_latency']
    if model['identity_fingerprint'] != frozen['identity_fingerprint']:
        raise ValueError('Frozen model identity mismatch')
    if model['schema'] != 'linear-latency-table-v1' or len(model['operators']) != 8:
        raise ValueError('Require frozen eight-case linear latency model')
    anchors = report['clock_anchors']
    offsets = sorted(a['wall_time_ns'] - (a['perf_before_ns'] + a['perf_after_ns']) // 2
                     for a in anchors)
    offset = offsets[len(offsets) // 2]
    dbpath = capture / 'gpu-clock.sqlite'
    db = sqlite3.connect(dbpath.resolve().as_uri() + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    try:
        diagnostics = [dict(x) for x in db.execute('SELECT * FROM DIAGNOSTIC_EVENT')]
        epoch = db.execute('SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME').fetchone()[0]
        definitions = list(db.execute("SELECT DISTINCT typeId,metricId FROM TARGET_INFO_GPU_METRICS WHERE metricName LIKE '%GPC Clock%'"))
        gpus = [dict(x) for x in db.execute('SELECT id,uuid,busLocation FROM TARGET_INFO_GPU')]
        traces, mapping = {}, []
        for definition in definitions:
            tid, mid = definition['typeId'], definition['metricId']
            matches = [g for g in gpus if g['id'] == (tid & 255)]
            if len(matches) != 1:
                raise ValueError('Ambiguous GPU mapping')
            gpu = matches[0]
            uuid = gpu['uuid'] if gpu['uuid'].startswith('GPU-') else 'GPU-' + gpu['uuid']
            if uuid not in UUIDS or uuid in traces:
                raise ValueError('Unexpected/duplicate GPU metric')
            mapping.append(dict(type_id=tid, metric_id=mid, uuid=uuid, pci=gpu['busLocation']))
            traces[uuid] = [(x['timestamp'] + epoch - offset, decode(x['value'], unit, unsigned32))
                            for x in db.execute('SELECT timestamp,value FROM GPU_METRICS WHERE typeId=? AND metricId=? ORDER BY timestamp', (tid, mid))]
        if set(traces) != set(UUIDS):
            raise ValueError('Missing paired GPC traces')
    finally:
        db.close()
    transitions = []
    for transition in report['transitions']:
        begin, end = transition['begin_ns'], transition['end_ns']
        cards = []
        for uuid in UUIDS:
            commands = [c for c in report['commands'] if c['uuid'] == uuid and c['option'] == '-lgc'
                        and begin <= c['begin_ns'] <= end]
            if len(commands) != 1 or commands[0]['returncode'] != 0:
                raise ValueError('Missing successful per-card command')
            command = commands[0]
            if command['value'] != '%d,%d' % ((transition['requested_mhz'],) * 2):
                raise ValueError('Command target mismatch')
            b = command['begin_ns']
            pre = [v for t, v in traces[uuid] if b - 200000000 <= t < b]
            post = sorted(v for t, v in traces[uuid] if b + 2000000000 <= t < b + 4000000000)
            if len(pre) < 100 or len(post) < 1000 or b + 4000000000 > end:
                raise ValueError('Insufficient baseline/plateau coverage')
            old, new = median(pre), median(post)
            if abs(new - old) <= 60:
                raise ValueError('Insufficient plateau separation')
            direction = 1 if new > old else -1
            lo, hi = post[int((len(post) - 1) * .1)] - 15, post[int((len(post) - 1) * .9)] + 15
            if not (hi < old - 30 if direction < 0 else lo > old + 30):
                raise ValueError('Plateau band overlaps old platform')
            segment = [(t, v) for t, v in traces[uuid] if b <= t < b + 4000000000]
            cards.append(dict(uuid=uuid, api_ms=command['command_ms'], before_mhz=old,
                              after_mhz=new, plateau_band_mhz=[lo, hi],
                              max_sample_gap_ms=max(y[0] - x[0] for x, y in zip(segment, segment[1:])) / 1e6,
                              midpoint=sustained(segment, lambda v: direction * (v - (old + new) / 2) >= 0, 2000000),
                              plateau=sustained(segment, lambda v: lo <= v <= hi, 10000000)))
        transitions.append(dict(requested_mhz=transition['requested_mhz'], cards=cards,
                                pair=pair_timing(begin, cards)))
    return dict(schema='switch-partition-screen-v1', clock_writes=False, deployable=False,
                identity_fingerprint=frozen['identity_fingerprint'],
                capture_identity_binding_verified=False,
                metric_mapping=mapping, diagnostics=diagnostics,
                candidate_decoding=dict(value_unit=unit, unsigned32=unsigned32, validated=False),
                anchor_offset_spread_ms=(max(offsets) - min(offsets)) / 1e6,
                transitions=transitions, sensitivity=sensitivity(model, request_s),
                hypothetical_request_s=request_s,
                blockers=['per-unit two-frequency latency/energy missing',
                          'whole-block timing and legal dual-rank boundaries unverified',
                          'switch bound, metric decoding, capture identity and cross-clock alignment unverified'],
                limitations=['GPC midpoint is not hardware completion',
                             'plateau is retrospective p10/p90 +/-15MHz lasting 10ms; no activity gate',
                             'same-shape repeated linear counts are scale illustrations, not full model layers',
                             'switch budget excludes compute slowdown, coordination, initial set and final restore'],
                evidence_sha256={str(x.resolve()): sha(x) for x in
                                 (capture / 'report.json', dbpath, stage / 'stage-freeze.json')})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('capture', 'stage', 'out'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--value-unit', choices=('hz', 'mhz'), required=True)
    p.add_argument('--unsigned32', action='store_true')
    p.add_argument('--hypothetical-request-s', type=float)
    a = p.parse_args()
    if a.hypothetical_request_s is not None and (not math.isfinite(a.hypothetical_request_s) or a.hypothetical_request_s <= 0):
        p.error('Hypothetical request duration must be positive and finite')
    result = analyze(Path(a.capture), Path(a.stage), a.value_unit, a.unsigned32, a.hypothetical_request_s)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    lines = ['Offline pair timing and partition sensitivity; no live access or deployment.',
             'Candidate decoding only; alignment spread %.3f ms.' % result['anchor_offset_spread_ms'],
             'target | paired midpoint ms | paired plateau start ms | paired confirmation ms']
    for t in result['transitions']:
        v = t['pair']
        lines.append('%s | %s | %s | %s' % (t['requested_mhz'], v['midpoint_start_ms'], v['plateau_start_ms'], v['plateau_confirmed_ms']))
    lines.append('Assumed pair switch ms | Eq3 block strictly longer than ms | hypothetical 5% switch ceiling')
    for s in result['sensitivity']:
        lines.append('%s | %s | %s' % (s['assumed_pair_switch_ms'], s['strict_block_duration_gt_ms'], s['hypothetical_5pct_switch_count_ceiling']))
    lines.append('Per-case repeated-call scale estimates are in audit.json; not executable blocks.')
    lines.extend('Blocked: ' + s for s in result['blockers'])
    text = '\n'.join(lines)
    (out / 'REPORT.txt').write_text(text + '\n', encoding='utf-8')
    print(text)
    print('Saved:', out)


if __name__ == '__main__':
    main()
