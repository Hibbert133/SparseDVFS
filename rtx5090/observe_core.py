"""CPU-only contracts for target-vLLM observation. No model or DVFS imports."""
from bisect import bisect_right
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

SCHEMA = 'eq1-vllm-observation-v1'
MARKER = 'eq1.observe|'
GPU_UUIDS = ['GPU-22d1cd22-d198-45c4-2b54-974484026ab5',
             'GPU-e2f02606-bac6-52cd-f8a1-15a027db7da0']
DEVELOPMENT = (128, 512, 2048)
HOLDOUT = 1024


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def canonical_uuid(value):
    value = str(value)
    return value if value.startswith('GPU-') else 'GPU-' + value


def validate_ranks(identities, expected=GPU_UUIDS):
    if len(identities) != 2 or {x['rank'] for x in identities} != {0, 1}:
        raise ValueError('Require both TP ranks 0 and 1')
    for row in identities:
        if row['gpu_uuid'].lower() != expected[row['rank']].lower():
            raise ValueError('TP rank/GPU UUID mapping differs from requested order')
        if row['tp_size'] != 2 or row['v2_runner'] or row['dtype'] != 'torch.bfloat16':
            raise ValueError('Require BF16 TP=2 V1 workers')
        if not row['enforce_eager'] or row['load_format'] != 'dummy' or row['quantization'] is not None:
            raise ValueError('Require eager dummy unquantized model')
        if row.get('communication_env') != dict(NCCL_P2P_DISABLE='1', NCCL_SHM_DISABLE='1', NCCL_IB_DISABLE='1'):
            raise ValueError('Worker NCCL environment differs from the supplied working launch')
    if len({x['pid'] for x in identities}) != 2:
        raise ValueError('TP worker PIDs must be distinct')
    return sorted(identities, key=lambda x: x['rank'])


class PhaseTracker:
    """Track scheduler counters, including partial and cached prefill."""
    def __init__(self):
        self.requests = {}

    def step(self, output):
        for rid in output.finished_req_ids:
            self.requests.pop(rid, None)
        for req in output.scheduled_new_reqs:
            if req.prompt_token_ids is None or req.mm_features:
                raise ValueError('Observation requires text token-ID requests')
            self.requests[req.req_id] = dict(prompt=len(req.prompt_token_ids),
                computed=req.num_computed_tokens, step=0,
                admitted_computed=req.num_computed_tokens)
        cached = output.scheduled_cached_reqs
        if len(cached.req_ids) != len(cached.num_computed_tokens):
            raise ValueError('Malformed scheduler counters')
        for rid, count in zip(cached.req_ids, cached.num_computed_tokens):
            if rid not in self.requests:
                raise ValueError('Observation began mid-request; start before generate')
            self.requests[rid]['computed'] = count
        scheduled = {k: v for k, v in output.num_scheduled_tokens.items() if v > 0}
        if sum(scheduled.values()) != output.total_num_scheduled_tokens:
            raise ValueError('Inconsistent scheduler token counts')
        if not scheduled:
            return None
        if len(scheduled) != 1:
            raise ValueError('This inventory protocol requires concurrency=1')
        rid, count = next(iter(scheduled.items()))
        state = self.requests[rid]
        computed, prompt = state['computed'], state['prompt']
        phase = 'prefill' if computed < prompt else 'decode'
        if computed < prompt < computed + count:
            phase = 'mixed'
        context = dict(request_id=rid, phase=phase, step=state['step'],
            prompt_tokens=prompt, num_computed_tokens=computed, num_scheduled_tokens=count,
            admitted_computed_tokens=state['admitted_computed'],
            resumed=rid in cached.resumed_req_ids,
            preempted=rid in (getattr(output, 'preempted_req_ids', None) or set()),
            speculative_tokens=len(output.scheduled_spec_decode_tokens.get(rid, [])))
        state.update(computed=computed + count, step=state['step'] + 1)
        return context


def category_hint(name):
    name = name.lower()
    if any(x in name for x in ('all_reduce', 'all_gather', 'reduce_scatter', 'nccl', 'allreduce')):
        return 'communication'
    if any(x in name for x in ('copy_', '_to_copy', 'memcpy', 'memset')):
        return 'data_movement'
    if name.startswith(('aten.mm.', 'aten.bmm.', 'aten.addmm.')):
        return 'matrix_multiply'
    if any(x in name for x in ('norm', 'silu', 'gelu', 'relu', 'aten.add.', 'aten.mul.')):
        return 'elementwise_or_normalization'
    return 'unknown'


def signature(row):
    # Context is retained conservatively for opaque ops: cache length and state
    # can change their work even when all visible tensor shapes are unchanged.
    value = {k: row[k] for k in ('rank', 'phase', 'overload', 'inputs', 'outputs')}
    value['kernel_names'] = sorted({k['name'] for k in row.get('gpu_activities', [])})
    if row['opaque']:
        value['execution_context'] = {k: row['context'].get(k) for k in
            ('num_computed_tokens', 'num_scheduled_tokens', 'prompt_tokens')}
    return digest(value), value


def correlate(events, rows):
    """CPU containment + launch correlation. Unknown activity stays unknown.

    No GPU-vs-CPU timestamp containment, no single-stream assumption. A
    correlation collision is ambiguous, even if its duration looks plausible.
    """
    expected = {r['marker']: r for r in rows}
    if len(expected) != len(rows):
        raise ValueError('Duplicate dispatcher markers')
    scopes, seen = defaultdict(list), Counter()
    for e in events:
        # Kineto can emit a same-name gpu_user_annotation span. Only the
        # CPU record_function scope owns CUDA launch API calls. GPU spans
        # neither duplicate that scope nor count as additional kernel work.
        if (e.get('ph') == 'X' and e.get('cat') == 'user_annotation'
                and e.get('name') in expected):
            scopes[(e.get('pid'), e.get('tid'))].append(e)
            seen[e['name']] += 1
    if set(seen) != set(expected) or any(v != 1 for v in seen.values()):
        missing = len(set(expected) - set(seen))
        duplicate = sum(v != 1 for v in seen.values())
        raise ValueError('Profiler is missing or duplicating dispatcher scopes '
                         f'(CPU user_annotation: missing={missing}, duplicate={duplicate})')
    starts = {}
    for thread, entries in scopes.items():
        entries.sort(key=lambda e: e['ts'])
        starts[thread] = [e['ts'] for e in entries]
        for a, b in zip(entries, entries[1:]):
            if a['ts'] + a.get('dur', 0) > b['ts']:
                raise ValueError('Overlapping dispatcher scopes cannot be uniquely attributed')
    def correlation(e):
        for key in ('correlation', 'correlation id', 'correlationId', 'Correlation ID'):
            if key in e.get('args', {}):
                return str(e['args'][key])
        return None
    launches = defaultdict(list)
    for e in events:
        cat = str(e.get('cat', '')).lower()
        name = str(e.get('name', '')).lower()
        if not ('cuda_runtime' in cat or 'cuda_driver' in cat):
            continue
        if not any(k in name for k in ('launchkernel', 'launchcooperative', 'memcpy', 'memset')):
            continue
        cid = correlation(e)
        if cid is None:
            continue
        thread = (e.get('pid'), e.get('tid'))
        owner = None
        index = bisect_right(starts.get(thread, []), e.get('ts', -1)) - 1
        if index >= 0:
            scope = scopes[thread][index]
            if e['ts'] + e.get('dur', 0) <= scope['ts'] + scope.get('dur', 0):
                owner = scope['name']
        launches[cid].append(owner)
    activities, unknown = [], []
    for e in events:
        cat = str(e.get('cat', '')).lower()
        if cat not in ('kernel', 'gpu_memcpy', 'gpu_memset'):
            continue
        duration = e.get('dur')
        if not isinstance(duration, (float, int)) or not math.isfinite(duration) or duration < 0:
            raise ValueError('Invalid GPU activity duration')
        args = e.get('args', {})
        owners = launches.get(correlation(e), [])
        # Multiple API layers can refer to the same launch. Only an identical,
        # non-null owner is unambiguous; conflicting/null scopes remain unknown.
        owner = owners[0] if owners and owners[0] is not None and len(set(owners)) == 1 else None
        activity = dict(name=e.get('name'), category=cat, duration_us=duration,
                        stream=args.get('stream'), device=args.get('device'),
                        correlation=correlation(e), marker=owner)
        activities.append(activity)
        if owner is None:
            unknown.append(activity)
        else:
            expected[owner].setdefault('gpu_activities', []).append(activity)
    total = sum(x['duration_us'] for x in activities)
    unmatched = sum(x['duration_us'] for x in unknown)
    return dict(gpu_activities=len(activities), gpu_activity_sum_us=total,
        attributed_activity_sum_us=total-unmatched, unattributed_activities=unknown,
        attribution_fraction=(total-unmatched)/total if total else None,
        denominator='sum_of_GPU_activity_durations_not_request_wall_time',
        fully_attributed=bool(activities) and not unknown)


def inventory(requests):
    """Aggregate observed signatures, never declare replay support."""
    grouped, total, attributed = {}, 0.0, 0.0
    phase_calls, rank_activity = Counter(), Counter()
    for request in requests:
        if request.get('split') != 'development' or request['prompt_tokens'] not in DEVELOPMENT:
            raise ValueError('Holdout data cannot enter the development inventory')
        if request.get('status') != 'complete':
            raise ValueError('Incomplete request cannot enter inventory')
        total += request['attribution']['gpu_activity_sum_us']
        attributed += request['attribution']['attributed_activity_sum_us']
        rank_activity[str(request['rank'])] += request['attribution']['gpu_activity_sum_us']
        for row in request['operators']:
            phase_calls[row['phase']] += 1
            key, value = signature(row)
            if key not in grouped:
                grouped[key] = dict(signature_id=key, signature=value, calls=0, parents=set(),
                    sources=[], gpu_activity_sum_us=0.0, stream_event_sum_ms=0.0,
                    category_hint=category_hint(row['overload']),
                    classification_basis='operator_name_heuristic_not_bottleneck_evidence',
                    replay_status='not_implemented', equation_validated=False,
                    opaque=row['opaque'])
            group = grouped[key]
            group['calls'] += 1
            group['parents'].add(row['parent'])
            source = request['request_key']
            if source not in group['sources']:
                group['sources'].append(source)
            group['gpu_activity_sum_us'] += sum(x['duration_us'] for x in row.get('gpu_activities', []))
            group['stream_event_sum_ms'] += row.get('stream_event_ms') or 0.0
    for group in grouped.values():
        group['parents'] = sorted(group['parents'])
        group['fraction_of_all_gpu_activity'] = group['gpu_activity_sum_us']/total if total else None
    return dict(schema=SCHEMA, status='inventory_only', split='development',
        signatures=sorted(grouped.values(), key=lambda g: -g['gpu_activity_sum_us']),
        summary=dict(unique_signatures=len(grouped), requests=len(requests),
                     observed_calls=sum(g['calls'] for g in grouped.values()),
                     gpu_activity_sum_us=total, attributed_activity_sum_us=attributed,
                     calls_by_phase=dict(phase_calls), gpu_activity_us_by_rank=dict(rank_activity),
                     replay_supported_calls=0, equation_validated_calls=0),
        warning='Profiler/event sums can overlap; not end-to-end latency or calibrated Eq1 data')
