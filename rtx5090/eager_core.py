"""BF16/TP2 contracts. No torch import, clock writes, or inferred hardware passes."""
import json
import math
from pathlib import Path
import struct

from .observe_core import GPU_UUIDS, digest
from .service_core import positive, errors

SCHEMA = 'eager-tp2-bf16-v1'
RATES = (2250, 2527)
UNITS = ['embed'] + [f'layer.{i}' for i in range(64)] + ['norm', 'logits', 'sample']
CAPABILITIES = {
    'gpu_pair': 'requires_loaded_clock_and_restore_acceptance',
    'cpu': 'unsupported_by_current_windows_wsl_backend',
    'emc': 'unsupported_no_jetson_shared_memory_controller',
    'effective_sparsity': 'unverified_use_zero',
    'equation_1_parameters': 'unverified_same_path_calibration_required',
    'equation_2_parameters': 'unverified_voltage_and_identifiability_required',
    'lookahead': 'disabled_pending_serial_and_transition_evidence',
}


def weights(path):
    """Check all indexed tensor headers and file extents, without loading tensors.

    This is not a checksum of tensor payloads or proof of the actual loader.
    """
    root = Path(path).resolve()
    config = json.loads((root/'config.json').read_text(encoding='utf-8'))
    text = config.get('text_config', config)
    if text.get('num_hidden_layers') != 64 or config.get('quantization_config') or text.get('quantization_config'):
        raise ValueError('Require unquantized 64-layer Qwen3.5-27B')
    if 'qwen3_5' not in str(config.get('model_type', '')).lower():
        raise ValueError('Unexpected model_type; audit before loading')
    index_file = root/'model.safetensors.index.json'
    index = json.loads(index_file.read_text(encoding='utf-8'))
    mapping = index.get('weight_map', {})
    if not mapping: raise ValueError('Missing nonempty safetensors weight_map')
    headers, shards, dtypes = {}, [], set()
    sizes = {'BF16': 2, 'F32': 4, 'I64': 8, 'I32': 4, 'I16': 2, 'I8': 1, 'U8': 1, 'BOOL': 1}
    for name in sorted(set(mapping.values())):
        file = (root/name).resolve()
        if not file.is_relative_to(root) or not file.is_file():
            raise ValueError(f'Missing/outside model shard: {name}')
        with file.open('rb') as stream:
            prefix = stream.read(8)
            if len(prefix) != 8: raise ValueError(f'Truncated shard: {name}')
            length = struct.unpack('<Q', prefix)[0]
            if not 2 <= length <= 64*1024*1024: raise ValueError('Invalid safetensors header size')
            raw = stream.read(length)
            header = json.loads(raw)
        intervals = []
        for key, row in header.items():
            if key == '__metadata__': continue
            dtype, shape, offsets = row['dtype'], row['shape'], row['data_offsets']
            if dtype not in sizes or any(type(x) is not int or x < 0 for x in shape):
                raise ValueError(f'Unexpected dtype/shape in {name}: {key}')
            if (len(offsets) != 2 or any(type(x) is not int for x in offsets)
                    or offsets[0] < 0 or offsets[1]-offsets[0] != math.prod(shape)*sizes[dtype]
                    or 8+length+offsets[1] > file.stat().st_size):
                raise ValueError(f'Truncated/invalid tensor extent: {name}: {key}')
            intervals.append(tuple(offsets)); dtypes.add(dtype)
        intervals.sort()
        if any(a[1] > b[0] for a, b in zip(intervals, intervals[1:])):
            raise ValueError('Overlapping tensor extents')
        headers[name] = header
        shards.append(dict(name=name, size=file.stat().st_size, header_sha256=digest(header)))
    for tensor, shard in mapping.items():
        if tensor not in headers[shard]: raise ValueError(f'Indexed tensor absent: {tensor}')
    if 'BF16' not in dtypes: raise ValueError('No BF16 weights found')
    return dict(path=str(root), config_sha256=digest(config), index_sha256=digest(index),
                shards=shards, tensors=len(mapping), dtypes=sorted(dtypes),
                verification='all_indexed_headers_and_extents_not_payload_checksums')


def validate_ranks(rows):
    if len(rows) != 2 or {r['rank'] for r in rows} != {0, 1}:
        raise ValueError('Need both actual ranks')
    for r in rows:
        if r['gpu_uuid'] != GPU_UUIDS[r['rank']] or r['tp_size'] != 2:
            raise ValueError('TP/UUID mapping mismatch')
        if (r['dtype'] != 'torch.bfloat16' or r['load_format'] != 'auto'
                or r['quantization'] is not None or not r['enforce_eager'] or r['v2_runner']):
            raise ValueError('Require real BF16/auto/unquantized/TP2/eager/V1')
        actual = r['actual_config']
        if actual['scheduler_config']['max_num_seqs'] != 1:
            raise ValueError('Concurrency must be one')
        if r['num_hidden_layers'] != 64 or r['units'] != UNITS:
            raise ValueError('Complete model boundary discovery failed')
        if r['speculative_config'] is not None or r['compilation_disabled'] is not True or r['graphs_disabled'] is not True:
            raise ValueError('Speculation/compile/graphs not verified disabled')
        if any(r['communication_env'].values()) or r['nccl_cumem_host_enable'] != '0':
            raise ValueError('Successful NCCL settings were not preserved')
    if len({r['pid'] for r in rows}) != 2 or len({r['session'] for r in rows}) != 1:
        raise ValueError('Workers do not belong to one two-rank session')
    for key in ('actual_config', 'vllm_sources_sha256', 'triton', 'weights', 'tool_hashes'):
        if rows[0][key] != rows[1][key]: raise ValueError(f'Rank environment mismatch: {key}')
    return sorted(rows, key=lambda r: r['rank'])


def context_key(context):
    if (context['prompt_tokens'] != 512 or context['admitted_computed_tokens'] != 0
            or context['resumed'] or context['preempted'] or context['speculative_tokens']):
        raise ValueError('Need a fresh, uncached, non-speculative 512-token request')
    if context['phase'] == 'prefill':
        if context['num_computed_tokens'] != 0 or context['num_scheduled_tokens'] != 512:
            raise ValueError('Partial prefill is recorded but cannot use the full-prefill plan')
        return 'prefill'
    if context['phase'] != 'decode' or context['num_scheduled_tokens'] != 1:
        raise ValueError('Unexpected execution phase')
    offset = context['num_computed_tokens']-512
    if not 0 <= offset < 63: raise ValueError('Decode context outside 512/64 protocol')
    # Exact cache position, never pool different decode states silently.
    return f'decode.{offset}'


def require_gate(gate, identity, telemetry_hash):
    if (gate.get('schema') != SCHEMA or gate.get('identity_fingerprint') != identity
            or gate.get('telemetry_config_sha256') != telemetry_hash
            or gate.get('status') != 'passed' or gate.get('synthetic') is not False):
        raise ValueError('Missing current eager hardware gate')
    for name in ('normal_restoration', 'exception_restoration', 'disconnect_restoration'):
        value = gate.get(name, {})
        evidence = value.get('evidence', value)
        if (not value.get('verified') or not evidence.get('verified')
                or set(evidence.get('by_uuid', {})) != set(GPU_UUIDS)
                or not all(x.get('verified') for x in evidence['by_uuid'].values())):
            raise ValueError(f'Unverified dual-card {name}')
    if {r['target_mhz'] for r in gate.get('cases', []) if r.get('loaded_readback_verified')} != set(RATES):
        raise ValueError('Both frequency pairs must be accepted')


def validate_plan(plan, fingerprint):
    if (plan.get('schema') != SCHEMA or plan.get('identity_fingerprint') != fingerprint
            or plan.get('synthetic') is not False):
        raise ValueError('Plan belongs to a different eager session')
    if plan.get('mode') not in ('serial', 'lookahead'):
        raise ValueError('Unknown runtime mode')
    if plan['mode'] == 'lookahead':
        evidence = plan.get('serial_acceptance', {})
        base = {k: v for k, v in plan.items() if k != 'serial_acceptance'}
        base['mode'] = 'serial'
        if (evidence.get('status') != 'control_completed'
                or evidence.get('identity_fingerprint') != fingerprint
                or evidence.get('plan_sha256') != digest(base)
                or evidence.get('restoration', {}).get('verified') is not True):
            raise ValueError('Lookahead requires matching real serial acceptance and physical restoration')
        if plan.get('lookahead_bound_verified') is not True:
            raise ValueError('Transition bounds cannot resolve lookahead horizon')
    if plan.get('model_validation', {}).get('passed') is not True:
        raise ValueError('Independent model validation is required before deployment')
    validation = plan['model_validation']
    if (validation.get('identity_fingerprint') != fingerprint or validation.get('synthetic') is not False
            or validation.get('model_sha256') != plan.get('model_sha256')
            or validation.get('independent_validation') is not True):
        raise ValueError('Model validation is not bound to this frozen model/session')
    if plan.get('coverage', {}).get('passed') is not True:
        raise ValueError('Two-rank boundary coverage certificate required')
    if plan['coverage'].get('identity_fingerprint') != fingerprint:
        raise ValueError('Boundary coverage belongs to a different session')
    if set(plan.get('temperature_ranges', {})) != {'0', '1'}:
        raise ValueError('Calibrated temperature ranges for both ranks required')
    for limits in plan['temperature_ranges'].values():
        if len(limits) != 2 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in limits) or limits[0] >= limits[1]:
            raise ValueError('Invalid calibrated temperature range')
    if plan.get('cpu') is not None or plan.get('emc') is not None:
        raise ValueError('No CPU/EMC backend has been accepted on this platform')
    expected = {'prefill'} | {f'decode.{i}' for i in range(63)}
    if set(plan['contexts']) != expected: raise ValueError('Incomplete context coverage')
    for key, blocks in plan['contexts'].items():
        if [u for b in blocks for u in b['units']] != UNITS:
            raise ValueError(f'Duplicate, missing or reordered boundaries in {key}')
        for b in blocks:
            if not b['units'] or type(b['mhz']) is not int or b['mhz'] not in RATES: raise ValueError('Invalid block')
            positive(b['estimated_s'], 'block duration')
        if plan['mode'] == 'lookahead' and any(b['estimated_s'] <= 5*plan['switch_s'] for b in blocks[:-1]):
            raise ValueError('Lookahead block cannot amortize the measured switching bound')
    return plan


def freeze_window_model(data):
    """512-only empirical benchmark, deliberately not Eq1/Eq2 or a block model."""
    check_dataset(data, 'train')
    targets = ('ttft_s', 'tpot_s', 'window_s', 'energy_j')
    from statistics import mean
    cells = {f: {t: mean(w[t] for w in data['windows'] if w['frequency'] == f) for t in targets}
             for f in ('auto', '2250', '2527')}
    feasible = [f for f in ('2250', '2527') if cells[f]['window_s'] <= cells['auto']['window_s']*1.05]
    return dict(schema=SCHEMA, identity_fingerprint=data['identity_fingerprint'],
        protocol=data['protocol'], source_sha256=digest(data),
        train_prompt_hashes=sorted({p for w in data['windows'] for p in w['prompt_hashes']}),
        scope='512_request_windows_empirical_not_equation1_or_equation2', cells=cells,
        best_fixed=min(feasible, key=lambda f: (cells[f]['energy_j'], cells[f]['window_s'], f)) if feasible else None,
        block_selection_authorized=False)


def check_dataset(data, split):
    if data.get('schema') != SCHEMA or data.get('split') != split or data.get('status') != 'collected' or data.get('synthetic') is not False:
        raise ValueError('Require a complete real dataset with the declared split')
    if data.get('restoration', {}).get('verified') is not True:
        raise ValueError('Current physical restoration not verified')
    protocol = data['protocol']
    if protocol != dict(input=512, output=64, concurrency=1, windows=3, requests=protocol.get('requests')) or protocol['requests'] < 20:
        raise ValueError('Unexpected frozen experiment protocol')
    counts, prompts = {}, {}
    for w in data['windows']:
        f = w['frequency']
        if f not in ('auto', '2250', '2527'): raise ValueError('Unexpected frequency')
        counts[f] = counts.get(f, 0)+1
        if w['status'] != 'valid' or w['quality_errors'] or w['cache_status'] != 'verified_miss':
            raise ValueError('Failed samples must not be discarded to fit a model')
        if w['requests'] != protocol['requests'] or len(set(w['prompt_hashes'])) != w['requests']:
            raise ValueError('Request count/uniqueness mismatch')
        if w.get('energy_relative_uncertainty', 1) > .03: raise ValueError('Energy budget exceeded')
        for t in ('ttft_s', 'tpot_s', 'window_s', 'energy_j'): positive(w[t], t)
        if set(w.get('energy_by_uuid_j', {})) != set(GPU_UUIDS): raise ValueError('Missing board energy')
        for u in GPU_UUIDS: positive(w['energy_by_uuid_j'][u], 'board energy')
        if not math.isclose(sum(w['energy_by_uuid_j'].values()), w['energy_j'], rel_tol=1e-8):
            raise ValueError('Energy sum mismatch')
        prompts.setdefault(f, []).extend(w['prompt_hashes'])
    if counts != {'auto': 3, '2250': 3, '2527': 3}: raise ValueError('Incomplete 3x3 matrix')
    if any(len(set(p)) != len(p) for p in prompts.values()): raise ValueError('Reused requests within condition')
    if any(set(p) != set(prompts['auto']) for p in prompts.values()): raise ValueError('Unpaired conditions')


def validate_window_model(data, model):
    check_dataset(data, 'validation')
    if data['identity_fingerprint'] != model['identity_fingerprint'] or data['protocol'] != model['protocol']:
        raise ValueError('Frozen model environment/protocol changed')
    if set(model['train_prompt_hashes']) & {p for w in data['windows'] for p in w['prompt_hashes']}:
        raise ValueError('Training/validation leakage')
    scores = {}
    for f, targets in model['cells'].items():
        rows = [w for w in data['windows'] if w['frequency'] == f]
        scores[f] = {t: errors([w[t] for w in rows], [v]*len(rows)) for t, v in targets.items()}
    passed = all(s['wape'] <= .10 and s['p90_relative_error'] <= .20 for v in scores.values() for s in v.values())
    from statistics import mean
    observed = {f: {t: mean(w[t] for w in data['windows'] if w['frequency'] == f) for t in ('window_s', 'energy_j')} for f in scores}
    comparisons = {f: dict(energy_saving=1-v['energy_j']/observed['auto']['energy_j'],
        latency_change=v['window_s']/observed['auto']['window_s']-1) for f, v in observed.items() if f != 'auto'}
    return dict(schema=SCHEMA, passed=passed, scores=scores, comparisons=comparisons,
        frozen_best_fixed=model['best_fixed'], model_sha256=digest(model),
        data_sha256=digest(data), scope=model['scope'], block_selection_authorized=False)
