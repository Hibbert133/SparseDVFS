"""Same-path Eq1/Eq2 calibration and Algorithm 1 adapter for two GPU domains.

Input power labels MUST be independently measured repeated-unit board windows.
The eager diagnostic trace is intentionally not automatically converted into
these inputs. Unknown replay/state/voltage cannot become a deployable model.
"""
from dataclasses import dataclass
from statistics import mean

from sparsedvfs.paper_core import timeline_seconds, thermal_power_w, OrderedDAG, greedy_partition
from sparsedvfs.calibration_fit import least_squares
from . import eager_core as core
from .observe_core import digest, GPU_UUIDS
from .service_core import positive, errors


@dataclass(frozen=True)
class PairFrequency:
    values: tuple

    @classmethod
    def of(cls, value):
        if isinstance(value, cls): return value
        values = tuple(value)
        if len(values) != 2 or any(type(v) is not int for v in values) or values[0] != values[1] or values[0] not in core.RATES:
            raise ValueError('Only accepted equal GPU pairs are allowed')
        return cls(values)

    def conservative_merge(self, other):
        return PairFrequency.of(tuple(max(a, b) for a, b in zip(self.values, other.values)))


def check_data(data, split, allow_synthetic=False):
    if data.get('schema') != core.SCHEMA or data.get('split') != split:
        raise ValueError('Wrong equation dataset schema/split')
    if data.get('synthetic') is not False and not allow_synthetic:
        raise ValueError('Synthetic data cannot authorize hardware')
    if data.get('scope') != 'same_path_repeated_unit_board_windows':
        raise ValueError('Request-average energy and diagnostic stream events are not equation labels')
    if data.get('restoration', {}).get('verified') is not True:
        raise ValueError('Missing physical restoration')
    if data.get('units') != dict(latency='s', power='W', voltage='V', temperature='C', frequency='MHz', work='FLOP', data='byte'):
        raise ValueError('Explicit physical units required')
    if not data.get('records'): raise ValueError('No measured records')
    groups = {}
    ids = set()
    for row in data['records']:
        if row.get('rank') not in (0, 1) or row['uuid'] != GPU_UUIDS[row['rank']] or row['mhz'] not in core.RATES:
            raise ValueError('Wrong rank/UUID/frequency')
        if abs(row['actual_mhz']-row['mhz']) > 15: raise ValueError('Frequency did not settle')
        if row['sample_id'] in ids: raise ValueError('Duplicate sample')
        ids.add(row['sample_id'])
        if (row.get('same_path_verified') is not True or not row.get('path_evidence_sha256')
                or not row.get('signature') or row.get('power_attribution') != 'repeated_unit_board_measurement'
                or not row.get('power_evidence_sha256') or not row.get('prompt_hash')):
            raise ValueError('Missing same-path/state or measured board power provenance')
        for field in ('latency_s', 'power_w', 'voltage_v', 'window_s', 'sensor_interval_s'):
            positive(row[field], field)
        positive(row['temperature_c'], 'temperature', zero=True)
        if row['window_s'] < 2*row['sensor_interval_s'] or not 0 <= row.get('energy_uncertainty', 1) <= .03:
            raise ValueError('Unit power window is unresolved')
        key = f"{row['rank']}:{row['context']}/{row['unit']}"
        spec = data['workloads'][key]
        if not spec.get('workload_provenance') or not spec.get('data_semantics'):
            raise ValueError('FLOP/byte definitions need provenance')
        if spec.get('s_comp', 0) != 0 or spec.get('s_mem', 0) != 0:
            raise ValueError('Positive effective sparsity needs a separately audited kernel adapter')
        cap = data['capacities'][str(row['rank'])]
        if not cap.get('same_path_provenance'): raise ValueError('Capacity must be independently measured on same path')
        positive(cap['bandwidth_Bps'], 'bandwidth')
        positive(cap['peak_flops'][str(row['mhz'])], 'capacity')
        if row['signature'] != spec['signature']: raise ValueError('Replay signature/state changed')
        groups.setdefault(key, []).append(row)
    for rows in groups.values():
        if {r['mhz'] for r in rows} != set(core.RATES) or len(rows) < 4:
            raise ValueError('Each rank/unit needs both frequencies and independent repetitions')
    return groups


def fit(data, allow_synthetic=False):
    groups = check_data(data, 'train', allow_synthetic)
    model = dict(schema=core.SCHEMA, synthetic=data['synthetic'], identity_fingerprint=data['identity_fingerprint'],
        training_sha256=digest(data), training_prompts=sorted({r['prompt_hash'] for r in data['records']}),
        capacities=data['capacities'], workloads=data['workloads'], operators={}, thermal={},
        scope='GPU_only_same_path_equation1_equation2_not_CPU_EMC',
        units=data['units'], block_selection_authorized=False)
    # Eliminate one alpha per unit, then solve ONLY the two shared thermal
    # coefficients per board. This avoids a dense O(units^2) regression matrix.
    for rank in (0, 1):
        subset = {k: rows for k, rows in groups.items() if k.startswith(f'{rank}:')}
        if not subset: raise ValueError('Both ranks need observations')
        matrix, targets = [], []
        for rows in subset.values():
            a = [r['voltage_v']**2*r['mhz']*1e6 for r in rows]
            x = [[r['temperature_c']*r['voltage_v'], r['voltage_v']] for r in rows]
            y = [r['power_w'] for r in rows]
            norm = sum(v*v for v in a)
            projx = [sum(v*xx[j] for v, xx in zip(a, x))/norm for j in range(2)]
            projy = sum(v*yy for v, yy in zip(a, y))/norm
            matrix.extend([[xx[j]-v*projx[j] for j in range(2)] for v, xx in zip(a, x)])
            targets.extend([yy-v*projy for v, yy in zip(a, y)])
        k1, k2 = least_squares(matrix, targets)
        model['thermal'][str(rank)] = dict(k1=k1, k2=k2,
            temperature_range_c=[min(r['temperature_c'] for rows in subset.values() for r in rows),
                                 max(r['temperature_c'] for rows in subset.values() for r in rows)])
        for key, rows in subset.items():
            spec, cap = data['workloads'][key], data['capacities'][str(rank)]
            a = [r['voltage_v']**2*r['mhz']*1e6 for r in rows]
            residual = [r['power_w']-(k1*r['temperature_c']+k2)*r['voltage_v'] for r in rows]
            alpha = sum(v*y for v, y in zip(a, residual))/sum(v*v for v in a)
            if alpha < 0: raise ValueError('Negative activity factor; insufficient/inconsistent physical evidence')
            overhead = max(0, mean(r['latency_s']-timeline_seconds(spec['work_flop'], spec['data_byte'], 0, 0,
                cap['peak_flops'][str(r['mhz'])], cap['bandwidth_Bps'], 0) for r in rows))
            model['operators'][key] = dict(alpha=alpha, overhead_s=overhead,
                voltage_by_mhz={str(f): mean(r['voltage_v'] for r in rows if r['mhz'] == f) for f in core.RATES})
    return model


def predict_latency(model, rank, context, unit, mhz):
    """Eq.(1) independently of voltage/thermal evidence (no deployment grant)."""
    key = f'{rank}:{context}/{unit}'
    spec, op, cap = model['workloads'][key], model['operators'][key], model['capacities'][str(rank)]
    return timeline_seconds(spec['work_flop'], spec['data_byte'], 0, 0,
        cap['peak_flops'][str(mhz)], cap['bandwidth_Bps'], op['overhead_s'])


def predict_power(model, rank, context, unit, mhz, temperature):
    """Eq.(2) independently of the timeline parameters."""
    key = f'{rank}:{context}/{unit}'
    op = model['operators'][key]
    thermal = model['thermal'][str(rank)]
    lo, hi = thermal['temperature_range_c']
    if not lo <= temperature <= hi: raise ValueError('Temperature outside calibrated range')
    return thermal_power_w(op['alpha'], op['voltage_by_mhz'][str(mhz)], mhz*1e6,
                           thermal['k1'], thermal['k2'], temperature)


def predict(model, rank, context, unit, mhz, temperature):
    return (predict_latency(model, rank, context, unit, mhz),
            predict_power(model, rank, context, unit, mhz, temperature))


def validate(data, model, allow_synthetic=False):
    groups = check_data(data, 'validation', allow_synthetic)
    if (data['identity_fingerprint'] != model['identity_fingerprint']
            or data['workloads'] != model['workloads'] or data['capacities'] != model['capacities']):
        raise ValueError('Validation changed frozen workload/capacity/environment')
    if set(groups) != set(model['operators']): raise ValueError('Missing validation units')
    if set(model['training_prompts']) & {r['prompt_hash'] for r in data['records']}:
        raise ValueError('Training/validation requests overlap')
    scores = {}
    for rank in (0, 1):
        for frequency in core.RATES:
            rows = [r for r in data['records'] if r['rank'] == rank and r['mhz'] == frequency]
            predicted = [predict(model, rank, r['context'], r['unit'], frequency, r['temperature_c']) for r in rows]
            scores[f'{rank}:{frequency}'] = dict(
                latency=errors([r['latency_s'] for r in rows], [p[0] for p in predicted]),
                energy=errors([r['power_w']*r['latency_s'] for r in rows], [p[0]*p[1] for p in predicted]),
                energy_label='repeated_unit_average_power_times_unit_latency_not_request_allocation')
    passed = all(v[k]['wape'] <= .1 and v[k]['p90_relative_error'] <= .2 for v in scores.values() for k in ('latency', 'energy'))
    return dict(schema=core.SCHEMA, passed=passed, synthetic=data['synthetic'] or model['synthetic'],
        identity_fingerprint=model['identity_fingerprint'], model_sha256=digest(model), data_sha256=digest(data),
        scores=scores, scope=model['scope'], independent_validation=True,
        validation_prompts=sorted({r['prompt_hash'] for r in data['records']}))


def plan(model, validation, coverage, switching, temperature):
    if (not validation.get('passed') or validation.get('synthetic') is not False
            or validation['model_sha256'] != digest(model) or model.get('synthetic') is not False):
        raise ValueError('Frozen equation model needs real independent validation')
    fingerprint = model['identity_fingerprint']
    if any(v.get('identity_fingerprint') != fingerprint for v in (validation, coverage, switching)):
        raise ValueError('Evidence belongs to a different process/configuration')
    if not coverage.get('passed') or coverage.get('units') != core.UNITS:
        raise ValueError('Actual two-rank module boundary evidence missing')
    if switching.get('conservative_bound_verified') is not True or switching.get('restoration', {}).get('verified') is not True:
        raise ValueError('Measured switching upper bound and physical restoration required')
    switch_s = positive(switching['conservative_switch_s'], 'measured switch bound')
    contexts = ['prefill']+[f'decode.{i}' for i in range(63)]
    result = dict(schema=core.SCHEMA, synthetic=False, identity_fingerprint=fingerprint,
        mode='serial', cpu=None, emc=None, model_validation=validation, coverage=coverage,
        model_sha256=digest(model), switching_sha256=digest(switching), contexts={},
        partition_decisions={},eq3_by_context={},
        temperature_c=temperature, N=5, similarity=.05, latency_budget=1.05,
        scope='GPU_pair_module_chain_projection_not_full_paper_mechanisms',
        temperature_ranges={r: v['temperature_range_c'] for r, v in model['thermal'].items()},
        choices='minimum_predicted_board_energy_under_per_unit_high_frequency_5pct_budget',
        deadline_note='per-unit budget does not certify HTTP end-to-end 5pct; must compare to AUTO',
        source='Eq1_Eq2_Algorithm1', switch_s=switch_s)
    result['excluded_prompt_hashes'] = sorted(set(model['training_prompts']) | set(validation['validation_prompts']))
    result['lookahead_bound_verified'] = switching.get('fine_boundary_precision_verified') is True
    for context in contexts:
        table = {}
        for unit in core.UNITS:
            table[unit] = {}
            for mhz in core.RATES:
                values = [predict(model, rank, context, unit, mhz, temperature[str(rank)]) for rank in (0, 1)]
                seconds = max(v[0] for v in values)
                # Explicit paired-window approximation, not a measured unit label.
                energy = seconds*sum(v[1] for v in values)
                table[unit][mhz] = (seconds, energy)
        choices = {}
        for unit, candidates in table.items():
            feasible = [f for f in core.RATES if candidates[f][0] <= candidates[2527][0]*1.05]
            choices[unit] = min(feasible, key=lambda f: (candidates[f][1], candidates[f][0], f))
        blocks, decisions = greedy_partition(OrderedDAG.chain(core.UNITS),
            lambda u: (choices[u], choices[u]),
            lambda units, freq: sum(table[u][freq.values[0]][0] for u in units),
            lambda a, b: all(abs(x-y) <= .05*max(x,y) for x,y in zip(a.values,b.values)),
            switch_s, 5, frequency_factory=PairFrequency.of)
        result['contexts'][context] = [dict(units=b['nodes'], mhz=b['triplet'][0],
            estimated_s=b['estimated_s'], amortized_strict_gt=b['amortized_strict_gt']) for b in blocks]
        result['partition_decisions'][context] = decisions
        result['eq3_by_context'][context] = dict(all_blocks_strict_gt=all(b['amortized_strict_gt'] for b in blocks),
            tail_short_or_equal=not blocks[-1]['amortized_strict_gt'],
            note='Literal Algorithm1 may retain an equality boundary or short tail; flags do not certify switching amortization')
    core.validate_plan(result, fingerprint)
    return result
