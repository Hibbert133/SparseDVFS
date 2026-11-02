"""Paper equations (1)/(2), plus an explicitly separate empirical demo selector."""
import math
from .paper_core import timeline_seconds as _eq1, thermal_power_w as _eq2
from .paper_calibration import predicted_candidates, coordination_vectors


def timeline_seconds(work_flop, data_byte, s_comp, s_mem, peak_flops, bandwidth_Bps, overhead_s):
    """arXiv:2603.21908v1 Eq. (1). All inputs require target calibration."""
    return _eq1(work_flop, data_byte, s_comp, s_mem, peak_flops, bandwidth_Bps, overhead_s)


def thermal_power_w(alpha, voltage_v, frequency_hz, k1, k2, temperature_c):
    """Paper Eq. (2), the scalar effective-domain form actually specified there."""
    return _eq2(alpha, voltage_v, frequency_hz, k1, k2, temperature_c)


def predicted_table(profile, calibration):
    """Require complete user calibration; never fill missing coefficients with guesses."""
    if calibration.get('schema') == 4:
        return predicted_candidates(profile, calibration, allow_synthetic=bool(profile.get('synthetic')))
    if calibration.get('synthetic', False):
        raise ValueError('Synthetic calibration cannot produce a hardware deployment plan')
    table = []
    for candidate in profile['candidates']:
        freq = candidate['triplet']
        key = ','.join(map(str, freq))
        opp = calibration['opps'][key]
        operators = []
        for record in candidate['operators']:
            spec = calibration['operators'][record['phase'] + '/' + record['name']]
            if (spec['s_comp'] > 0 or spec['s_mem'] > 0) and not spec.get('effective_sparsity_verified', False):
                raise ValueError('Positive effective sparsity requires verified skip/compression support; raw zeros are insufficient')
            # alpha(S_comp) is calibrated per operator/OPP, not inferred from raw zeros.
            power = thermal_power_w(spec['alpha_by_opp'][key], opp['voltage_v'], opp['effective_frequency_hz'],
                                    opp['k1'], opp['k2'], calibration['temperature_c'])
            latency = timeline_seconds(spec['work_flop'], spec['data_byte'], spec['s_comp'], spec['s_mem'],
                                       opp['peak_flops'], opp['bandwidth_Bps'], spec['overhead_s']) * 1000
            operators.append(dict(record, latency_ms=latency, power_w=power,
                                  s_comp=spec['s_comp'], s_mem=spec['s_mem'],
                                  compute_ms=spec['work_flop'] * (1 - spec['s_comp']) / opp['peak_flops'] * 1000,
                                  memory_ms=spec['data_byte'] * (1 - spec['s_mem']) / opp['bandwidth_Bps'] * 1000,
                                  energy_proxy_j=power * latency / 1000))
        table.append(dict(candidate, operators=operators))
    return table


def select_operators(profile, max_slowdown=0.05, selector='measured', calibration=None, memory_policy='calibrated'):
    if not math.isfinite(max_slowdown) or max_slowdown < 0:
        raise ValueError('max_slowdown must be finite and nonnegative')
    if not profile['candidates']:
        raise ValueError('empty profile')
    if selector == 'paper':
        if calibration is None:
            raise ValueError('--selector paper requires complete target --calibration')
        candidates = predicted_table(profile, calibration)
    elif selector == 'measured':
        candidates = profile['candidates']
    else:
        raise ValueError('unknown selector')
    if memory_policy not in ('calibrated', 'paper'):
        raise ValueError('unknown memory policy')
    if memory_policy == 'paper' and selector != 'paper':
        raise ValueError('paper memory coordination requires calibrated compute and bandwidth paths')
    reference = tuple(profile['reference'])
    if len({tuple(c['triplet']) for c in candidates}) != len(candidates):
        raise ValueError('duplicate profile frequency vector')
    for candidate in candidates:
        keys_in_candidate = [(r['phase'], r['name']) for r in candidate['operators']]
        if not keys_in_candidate or len(set(keys_in_candidate)) != len(keys_in_candidate):
            raise ValueError('empty/duplicate operator rows')
    indexed = {tuple(c['triplet']): {(r['phase'], r['name']): r for r in c['operators']} for c in candidates}
    if reference not in indexed:
        raise ValueError('high-frequency reference missing')
    keys = list(indexed[reference])
    for freq, rows in indexed.items():
        if set(rows) != set(keys):
            raise ValueError(f'incomplete operator coverage at {freq}')
    chosen = []
    for phase, name in keys:
        ref_record = indexed[reference][(phase, name)]
        baseline = ref_record['latency_ms']
        resource_class = 'unclassified_measured'
        if selector == 'paper':
            resource_class = ('effective_sparse_memory_bound' if ref_record['s_comp'] > 0 and
                              ref_record['memory_ms'] >= ref_record['compute_ms'] else
                              'compute_bound' if ref_record['compute_ms'] >= ref_record['memory_ms'] else
                              'dense_memory_bound')
        options = []
        allowed, coordination = None, 'not_applied'
        strict_coordination = memory_policy == 'paper' and calibration.get('schema') == 4
        if strict_coordination:
            allowed, coordination = coordination_vectors(indexed, (phase, name), resource_class)
        for freq, rows in indexed.items():
            record = rows[(phase, name)]
            power = record.get('power_w')
            latency = record['latency_ms']
            if power is None or not math.isfinite(power) or power <= 0:
                raise ValueError('No valid measured power. Select a real total power sensor; no 0.5 W fallback exists.')
            if not math.isfinite(latency) or latency <= 0:
                raise ValueError('invalid measured/predicted latency')
            if allowed is not None and freq not in allowed:
                continue
            if memory_policy == 'paper' and not strict_coordination:
                if resource_class == 'effective_sparse_memory_bound' and freq[2] != reference[2]:
                    continue  # Section 4.3.2: high EMC for an effectively sparse, memory-bound block.
                if resource_class == 'compute_bound' and record['memory_ms'] > record['compute_ms'] * (1 + max_slowdown):
                    continue  # Select enough bandwidth to support the chosen compute throughput.
            if latency <= baseline * (1 + max_slowdown):
                options.append((power * latency / 1000, latency, freq, record))
        if not options:
            raise ValueError('no feasible OPP')
        energy, latency, freq, record = min(options, key=lambda r: (r[0], r[1], r[2]))
        chosen.append({'phase': phase, 'name': name, 'triplet': list(freq), 'latency_ms': latency,
                       'energy_proxy_j': energy, 'sparsity_exact_zero': record.get('sparsity_exact_zero'),
                       'resource_class': resource_class,
                       'coordination_rule': coordination,
                       's_comp': record.get('s_comp', 0.0), 's_mem': record.get('s_mem', 0.0)})
    def duration(names, phase, freq):
        if tuple(freq) not in indexed:
            raise ValueError(f'Conservatively merged OPP {freq} missing from profile; use a complete Cartesian grid')
        rows = indexed[tuple(freq)]
        return sum(rows[(phase, name)]['latency_ms'] for name in names)
    return chosen, duration
