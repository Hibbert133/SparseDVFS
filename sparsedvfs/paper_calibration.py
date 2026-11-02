"""Schema-4 calibration for the written argument dependencies of Eq. (1)/(2).

This validates a supplied calibration; it does not estimate missing coefficients
or establish that a claimed hardware sparsity mechanism exists.
"""
from math import isclose
from .paper_core import Frequency, finite, timeline_seconds, thermal_power_w


def text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} requires a recorded source/description')
    return value


def validate(profile, calibration, allow_synthetic=False):
    if calibration.get('schema') != 4:
        raise ValueError('Paper-equation calibration requires schema 4; migrate legacy per-OPP alpha/k1/k2')
    if calibration.get('synthetic') and not allow_synthetic:
        raise ValueError('Synthetic calibration cannot produce a hardware deployment plan')
    text(calibration.get('provenance'), 'calibration provenance')
    text(calibration.get('scalar_vf_scope'), 'scalar V/f mapping')
    temperature = finite(calibration.get('temperature_c'), 'calibrated temperature')
    limits = calibration.get('temperature_range_c', [])
    if len(limits) != 2:
        raise ValueError('temperature_range_c needs two measured bounds')
    lo, hi = (finite(v, 'temperature bound') for v in limits)
    if not lo < hi or not lo <= temperature <= hi:
        raise ValueError('Temperature is outside the calibrated range')
    text(calibration.get('temperature_sensor'), 'temperature sensor')
    thermal = calibration.get('device_thermal', {})
    finite(thermal.get('k1'), 'shared device k1')
    finite(thermal.get('k2'), 'shared device k2')
    capacities, bandwidths, vectors, keys = {}, {}, set(), None
    for candidate in profile['candidates']:
        vector = Frequency.of(candidate['triplet'])
        if vector in vectors:
            raise ValueError('Duplicate candidate frequency vector')
        vectors.add(vector)
        key = ','.join(map(str, vector.values))
        try:
            opp = calibration['opps'][key]
        except KeyError as exc:
            raise ValueError(f'Missing calibrated OPP {key}') from exc
        if 'k1' in opp or 'k2' in opp:
            raise ValueError('k1/k2 must be shared device coefficients, not per-OPP free values')
        peak, bandwidth = finite(opp.get('peak_flops'), 'P_peak', positive=True), finite(opp.get('bandwidth_Bps'), 'B_mem', positive=True)
        finite(opp.get('voltage_v'), 'V', positive=True)
        finite(opp.get('effective_frequency_hz'), 'effective scalar f', positive=True)
        pair = vector.values[:2]
        if pair in capacities and not isclose(peak, capacities[pair], rel_tol=1e-12):
            raise ValueError('Eq. (1): P_peak must depend only on CPU/GPU, not EMC')
        if vector.emc_hz in bandwidths and not isclose(bandwidth, bandwidths[vector.emc_hz], rel_tol=1e-12):
            raise ValueError('Eq. (1): B_mem must depend only on EMC, not CPU/GPU')
        capacities[pair], bandwidths[vector.emc_hz] = peak, bandwidth
        rows = [(r['phase'], r['name']) for r in candidate['operators']]
        if len(rows) != len(set(rows)) or not rows:
            raise ValueError('Duplicate/empty operator coverage')
        if keys is None:
            keys = set(rows)
        elif set(rows) != keys:
            raise ValueError('Incomplete operator coverage across OPPs')
    if not vectors or Frequency.of(profile['reference']) not in vectors:
        raise ValueError('Missing reference OPP')
    for phase, name in keys:
        key = phase + '/' + name
        try:
            spec = calibration['operators'][key]
        except KeyError as exc:
            raise ValueError(f'Missing operator calibration {key}') from exc
        if 'alpha_by_opp' in spec:
            raise ValueError('Eq. (2): use one activity_factor at the recorded S_comp, not alpha_by_opp')
        alpha = finite(spec.get('activity_factor'), 'alpha(S_comp)', 0)
        sc, sm = finite(spec.get('s_comp'), 'S_comp', 0), finite(spec.get('s_mem'), 'S_mem', 0)
        if sc > 1 or sm > 1 or not isclose(finite(spec.get('activity_s_comp'), 'activity S_comp'), sc, abs_tol=1e-12):
            raise ValueError('Activity factor must be calibrated at this effective S_comp within [0,1]')
        text(spec.get('workload_provenance'), 'operator FLOP/traffic provenance')
        if (sc > 0 or sm > 0) and (not spec.get('effective_sparsity_verified') or not spec.get('effective_sparsity_evidence')):
            raise ValueError('Positive effective sparsity requires verified skip/compression evidence; raw zeros are insufficient')
        # Validate workload and total power at every supplied point.
        for opp in calibration['opps'].values():
            timeline_seconds(spec.get('work_flop'), spec.get('data_byte'), sc, sm,
                             opp['peak_flops'], opp['bandwidth_Bps'], spec.get('overhead_s'))
            thermal_power_w(alpha, opp['voltage_v'], opp['effective_frequency_hz'],
                            thermal['k1'], thermal['k2'], temperature)
    return calibration


def predicted_candidates(profile, calibration, allow_synthetic=False):
    validate(profile, calibration, allow_synthetic)
    temperature, thermal = calibration['temperature_c'], calibration['device_thermal']
    candidates = []
    for candidate in profile['candidates']:
        opp = calibration['opps'][','.join(map(str, candidate['triplet']))]
        operators = []
        for record in candidate['operators']:
            spec = calibration['operators'][record['phase'] + '/' + record['name']]
            seconds = timeline_seconds(spec['work_flop'], spec['data_byte'], spec['s_comp'], spec['s_mem'],
                                       opp['peak_flops'], opp['bandwidth_Bps'], spec['overhead_s'])
            power = thermal_power_w(spec['activity_factor'], opp['voltage_v'], opp['effective_frequency_hz'],
                                    thermal['k1'], thermal['k2'], temperature)
            operators.append(dict(record, latency_ms=seconds * 1000, power_w=power,
                s_comp=spec['s_comp'], s_mem=spec['s_mem'],
                compute_ms=spec['work_flop'] * (1-spec['s_comp']) / opp['peak_flops'] * 1000,
                memory_ms=spec['data_byte'] * (1-spec['s_mem']) / opp['bandwidth_Bps'] * 1000,
                energy_proxy_j=seconds*power, activity_factor=spec['activity_factor'],
                power_attribution='Eq2_scalar_prediction_not_measurement'))
        candidates.append(dict(candidate, operators=operators))
    return candidates


def coordination_vectors(indexed, operator_key, resource_class):
    """An explicit quantitative interpretation of section 4.3.2.

    Max available EMC for effective-sparse memory work; minimum sufficient EMC
    per CPU/GPU for compute-bound work. The paper leaves dense-memory work
    unspecified, so the returned annotation exposes the proposed fallback.
    """
    if resource_class == 'effective_sparse_memory_bound':
        highest = max(freq[2] for freq in indexed)
        return {freq for freq in indexed if freq[2] == highest}, 'max_profiled_emc_for_sparse_memory'
    if resource_class == 'compute_bound':
        sufficient = {freq for freq, rows in indexed.items()
                      if rows[operator_key]['memory_ms'] <= rows[operator_key]['compute_ms']}
        minima = {}
        for freq in sufficient:
            minima[freq[:2]] = min(minima.get(freq[:2], freq[2]), freq[2])
        return {freq for freq in sufficient if freq[2] == minima[freq[:2]]}, 'minimum_emc_with_memory_time_not_above_compute_time'
    return set(indexed), 'PROPOSED_dense_memory_fallback_energy_under_user_budget'
