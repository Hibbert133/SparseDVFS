"""Small, dependency-free Eq. (1)/(2) calibration with explicit evidence gates.

Raw observations are inputs, not inferred from activation zeros or request-average
power. Thermal/activity regression uses shared k1/k2 and one alpha per unit.
"""
import math
import statistics
from .paper_core import timeline_seconds, thermal_power_w


def least_squares(matrix, target):
    """Column-scaled modified Gram-Schmidt QR with reorthogonalization."""
    if not matrix or len(matrix) != len(target):
        raise ValueError('Empty/inconsistent regression data')
    n = len(matrix[0])
    if len(matrix) < n or any(len(row) != n for row in matrix):
        raise ValueError('Rank deficient: fewer independent observations than parameters')
    if any(not math.isfinite(v) for row in matrix for v in row) or any(not math.isfinite(v) for v in target):
        raise ValueError('Nonfinite regression data')
    scales = [math.sqrt(sum(row[j]**2 for row in matrix)) for j in range(n)]
    if any(scale == 0 for scale in scales):
        raise ValueError('Rank deficient zero column')
    q, r = [], [[0.0]*n for _ in range(n)]
    for j in range(n):
        col = [row[j]/scales[j] for row in matrix]
        for _ in range(2):
            for i, qi in enumerate(q):
                projection = sum(a*b for a, b in zip(qi, col))
                r[i][j] += projection
                col = [a-projection*b for a, b in zip(col, qi)]
        norm = math.sqrt(sum(a*a for a in col))
        if norm < 1e-10:
            raise ValueError('Rank deficient/ill-conditioned: voltage/frequency/temperature variation insufficient')
        r[j][j] = norm
        q.append([a/norm for a in col])
    rhs = [sum(a*b for a, b in zip(col, target)) for col in q]
    x = [0.0]*n
    for i in reversed(range(n)):
        x[i] = (rhs[i]-sum(r[i][j]*x[j] for j in range(i+1, n)))/r[i][i]
    return [v/s for v, s in zip(x, scales)]


def required_units(profile):
    return [r['phase']+'/'+r['name'] for r in profile['candidates'][0]['operators']]


def validate_observation_units(observations):
    if observations.get('units') != dict(power='W', voltage='V', frequency='Hz', temperature='C', latency='s'):
        raise ValueError('Observations require explicit W/V/Hz/C/s units; no guessed conversions')


def fit_calibration(profile, observations, workloads, max_relative_error=None):
    validate_observation_units(observations)
    if observations.get('identity') != profile['identity']:
        raise ValueError('Calibration/profile identities differ')
    units = required_units(profile)
    pending = []
    specs = workloads.get('operators', {})
    for unit in units:
        spec = specs.get(unit, {})
        for name in ('work_flop', 'data_byte'):
            value = spec.get(name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                pending.append(unit+': unknown '+name)
        if not spec.get('workload_provenance') or spec.get('data_semantics') not in (
                'measured_dram_traffic', 'explicit_paper_effective_traffic_estimate'):
            pending.append(unit+': require FLOP/traffic provenance and explicit byte semantics')
        if spec.get('s_comp', 0) != 0 or spec.get('s_mem', 0) != 0:
            raise ValueError('Dense BF16 target requires S_comp=S_mem=0')
    records = observations.get('unit_records', [])
    grouped = {unit: [] for unit in units}
    for row in records:
        unit = row['unit']
        if unit not in grouped:
            raise ValueError('Observation names an unknown unit')
        if row.get('power_attribution') != 'isolated_unit_measured' or not row.get('power_evidence'):
            raise ValueError('Request-average power cannot fit per-unit alpha; supply isolated measured evidence')
        if row.get('power_scope') != observations.get('power_scope'):
            raise ValueError('Power rail/scope mismatch')
        duration, interval = row.get('window_duration_s', 0), row.get('sensor_interval_s', 0)
        if (not isinstance(duration,(int,float)) or not isinstance(interval,(int,float)) or
                not math.isfinite(duration) or not math.isfinite(interval) or interval <= 0 or duration < 2*interval):
            raise ValueError('Isolated power window must cover at least two documented sensor intervals')
        for field in ('power_w', 'voltage_v', 'effective_frequency_hz', 'latency_s'):
            if not math.isfinite(row[field]) or row[field] <= 0:
                raise ValueError('Invalid measured '+field)
        if not math.isfinite(row['temperature_c']):
            raise ValueError('Invalid measured temperature')
        grouped[unit].append(row)
    if not observations.get('scalar_vf_scope') or not observations.get('temperature_sensor'):
        pending.append('Specify scalar V/f projection, power scope and temperature sensor')
    if not observations.get('power_scope'):
        pending.append('Missing power rail scope')
    if any(len(rows) < 4 for rows in grouped.values()):
        pending.append('At least four independently measured observations per unit are required (one held out)')
    opps, capacities, bandwidths = {}, observations.get('capacities', {}), observations.get('bandwidths', {})
    for candidate in profile['candidates']:
        c, g, e = candidate['triplet']
        key = ','.join(map(str, (c, g, e)))
        raw = observations.get('opps', {}).get(key, {})
        peak, bw = capacities.get(f'{c},{g}'), bandwidths.get(str(e))
        if (not peak or not bw or not raw.get('voltage_v') or not raw.get('effective_frequency_hz') or
                not raw.get('capacity_provenance')):
            pending.append(key+': missing capacity/bandwidth/voltage evidence')
        else:
            opps[key] = dict(peak_flops=peak, bandwidth_Bps=bw, voltage_v=raw['voltage_v'],
                             effective_frequency_hz=raw['effective_frequency_hz'])
    base = dict(schema=4, synthetic=bool(observations.get('synthetic') or profile.get('synthetic')),
                provenance=observations.get('provenance', 'missing'),
                scalar_vf_scope=observations.get('scalar_vf_scope'),
                temperature_sensor=observations.get('temperature_sensor'),
                power_scope=observations.get('power_scope'), operators={}, opps=opps)
    if pending:
        base['v5_validation'] = dict(deployable=False, status='pending', reasons=sorted(set(pending)))
        return base
    train, heldout = [], []
    for rows in grouped.values():
        for i, row in enumerate(rows):
            (heldout if i % 4 == 3 else train).append(row)
    matrix, target = [], []
    for row in train:
        v, f, t = row['voltage_v'], row['effective_frequency_hz'], row['temperature_c']
        matrix.append([v*v*f if unit == row['unit'] else 0.0 for unit in units] + [t*v, v])
        target.append(row['power_w'])
    try:
        coeff = least_squares(matrix, target)
    except ValueError as exc:
        base['v5_validation'] = dict(deployable=False, status='pending', reasons=[str(exc)])
        return base
    k1, k2 = coeff[-2:]
    if any(alpha < 0 for alpha in coeff[:-2]):
        raise ValueError('Fitted activity factor is negative; the selected model/scope is unsuitable')
    base['device_thermal'] = dict(k1=k1, k2=k2)
    temperatures = [r['temperature_c'] for r in records]
    base['temperature_c'] = statistics.mean(temperatures)
    base['temperature_range_c'] = [min(temperatures), max(temperatures)]
    for index, unit in enumerate(units):
        spec = specs[unit]
        residuals = []
        for row in train:
            if row['unit'] != unit:
                continue
            opp = opps[','.join(map(str, row['triplet']))]
            roofline = timeline_seconds(spec['work_flop'], spec['data_byte'], 0, 0,
                                       opp['peak_flops'], opp['bandwidth_Bps'], 0)
            residuals.append(row['latency_s']-roofline)
        overhead = max(0.0, statistics.median(residuals))
        base['operators'][unit] = dict(spec, overhead_s=overhead, activity_factor=coeff[index],
                                       activity_s_comp=0.0, s_comp=0.0, s_mem=0.0)
    validation = validate_fit(base, observations, heldout, max_relative_error)
    validation.update(train_records=len(train), heldout_records=len(heldout),
                      rank=len(coeff), qr_relative_rank_tolerance=1e-10,
                      synthetic=base['synthetic'])
    validation['deployable'] = validation['deployable'] and not base['synthetic']
    base['v5_validation'] = validation
    return base


def validate_fit(calibration, observations, records=None, max_relative_error=None):
    validate_observation_units(observations)
    rows = records if records is not None else observations.get('unit_records', [])
    errors, reasons = [], []
    if not rows or not calibration.get('operators') or not calibration.get('device_thermal'):
        return dict(deployable=False, status='pending', reasons=['Missing fitted coefficients or validation records'])
    thermal = calibration['device_thermal']
    for row in rows:
        spec = calibration['operators'][row['unit']]
        opp = calibration['opps'][','.join(map(str, row['triplet']))]
        t = timeline_seconds(spec['work_flop'], spec['data_byte'], 0, 0,
                             opp['peak_flops'], opp['bandwidth_Bps'], spec['overhead_s'])
        # Validate the SAME fixed OPP projection used by predicted_candidates,
        # not a more informative observed V/f unavailable to runtime planning.
        p = thermal_power_w(spec['activity_factor'], opp['voltage_v'], opp['effective_frequency_hz'],
                            thermal['k1'], thermal['k2'], row['temperature_c'])
        errors.append(dict(unit=row['unit'], latency_relative_error=abs(t-row['latency_s'])/row['latency_s'],
                           power_relative_error=abs(p-row['power_w'])/row['power_w']))
    worst = max(max(e['latency_relative_error'], e['power_relative_error']) for e in errors)
    if max_relative_error is None:
        reasons.append('Supply an explicit --max-relative-error; the paper specifies no acceptance threshold')
    elif not math.isfinite(max_relative_error) or max_relative_error < 0:
        raise ValueError('Invalid validation error threshold')
    elif worst > max_relative_error:
        reasons.append('Held-out model error exceeds the explicit acceptance threshold')
    if observations.get('synthetic') or calibration.get('synthetic'):
        reasons.append('Synthetic calibration is never deployable')
    return dict(deployable=not reasons, status='validated' if not reasons else 'pending',
                max_relative_error=worst, acceptance_relative_error=max_relative_error,
                records=errors, reasons=reasons)
