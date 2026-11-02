"""Target-observation extraction and offline fitting; never guesses per-op power."""
import statistics
from pathlib import Path
from sparsedvfs.hardware import load_json, write_json
from sparsedvfs.calibration_fit import fit_calibration, validate_fit
from .contracts import digest_json


def collect_profile(profile, hardware):
    if profile.get('synthetic') or not profile.get('dvfs_applied'):
        raise ValueError('Collect requires a real applied target profile')
    scope = hardware.get('calibration_scope', {})
    reference = profile['reference']
    capacities, bandwidths, opps, request_records = {}, {}, {}, []
    for candidate in profile['candidates']:
        c, g, e = candidate['triplet']
        micro = candidate.get('microbench', {})
        if e == reference[2] and micro.get('gemm_flops'):
            capacities[f'{c},{g}'] = micro['gemm_flops']
        if [c,g] == reference[:2] and micro.get('streaming_logical_Bps'):
            bandwidths[str(e)] = micro['streaming_logical_Bps']
        reading = candidate.get('readback', {}).get('observation', {})
        voltage = [r['value_v'] for r in reading.get('sensors', [])
                   if r['name'] == scope.get('voltage_sensor') and r.get('value_v') is not None]
        key = ','.join(map(str, candidate['triplet']))
        opps[key] = dict(voltage_v=voltage[0] if len(voltage) == 1 else None,
                         effective_frequency_hz=g if scope.get('frequency_domain') == 'gpc' else None,
                         capacity_provenance=micro.get('provenance'), observation=reading)
        request_records.append(dict(triplet=candidate['triplet'], measurement=candidate['measurement'],
                                    observation=reading, operators=candidate['operators'],
                                    power_attribution='whole_request_average_proxy_not_unit_power'))
    return dict(schema=5, identity=profile['identity'], profile_sha256=digest_json(profile),
                synthetic=False, provenance='Thor worker profile plus idle microbenchmark/readback',
                units=dict(power='W', voltage='V', frequency='Hz', temperature='C', latency='s'),
                scalar_vf_scope=scope.get('scalar_vf_scope'), power_scope=scope.get('power_scope'),
                temperature_sensor=scope.get('temperature_sensor'), opps=opps,
                capacities=capacities, bandwidths=bandwidths,
                request_records=request_records, unit_records=[],
                status='pending_isolated_unit_observations_and_workload_descriptors',
                missing=['Per-unit alpha requires isolated same-rail power with sufficient sensor coverage',
                         'Provide measured/provenanced work_flop and effective/DRAM data_byte per unit',
                         'Thermal k1/k2 require independent temperature variation; no fan/power-mode changes are automatic'],
                capacity_scope='effective GEMM/copy capacity estimates, not measured Qwen traffic')


def command(args, root):
    root = Path(root)
    out = Path(args.output) if args.output else root/'calibration.json'
    if args.stage == 'collect':
        if not args.apply:
            raise ValueError('Calibration collect requires explicit --apply on Thor')
        profile = load_json(root/'profile.json')
        observed = collect_profile(profile, load_json(args.results/'hardware.json'))
        path = Path(args.output) if args.output else root/'observations.json'
        write_json(path, observed)
        from .workloads import template
        workload_path = root/'workloads.template.json'
        if not workload_path.exists():
            write_json(workload_path,template(profile))
        print(f'Wrote {path}; isolated unit power/workload inputs remain explicitly pending')
        return
    if not args.observations:
        raise ValueError('Supply --observations from real collection or a labeled synthetic test')
    observations = load_json(args.observations)
    if args.stage == 'fit':
        if not args.workloads:
            raise ValueError('Fit requires --workloads with explicit FLOP/traffic provenance')
        profile = load_json(root/'profile.json')
        if observations.get('profile_sha256') != digest_json(profile):
            raise ValueError('Observation/profile checksum mismatch')
        value = fit_calibration(profile, observations, load_json(args.workloads), args.max_relative_error)
        value['profile_sha256'] = digest_json(profile)
        value['profile_identity'] = profile['identity']
        value['observations_sha256'] = digest_json(observations)
        write_json(out, value)
        print(value['v5_validation'])
    elif args.stage == 'validate':
        if not args.calibration:
            raise ValueError('Validate requires --calibration')
        value = load_json(args.calibration)
        if value.get('observations_sha256') != digest_json(observations):
            raise ValueError('Calibration/observation checksum mismatch')
        # Full supplied data is a diagnostic recheck; do not replace the stored
        # held-out acceptance record or upgrade a previously pending calibration.
        status = validate_fit(value, observations, max_relative_error=args.max_relative_error)
        status['deployable'] = status['deployable'] and value.get('v5_validation', {}).get('deployable', False)
        write_json(root/'calibration_validation.json', status)
        print(status)
    else:
        raise ValueError('Use --stage collect, fit or validate')
