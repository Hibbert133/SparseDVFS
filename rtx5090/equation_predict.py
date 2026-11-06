"""CPU-only Eq.(1)/(2) evaluator; never a calibration or DVFS certificate.

prepare imports existing workload evidence, predict evaluates supplied parameters,
demo runs explicitly illustrative numbers. Missing inputs remain null, independently
for each equation. No CUDA/NVML imports, requests, clock writes or parameter fitting.
"""
import argparse
import json
from pathlib import Path

from sparsedvfs.paper_core import finite, timeline_seconds, thermal_power_w
from .observe_core import GPU_UUIDS, digest
from .service_core import read, write_new

SCHEMA = 'sparsedvfs-offline-equations-v1'
UNITS = dict(work='FLOP', data='byte', capacity='FLOP/s', bandwidth='byte/s',
             time='s', frequency='Hz', voltage='V', temperature='C', power='W')
EQ1 = ('data_byte', 'peak_flops', 'bandwidth_Bps', 'overhead_s')
EQ2 = ('alpha', 'voltage_v', 'frequency_hz', 'k1', 'k2', 'temperature_c')


def make_template(workloads):
    if workloads.get('status') != 'workload_accounting_only':
        raise ValueError('Expected matrix_workloads workload_accounting_only input')
    rows = workloads['workloads']
    expected = {(r, p, m) for r in (0, 1) for p in ('prefill', 'decode')
                for m in ('gate_up_proj', 'down_proj')}
    keys = [(r['rank'], r['phase'], r['module']) for r in rows]
    if len(keys) != 8 or set(keys) != expected:
        raise ValueError('Require eight unique dual-rank MLP cases')
    cases = []
    for row in rows:
        if row['uuid'] != GPU_UUIDS[row['rank']]:
            raise ValueError('Rank/UUID mismatch')
        if not row.get('signature_id'):
            raise ValueError('Missing operator signature')
        m, n, k = (finite(row[x], x, positive=True) for x in ('M', 'N', 'K'))
        if row['work_flop'] != 2*m*n*k:
            raise ValueError('Workload does not match dense 2MNK convention')
        cases.append(dict(case_id=f"{row['rank']}:{row['phase']}/{row['module']}",
            workload=row,
            eq1={**dict.fromkeys(EQ1), 'data_semantics': None,
                 'capacity_condition': None, 'parameter_source': None},
            eq2={**dict.fromkeys(EQ2), 'power_scope': None,
                 'parameter_source': None}))
    return dict(schema=SCHEMA, units=UNITS, illustrative=False,
        identity_fingerprint=workloads['identity_fingerprint'],
        workloads_sha256=digest(workloads), cases=cases)


def evaluate(parameters, workloads=None, allow_illustrative=False):
    if parameters.get('schema') != SCHEMA or parameters.get('units') != UNITS:
        raise ValueError('Wrong equation schema or physical units')
    if type(parameters.get('illustrative')) is not bool:
        raise ValueError('Explicit illustrative boolean required')
    illustrative = parameters['illustrative']
    if illustrative and not allow_illustrative:
        raise ValueError('Illustrative inputs require --allow-illustrative')
    if not illustrative:
        if workloads is None:
            raise ValueError('Real workload inputs require --workloads binding')
        template = make_template(workloads)
        for field in ('identity_fingerprint', 'workloads_sha256'):
            if parameters[field] != template[field]:
                raise ValueError('Workload identity/hash changed')
        original = {c['case_id']: c['workload'] for c in template['cases']}
        if len(parameters['cases']) != len(original):
            raise ValueError('Missing/duplicate cases')
        if {c['case_id']: c['workload'] for c in parameters['cases']} != original:
            raise ValueError('Workload case changed')
    if not parameters.get('cases'):
        raise ValueError('No equation cases')
    results = []
    for case in parameters['cases']:
        work, eq1, eq2 = case['workload'], case['eq1'], case['eq2']
        if work.get('effective_s_comp', 0) != 0 or work.get('effective_s_mem', 0) != 0:
            raise ValueError('Dense adapter requires zero effective sparsity')
        missing1 = [k for k in (*EQ1, 'data_semantics', 'capacity_condition', 'parameter_source')
                    if eq1.get(k) is None or eq1.get(k) == '']
        missing2 = [k for k in (*EQ2, 'power_scope', 'parameter_source')
                    if eq2.get(k) is None or eq2.get(k) == '']
        # Bad supplied values must fail even if another parameter is missing.
        for key in EQ1:
            if eq1.get(key) is not None:
                finite(eq1[key], key, minimum=0, positive=key in ('peak_flops', 'bandwidth_Bps'))
        for key in EQ2:
            if eq2.get(key) is not None:
                finite(eq2[key], key, minimum=0 if key == 'alpha' else None,
                       positive=key in ('voltage_v', 'frequency_hz'))
        latency = power = dynamic = leakage = compute = memory = None
        if not missing1:
            latency = timeline_seconds(work['work_flop'], eq1['data_byte'], 0, 0,
                eq1['peak_flops'], eq1['bandwidth_Bps'], eq1['overhead_s'])
            compute = work['work_flop']/eq1['peak_flops']
            memory = eq1['data_byte']/eq1['bandwidth_Bps']
        if not missing2:
            dynamic = eq2['alpha']*eq2['voltage_v']**2*eq2['frequency_hz']
            leakage = (eq2['k1']*eq2['temperature_c']+eq2['k2'])*eq2['voltage_v']
            if leakage < 0:
                raise ValueError('Negative leakage at requested temperature; invalid physical parameters')
            power = thermal_power_w(*(eq2[k] for k in EQ2))
        results.append(dict(case_id=case['case_id'], rank=work['rank'], uuid=work['uuid'],
            missing_eq1=missing1, missing_eq2=missing2, compute_s=compute, memory_s=memory,
            latency_s=latency, dynamic_power_w=dynamic, leakage_power_w=leakage, power_w=power,
            model_energy_j=None if latency is None or power is None else finite(latency*power, 'energy', 0),
            bottleneck=None if compute is None else ('compute' if compute > memory else
                'memory' if memory > compute else 'balanced')))
    complete = all(not r['missing_eq1'] and not r['missing_eq2'] for r in results)
    return dict(schema=SCHEMA, status=('illustrative_predictions' if illustrative else
                'conditional_predictions_unvalidated') if complete else 'parameters_missing',
        illustrative=illustrative, clock_writes=False, hardware_validated=False,
        parameters_validated=False, block_selection_authorized=False,
        identity_fingerprint=parameters.get('identity_fingerprint'),
        parameters_sha256=digest(parameters), workloads_sha256=parameters.get('workloads_sha256'),
        scope='individual_MLP_cases_GPU_projection_not_full_model_or_CPU_EMC',
        note='Numerical evaluation only; supplied provenance is not independently certified. '
             'No profiler-duration fitting or request-energy allocation.', cases=results)


def demo_parameters():
    # Analytic fixture, deliberately NOT 5090 specifications or fitted parameters.
    return dict(schema=SCHEMA, units=UNITS, illustrative=True, cases=[dict(
        case_id='illustrative-arithmetic', workload=dict(rank=0, uuid='SYNTHETIC', work_flop=100),
        eq1=dict(data_byte=100, peak_flops=10, bandwidth_Bps=20, overhead_s=1,
                 data_semantics='illustrative bytes', capacity_condition='illustrative',
                 parameter_source='hand_calculation_fixture'),
        eq2=dict(alpha=1e-9, voltage_v=1, frequency_hz=1e9, k1=.1, k2=1,
                 temperature_c=40, power_scope='illustrative_domain',
                 parameter_source='hand_calculation_fixture'))])


def save_report(out, result):
    write_new(out/'predictions.json', result)
    lines = [result['status'], 'case | latency s | power W | model energy J | missing Eq1 / Eq2']
    for row in result['cases']:
        lines.append(f"{row['case_id']} | {row['latency_s']} | {row['power_w']} | "
                     f"{row['model_energy_j']} | {','.join(row['missing_eq1'])} / {','.join(row['missing_eq2'])}")
    lines.append('parameters_validated=false; hardware_validated=false; block_selection_authorized=false')
    (out/'REPORT.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare')
    prepare.add_argument('--workloads', required=True)
    predict = sub.add_parser('predict')
    predict.add_argument('--parameters', required=True)
    predict.add_argument('--workloads')
    predict.add_argument('--allow-illustrative', action='store_true')
    demo = sub.add_parser('demo')
    for p in (prepare, predict, demo):
        p.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    try:
        workloads = read(args.workloads) if getattr(args, 'workloads', None) else None
        if args.command == 'prepare':
            parameters = make_template(workloads)
        elif args.command == 'demo':
            parameters = demo_parameters()
        else:
            parameters = read(args.parameters)
        write_new(out/'parameters.json', parameters)
        if workloads is not None:
            write_new(out/'workloads.json', workloads)
        result = evaluate(parameters, workloads, args.command == 'demo' or
                          getattr(args, 'allow_illustrative', False))
        save_report(out, result)
        print('Saved:', out)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        write_new(out/'failure.json', dict(status='failed', error=f'{type(exc).__name__}: {exc}'))
        raise SystemExit(str(exc))


if __name__ == '__main__':
    main()
