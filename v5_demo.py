#!/usr/bin/env python3
"""CPU-only synthetic dual-granularity/fit demonstration; never deployable."""
import argparse
import copy
import json
from pathlib import Path
from sparsedvfs.paper_core import timeline_seconds, thermal_power_w
from sparsedvfs.calibration_fit import fit_calibration
from sparsedvfs.graph_partitioner import build_plan
from sparsedvfs_vllm.contracts import PROTOCOL, digest_json
from sparsedvfs_vllm.operator_contracts import layout_entry, marker_name, build_certificate


def synthetic_trace():
    identity = dict(backend=PROTOCOL, granularity='operator', workload=dict(input_tokens=2, output_tokens=2))
    steps, events = [], []
    for request in range(3):
        rid = f'synthetic-{request}'
        for step in range(2):
            context = dict(request_id=rid, phase='prefill' if step == 0 else 'decode', step=step,
                           num_computed_tokens=0 if step == 0 else 2,
                           num_scheduled_tokens=2 if step == 0 else 1)
            calls = []
            for ordinal, op in enumerate(('aten.mm.default', '_C.fused_norm.default', 'aten.sym_size.int')):
                layout = layout_entry('layer.0', ordinal, op, dict(shape=[2 if step == 0 else 1, 5120], dtype='bfloat16'))
                marker = marker_name(rid, step, ordinal)
                calls.append(dict(layout=layout, marker=marker))
                ts = request*1000+step*100+ordinal*10
                events.append(dict(ph='X', name=marker, cat='user_annotation', pid=99, tid=100, ts=ts, dur=8))
                if ordinal < 2:
                    cid = request*100+step*10+ordinal
                    events += [dict(ph='X', name='cudaLaunchKernel', cat='cuda_runtime', pid=99, tid=100,
                                    ts=ts+2, dur=2, args=dict(correlation=cid)),
                               dict(ph='X', name='synthetic_kernel', cat='kernel', pid=0, tid=7,
                                    ts=100000+ts, dur=4, args=dict(correlation=cid, stream=7, device=0))]
            steps.append(dict(context, calls=calls))
    trace = dict(identity=identity, operator_steps=steps, output_sha256=digest_json([1,2]),
                 native_output_sha256=digest_json([1,2]))
    return trace, events


def synthetic_fit_fixture():
    trace, events = synthetic_trace()
    cert = build_certificate(trace, events, synthetic=True)
    units = [(key, name) for key in cert['context_order'] for name in cert['cases'][key]['units']]
    workloads = {'operators': {phase+'/'+name: dict(work_flop=1e9/(i+1), data_byte=1e6*(i+1),
                    workload_provenance='Invented synthetic descriptors; no Qwen kernel measured',
                    data_semantics='explicit_paper_effective_traffic_estimate', s_comp=0, s_mem=0)
                    for i, (phase,name) in enumerate(units)}}
    candidates, opps, capacities, bandwidths, records = [], {}, {}, {}, []
    for cpu in (1000,2000):
        for gpu in (1000000000,2000000000):
            for emc in (1000000000,2000000000):
                triplet = [cpu,gpu,emc]
                peak, bw = 1e12*(gpu/1e9), 1e10*(emc/1e9)
                v = .7+.05*(gpu/1e9)+.02*(cpu/1000)+.01*(emc/1e9)
                capacities[f'{cpu},{gpu}'], bandwidths[str(emc)] = peak,bw
                key = ','.join(map(str,triplet))
                opps[key] = dict(voltage_v=v, effective_frequency_hz=gpu, capacity_provenance='invented synthetic capacities')
                rows = []
                for i,(phase,name) in enumerate(units):
                    unit = phase+'/'+name
                    spec = workloads['operators'][unit]
                    latency = timeline_seconds(spec['work_flop'],spec['data_byte'],0,0,peak,bw,.0002)
                    rows.append(dict(phase=phase,name=name,latency_ms=latency*1000,power_w=3.0))
                    for t in (35.0,45.0,55.0):
                        records.append(dict(unit=unit,triplet=triplet,voltage_v=v,effective_frequency_hz=gpu,
                            temperature_c=t,latency_s=latency,
                            power_w=thermal_power_w((i+1)*1e-9,v,gpu,.01,.2,t),
                            power_scope='synthetic_gpu_rail',power_attribution='isolated_unit_measured',
                            window_duration_s=1.0,sensor_interval_s=.02,
                            power_evidence='Synthetic equation output; not a sensor measurement'))
                candidates.append(dict(triplet=triplet,operators=rows))
    profile = dict(schema=5,synthetic=True,dvfs_applied=False,identity=trace['identity'],
                   granularity='operator',units={k:c['units'] for k,c in cert['cases'].items()},
                   reference=[2000,2000000000,2000000000],candidates=candidates,coverage_certificate=cert)
    observations = dict(schema=5,synthetic=True,identity=trace['identity'],profile_sha256=digest_json(profile),
                        units=dict(power='W',voltage='V',frequency='Hz',temperature='C',latency='s'),
                        power_scope='synthetic_gpu_rail',temperature_sensor='synthetic_temperature',
                        scalar_vf_scope='Invented V and GPC f projection',provenance='pure synthetic fixture',
                        capacities=capacities,bandwidths=bandwidths,opps=opps,unit_records=records)
    return profile, observations, workloads, cert


def demonstrate():
    profile, observed, workloads, cert = synthetic_fit_fixture()
    calibration = fit_calibration(profile, observed, workloads, .001)
    plan = build_plan(profile,100,factor=5,tolerance=0,max_slowdown=.05,
                      selector='paper',calibration=calibration,allow_synthetic=True)
    changed = copy.deepcopy(observed)
    for row in changed['unit_records']:
        row['temperature_c'] = 45
    unidentifiable = fit_calibration(profile,changed,workloads,.001)
    return dict(synthetic=True,deployable=False,tensors_allocated=False,
                certificate_contexts=cert['context_order'],
                gpu_units={k:len(c['units']) for k,c in cert['cases'].items()},
                atomic_fused_units=True,blocks=plan['blocks'],
                fitted_thermal=calibration.get('device_thermal'),
                heldout_error=calibration['v5_validation'].get('max_relative_error'),
                validation=calibration['v5_validation'],
                constant_temperature_rejection=unidentifiable['v5_validation'],
                hardware_energy_or_savings=None,paper_experiments_reproduced=False)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output')
    args=p.parse_args()
    result=demonstrate()
    rendered=json.dumps(result,indent=2,ensure_ascii=False)
    if args.output:
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        Path(args.output).write_text(rendered+'\n',encoding='utf-8')
    print(rendered)
