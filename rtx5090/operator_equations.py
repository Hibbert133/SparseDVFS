"""Eq1/Eq2 prediction and explicit no-slowdown selection. No GPU access or fitting."""
import argparse
import math
from .observe_core import GPU_UUIDS,digest
from .service_core import read,write_new,positive
from sparsedvfs.paper_core import timeline_seconds,thermal_power_w
from .partition_dryrun import partition

RATES=(2250,2527)


def select_no_slowdown(candidates):
    """Declared port choice, separate from literal Algorithm 1."""
    if set(candidates)!=set(RATES):raise ValueError('Require the two supported control settings')
    for row in candidates.values():
        positive(row['latency_s'],'latency');positive(row['energy_j'],'predicted energy')
    fastest=min(row['latency_s'] for row in candidates.values())
    # No epsilon, error-bar relaxation, 5% per-op allowance or energy-based merging.
    feasible=[f for f in RATES if candidates[f]['latency_s']<=fastest]
    return min(feasible,key=lambda f:(candidates[f]['energy_j'],candidates[f]['latency_s'],RATES.index(f)))


def validation_ok(record):
    if not isinstance(record,dict) or not record.get('evidence_sha256') or record.get('independent') is not True:
        return False
    train=record.get('training_ids',[]);held=record.get('holdout_ids',[])
    if not train or not held or set(train)&set(held):return False
    for f in ('2250','2527'):
        for name in ('latency','energy'):
            score=record.get('scores',{}).get(f,{}).get(name,{})
            values=[score.get('wape'),score.get('p90_relative_error')]
            if any(type(x) not in (float,int) or not math.isfinite(x) or x<0 for x in values):return False
            if values[0]>.1 or values[1]>.2:return False
    return True


def audit(bundle):
    if bundle.get('schema')!='operator-equations-v1':raise ValueError('Wrong operator model schema')
    issues=[]
    def need(test,reason,where):
        if not test:issues.append(dict(reason=reason,where=where))
    need(bundle.get('current_identity_verified') is True,'current_service_identity_unverified','bundle')
    need(bool(bundle.get('identity_fingerprint')),'missing_identity','bundle')
    need(bundle.get('units_contract')==dict(time='s',work='FLOP',data='byte',frequency='Hz',voltage='V',power='W'),
         'physical_units_unverified','bundle')
    need(bundle.get('rank_join_verified') is True and bool(bundle.get('rank_join_evidence_sha256')),'rank_graph_join_unverified','bundle')
    if not bundle['operators']:raise ValueError('Empty operator input')
    seen=set()
    for unit in bundle['operators']:
        if unit['id'] in seen:raise ValueError('Duplicate operator occurrence')
        seen.add(unit['id'])
        if len(unit['ranks'])!=2 or {r['rank'] for r in unit['ranks']}!={0,1}:
            raise ValueError('Each operator requires both ranks')
        for row in unit['ranks']:
            where=f"{unit['id']}@{row['rank']}"
            if row['uuid']!=GPU_UUIDS[row['rank']]:raise ValueError('Rank/UUID mismatch')
            need(bool(row.get('workload_provenance')),'workload_definition_missing',where)
            need(bool(row.get('data_semantics')),'data_semantics_missing',where)
            need(not row.get('unresolved_workload'),'workload_compute_path_unresolved',where)
            for field in ('work_flop','data_byte'):
                value=row.get(field)
                need(type(value) in (float,int) and math.isfinite(value) and value>=0,'missing_'+field,where)
            if row.get('s_comp')!=0 or row.get('s_mem')!=0:
                need(False,'effective_sparsity_requires_separate_hardware_adapter',where)
            family=bundle['families'].get(row['family_id'])
            need(family is not None,'missing_parameter_family',where)
            if family:
                need(family.get('rank')==row['rank'] and row['signature_id'] in family.get('signature_ids',[]),
                     'family_signature_mismatch',where)
    for key,family in bundle['families'].items():
        need(family.get('same_path_verified') is True and bool(family.get('path_evidence_sha256')),
             'same_path_state_kernel_evidence_missing',key)
        need(family.get('independent_capacity_calibration') is True and bool(family.get('capacity_evidence_sha256')),
             'independent_capacity_missing',key)
        need(family.get('power_scope')=='same_path_repeated_unit_board_windows','operator_power_scope_missing',key)
        need(family.get('paired_window_verified') is True and bool(family.get('power_evidence_sha256')),
             'paired_unit_power_window_missing',key)
        need(validation_ok(family.get('validation')),'independent_validation_missing_or_failed',key)
        need((family.get('validation') or {}).get('parameters_sha256')==digest(dict(
            alpha=family.get('alpha'),overhead_s=family.get('overhead_s'),capacities=family.get('capacities'),
            thermal=bundle.get('thermal',{}).get(str(family['rank'])))), 'validation_not_bound_to_parameters',key)
        for field in ('alpha','overhead_s'):
            value=family.get(field)
            need(type(value) in (float,int) and math.isfinite(value) and value>=0,'missing_'+field,key)
        for f in RATES:
            cap=family.get('capacities',{}).get(str(f),{})
            for field in ('peak_flops','bandwidth_Bps'):
                value=cap.get(field)
                need(type(value) in (float,int) and math.isfinite(value) and value>0,'missing_'+field,key+':'+str(f))
            state=bundle.get('states',{}).get(str(family['rank']),{}).get(str(f),{})
            bounds=cap.get('frequency_range_hz');value=state.get('effective_frequency_hz')
            need(isinstance(bounds,list) and len(bounds)==2 and all(type(x) in (float,int) and math.isfinite(x) for x in bounds)
                 and type(value) in (float,int) and bounds[0]<=value<=bounds[1],
                 ('frequency_outside_capacity_calibration' if type(value) in (float,int) and math.isfinite(value) and isinstance(bounds,list) and len(bounds)==2 and all(type(x) in (float,int) and math.isfinite(x) for x in bounds) else 'missing_frequency_calibration_coverage'),key+':'+str(f))
    for rank in (0,1):
        key=str(rank);thermal=bundle.get('thermal',{}).get(key,{})
        need(thermal.get('identifiability_passed') is True and bool(thermal.get('evidence_sha256')),
             'thermal_parameters_unidentified',key)
        for field in ('k1','k2'):
            value=thermal.get(field)
            need(type(value) in (float,int) and math.isfinite(value),'missing_'+field,key)
        if type(thermal.get('k1')) in (float,int) and math.isfinite(thermal['k1']):
            need(thermal['k1']>=0,'negative_thermal_slope',key)
        for f in RATES:
            state=bundle.get('states',{}).get(key,{}).get(str(f),{})
            need(state.get('known_before_prediction') is True and bool(state.get('evidence_sha256')),
                 'operating_state_not_available_before_prediction',key+':'+str(f))
            for field in ('effective_frequency_hz','voltage_v','temperature_c'):
                value=state.get(field)
                need(type(value) in (float,int) and math.isfinite(value) and value>0,'missing_'+field,key+':'+str(f))
            interval=thermal.get('temperature_range_c')
            need(isinstance(interval,list) and len(interval)==2 and all(type(x) in (float,int) and math.isfinite(x) for x in interval)
                 and type(state.get('temperature_c')) in (float,int) and interval[0]<=state['temperature_c']<=interval[1],
                 ('temperature_outside_calibration' if type(state.get('temperature_c')) in (float,int) and math.isfinite(state.get('temperature_c')) and isinstance(interval,list) and len(interval)==2 and all(type(x) in (float,int) and math.isfinite(x) for x in interval) else 'missing_temperature_calibration_coverage'),key+':'+str(f))
            voltage_range=thermal.get('voltage_range_v');voltage=state.get('voltage_v')
            need(isinstance(voltage_range,list) and len(voltage_range)==2 and all(type(x) in (float,int) and math.isfinite(x) for x in voltage_range)
                 and type(voltage) in (float,int) and voltage_range[0]<=voltage<=voltage_range[1],
                 ('voltage_outside_calibration' if type(voltage) in (float,int) and math.isfinite(voltage) and isinstance(voltage_range,list) and len(voltage_range)==2 and all(type(x) in (float,int) and math.isfinite(x) for x in voltage_range) else 'missing_voltage_calibration_coverage'),key+':'+str(f))
    return issues


def predict(bundle):
    issues=audit(bundle)
    result=dict(schema='operator-predictions-v1',source_sha256=digest(bundle),
        identity_fingerprint=bundle.get('identity_fingerprint'),deployable=False,clock_writes=False,
        status='blocked_parameter_evidence' if issues else 'equation_predictions_complete',
        blockers=issues,operators={},selection_policy='no_predicted_slowdown_minimum_predicted_energy',
        equation_implementation='Eq1_max_compute_memory_plus_overhead_and_Eq2',
        cpu_emc='unsupported_current_backend',pair_completion='max_of_rank_predictions_requires_join_evidence')
    if issues:return result
    for unit in bundle['operators']:
        options={}
        for f in RATES:
            ranks={}
            for row in unit['ranks']:
                rank=row['rank'];family=bundle['families'][row['family_id']]
                cap=family['capacities'][str(f)];state=bundle['states'][str(rank)][str(f)]
                thermal=bundle['thermal'][str(rank)]
                t=timeline_seconds(row['work_flop'],row['data_byte'],0,0,cap['peak_flops'],cap['bandwidth_Bps'],family['overhead_s'])
                p=thermal_power_w(family['alpha'],state['voltage_v'],state['effective_frequency_hz'],
                                  thermal['k1'],thermal['k2'],state['temperature_c'])
                ranks[str(rank)]=dict(latency_s=t,power_w=p)
            t=max(v['latency_s'] for v in ranks.values())
            # Both boards' power must be calibrated over the same paired-unit window.
            options[f]=dict(latency_s=t,energy_j=t*sum(v['power_w'] for v in ranks.values()),ranks=ranks)
        result['operators'][unit['id']]=dict(selected_mhz=select_no_slowdown(options),
                                             predictions={str(f):v for f,v in options.items()})
    return result


def partition_input(bundle,predictions):
    if predictions['status']!='equation_predictions_complete' or predictions['source_sha256']!=digest(bundle):
        raise ValueError('Complete predictions bound to this frozen bundle required')
    graph=bundle['graph']
    return dict(schema='pair-partition-input-v2',scope=bundle['scope'],synthetic=False,
        nodes=[o['id'] for o in bundle['operators']],edges=graph.get('edges') or [],
        candidates={k:{f:dict(pair_latency_s=v['latency_s']) for f,v in o['predictions'].items()}
                    for k,o in predictions['operators'].items()},
        selected_frequencies={k:o['selected_mhz'] for k,o in predictions['operators'].items()},
        timing_model=graph.get('timing_model') or 'legacy_serial_projection',
        dependency_evidence_sha256=graph.get('dependency_evidence_sha256'),
        serial_nonoverlap_verified=graph.get('serial_nonoverlap_verified') is True,
        switch_s=bundle.get('switch_s'),switch_bound_verified=bundle.get('switch_bound_verified') is True,
        identity_fingerprint=bundle['identity_fingerprint'],N=5,similarity=.05)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--out',required=True)
    a=p.parse_args()
    from pathlib import Path
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    bundle=read(a.input);result=predict(bundle);write_new(out/'predictions.json',result)
    print(result['status'],'missing evidence:',len(result['blockers']))
    if not result['blockers']:
        value=partition_input(bundle,result);write_new(out/'partition-input.json',value)
        blocks=partition(value);write_new(out/'partition.json',blocks)
        print(blocks['status'],'blocks:',len(blocks['blocks']))
    print('Saved:',out,'deployable=False')


if __name__=='__main__':main()
