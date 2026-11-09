"""Frozen analytical resource models. No GPU access, regularization or label refit.

Calibration scope is independent resident primitives, never automatic Qwen transfer.
"""
import argparse
import math
from pathlib import Path
import numpy as np
from .observe_core import digest
from .service_core import read, write_new, errors


def check_rows(rows):
    for r in rows:
        for key in ('frequency_hz','voltage_v','temperature_c','memory_mhz','latency_s','energy_j','data_byte'):
            if type(r[key]) not in (float,int) or not math.isfinite(r[key]) or r[key]<=0:
                raise ValueError('Invalid physical field: '+key)
        if r['work_flop'] is not None and (type(r['work_flop']) not in (float,int) or not math.isfinite(r['work_flop']) or r['work_flop']<0):
            raise ValueError('Invalid work')
        if len(r['integral_basis'])!=3 or any(not math.isfinite(x) or x<=0 for x in r['integral_basis']):
            raise ValueError('Invalid power basis')
        if not math.isfinite(r['energy_relative_budget']) or not 0<=r['energy_relative_budget']<=.03:
            raise ValueError('Energy quality budget failed')


def resource_class(overload):
    if 'all_reduce' in overload or 'all_gather' in overload: return 'paired_collective'
    if any(s in overload for s in ('attention', 'cache', 'gdn')): return 'stateful_attention'
    if 'linear' in overload or 'mm.' in overload: return 'bf16_tensor_candidate'
    if any(s in overload for s in ('rsqrt', 'sigmoid', 'silu', 'mean.', 'sum.', 'pow.')): return 'special_or_reduction'
    if any(s in overload for s in ('copy_', 'to.', '_to_copy')): return 'conversion_or_memory'
    if any(s in overload for s in ('add.', 'sub.', 'mul.')): return 'floating_arithmetic'
    return 'unresolved_not_zero_cost'


def solve(x, y, budget):
    """OLS with exact marginal energy-budget sensitivity; O(n*p), not 2**n.

Features held fixed. This is NOT a confidence interval or sensor certificate.
"""
    x, y, budget = (np.asarray(v, dtype=float) for v in (x, y, budget))
    if x.ndim != 2 or len(x) <= x.shape[1] or y.shape != (len(x),) or budget.shape != y.shape:
        raise ValueError('Need more independent observations than parameters')
    if not all(np.isfinite(v).all() for v in (x,y,budget)) or (y<=0).any() or (budget<0).any():
        raise ValueError('Invalid regression values')
    scale=np.linalg.norm(x,axis=0)
    if (scale==0).any(): return dict(status='unidentified', blockers=['zero_design_column'])
    z=x/scale; rank=int(np.linalg.matrix_rank(z)); condition=float(np.linalg.cond(z))
    result=dict(rank=rank, columns=x.shape[1], condition=condition, blockers=[])
    if rank!=x.shape[1] or condition>1000:
        return dict(result,status='unidentified',blockers=['rank_or_condition_failed'])
    inverse=np.linalg.pinv(z)/scale[:,None]
    coefficients=inverse@y
    delta=np.abs(inverse)@(budget*y)
    sensitivity=delta/np.maximum(np.abs(coefficients),1e-30)
    if (sensitivity>.2).any():result['blockers'].append('parameter_sensitivity_over_20_percent')
    result.update(status='candidate',coefficients=coefficients.tolist(),
        relative_sensitivity=sensitivity.tolist(),ranges=np.column_stack((coefficients-delta,coefficients+delta)).tolist(),
        uncertainty_scope='energy_or_timing_budget_only_fixed_features_not_confidence_interval')
    return result


def fit_latency(rows):
    """Eq1 h+max(a*W/f,b*D), calibrated fixed-memory-frequency interval.

Active sets solved from several deterministic initial branch assignments.
No per-occurrence coefficients. Failure to identify both paths is explicit.
"""
    if any(r['work_flop'] is None for r in rows):
        return dict(status='unidentified',blockers=['special_or_conversion_work_not_FLOP'])
    if all(r['work_flop']==0 for r in rows):
        fit=solve([[r['data_byte'],1] for r in rows],[r['latency_s'] for r in rows],[0]*len(rows))
        if fit['status']=='candidate':
            b,h=fit['coefficients']
            if b<=0 or h<0:return dict(status='unidentified',blockers=['nonphysical_memory_fit'])
            fit.update(a_s_cycles_per_flop=0,b_s_per_byte=b,overhead_s=h,compute_path='zero_work_not_peak_capacity_estimate')
        return fit
    w=np.array([r['work_flop']/r['frequency_hz'] for r in rows])
    d=np.array([r['data_byte'] for r in rows]); y=np.array([r['latency_s'] for r in rows])
    candidates=[]
    # Thresholds enumerate changes in arithmetic intensity, not holdout labels.
    intensity=w/np.maximum(d,1)
    for threshold in np.unique(intensity):
        mask=intensity>=threshold
        for _ in range(12):
            x=np.column_stack((w*mask,d*(~mask),np.ones(len(rows))))
            fit=solve(x,y,[r.get('timing_relative_budget',0) for r in rows])
            if fit['status']!='candidate':break
            a,b,h=fit['coefficients']
            if min(a,b)<=0 or h<0:break
            next_mask=a*w>=b*d
            if np.array_equal(mask,next_mask):
                fit.update(a_s_cycles_per_flop=a,b_s_per_byte=b,overhead_s=h,
                    loss=float(np.sum((h+np.maximum(a*w,b*d)-y)**2)))
                candidates.append(fit);break
            mask=next_mask
    if not candidates:return dict(status='unidentified',blockers=['Eq1_compute_memory_overhead_not_jointly_identified'])
    return min(candidates,key=lambda r:r['loss'])


def fit(rows):
    if not rows or any(r['split']!='train' for r in rows):raise ValueError('Fit accepts training rows only')
    ids=[r['id'] for r in rows]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate training IDs')
    check_rows(rows)
    if len({r['identity_fingerprint'] for r in rows})!=1:raise ValueError('Mixed service identity')
    result=dict(schema='resource-model-v1',training_ids=ids,training_sha256=digest(rows),
        training_requested_mhz=sorted({r['requested_mhz'] for r in rows}),
        identity_fingerprint=rows[0]['identity_fingerprint'],models={},deployable=False,
        scope='independent_primitives_conditional_board_power_not_Qwen_operator_certificate')
    joint={}
    for rank in sorted({r['rank'] for r in rows}):
        group=[r for r in rows if r['rank']==rank]
        families=sorted({r['family'] for r in group})
        x=[]
        for row in group:
            dynamic=[row['integral_basis'][0] if row['family']==f else 0 for f in families]
            x.append(dynamic+row['integral_basis'][1:])
        regression=solve(x,[r['energy_j'] for r in group],[r['energy_relative_budget'] for r in group])
        if regression['status']=='candidate':
            if min(regression['coefficients'][:-1])<0:regression['blockers'].append('negative_activity_or_thermal_slope')
            regression['residuals']=[dict(id=r['id'],family=r['family'],requested_mhz=r['requested_mhz'],
                temperature_c=r['temperature_c'],observed_j=r['energy_j'],
                residual_j=float(np.dot(v,regression['coefficients'])-r['energy_j'])) for r,v in zip(group,x)]
        joint[rank]=(families,regression)
    result['shared_thermal_fits']={str(k):v[1] for k,v in joint.items()}
    for rank,family in sorted({(r['rank'],r['family']) for r in rows}):
        group=[r for r in rows if (r['rank'],r['family'])==(rank,family)]
        if any(r['data_semantics']!='logical_tensor_bytes' for r in group):raise ValueError('Mixed byte semantics')
        if len({r['memory_mhz'] for r in group})!=1:raise ValueError('Memory frequency changed')
        rates=sorted({r['requested_mhz'] for r in group})
        if len(rates)<2 or rates!=result['training_requested_mhz']:raise ValueError('Each resource requires the complete multi-frequency training grid')
        families,regression=joint[rank]
        power=dict(regression)
        if power['status']=='candidate':
            indices=[families.index(family),len(families),len(families)+1]
            for key in ('coefficients','relative_sensitivity','ranges'):
                power[key]=[regression[key][i] for i in indices]
            power.pop('residuals',None)
        result['models'][f'{rank}:{family}']=dict(latency=fit_latency(group),power=power,
            training_shapes=sorted({str(r['shape']) for r in group}),
            frequency_range_hz=[min(r['frequency_hz'] for r in group),max(r['frequency_hz'] for r in group)],
            voltage_range_v=[min(r['voltage_v'] for r in group),max(r['voltage_v'] for r in group)],
            temperature_range_c=[min(r['temperature_c'] for r in group),max(r['temperature_c'] for r in group)],
            same_path_verified=False,parameters_validated=False,
            assumptions=['core_voltage_vs_board_energy','alpha_constant_within_resource_workload',
                         'feature_sensor_uncertainty_unverified','logical_bytes_not_DRAM_bytes'])
    return result


def validate(model, rows):
    if not rows or any(r['split']!='holdout' for r in rows):raise ValueError('Require holdout rows')
    ids=[r['id'] for r in rows]
    if len(set(ids))!=len(ids) or set(ids)&set(model['training_ids']):raise ValueError('Repeated holdout IDs')
    check_rows(rows)
    groups={}; missing=[]
    if any(r['requested_mhz'] in model.get('training_requested_mhz',[2250,2527]) for r in rows):
        raise ValueError('Require unseen intermediate frequency')
    for r in rows:
        if r['identity_fingerprint']!=model['identity_fingerprint']:raise ValueError('Identity changed')
        key=f"{r['rank']}:{r['family']}"; m=model['models'].get(key)
        if not m or m['latency']['status']!='candidate' or m['power']['status']!='candidate':
            missing.append(r['id']);continue
        lo,hi=m['frequency_range_hz']
        if not lo<=r['frequency_hz']<=hi:missing.append(r['id']);continue
        if any(not m[k+'_range_'+unit][0]<=r[k+'_'+unit]<=m[k+'_range_'+unit][1]
               for k,unit in [('voltage','v'),('temperature','c')]):
            missing.append(r['id']);continue
        if r['requested_mhz'] in model.get('training_requested_mhz',[2250,2527]):raise ValueError('Require unseen intermediate frequency')
        if str(r['shape']) in m['training_shapes']:raise ValueError('Holdout repeats training shape')
        t=m['latency']; duration=t['overhead_s']+max(t['a_s_cycles_per_flop']*r['work_flop']/r['frequency_hz'],t['b_s_per_byte']*r['data_byte'])
        # Conditional on observed feature trajectory; NEVER prospective operator energy.
        energy=float(np.dot(m['power']['coefficients'],r['integral_basis']))
        if energy<=0:missing.append(r['id']);continue
        groups.setdefault(key,[]).append(dict(id=r['id'],requested_mhz=r['requested_mhz'],predicted_latency_s=duration,
            observed_latency_s=r['latency_s'],predicted_energy_j=energy,observed_energy_j=r['energy_j']))
    scores={k:{field:errors([r['observed_'+field] for r in g],[r['predicted_'+field] for r in g])
               for field in ('latency_s','energy_j')} for k,g in groups.items()}
    per_frequency={}
    for key,group in groups.items():
        for rate in sorted({r['requested_mhz'] for r in group}):
            subset=[r for r in group if r['requested_mhz']==rate]
            per_frequency[f'{key}:{rate}']={field:errors([r['observed_'+field] for r in subset],
                [r['predicted_'+field] for r in subset]) for field in ('latency_s','energy_j')}
    return dict(status='conditional_validation_complete',model_sha256=digest(model),rows=groups,scores=scores,
        scores_by_frequency=per_frequency,
        unresolved_rows=missing,parameter_blockers={k:m['latency']['blockers']+m['power']['blockers'] for k,m in model['models'].items()},
        parameters_updated=False,parameters_validated=False,deployable=False,
        conditional_on_measured_holdout_V_f_T=True,
        numeric_limits_passed=not missing and bool(scores) and all(s['wape']<=.1 and s['p90_relative_error']<=.2 for g in per_frequency.values() for s in g.values()))


def select_budget(candidates, budget_s):
    if not math.isfinite(budget_s) or budget_s<=0:raise ValueError('Explicit positive latency budget required')
    if not candidates:raise ValueError('Empty candidates')
    for r in candidates.values():
        if any(not math.isfinite(r[k]) or r[k]<=0 for k in ('latency_s','energy_j')):raise ValueError('Invalid prediction')
    feasible=[f for f in candidates if candidates[f]['latency_s']<=budget_s]
    if not feasible:raise ValueError('No candidate meets latency budget')
    return min(feasible,key=lambda f:(candidates[f]['energy_j'],candidates[f]['latency_s'],int(f)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['merge','fit','validate'])
    p.add_argument('--input',required=True);p.add_argument('--model');p.add_argument('--out',required=True)
    p.add_argument('--extra',nargs='+')
    a=p.parse_args();data=read(a.input)
    if a.command=='merge':
        if not a.extra:p.error('--extra required for merge')
        from .resource_frequencies import merge_training
        result=merge_training([data]+[read(path) for path in a.extra])
        write_new(Path(a.out),result);print('Merged training settings:',result['training_requested_mhz']);return
    if data.get('status')!='resource_collection_complete':raise ValueError('Complete collection required; no partial fitting')
    if data.get('collection_role')=='pilot':raise ValueError('Pilot excluded from fitting and validation')
    if a.command=='validate' and not a.model:p.error('--model required for validate')
    result=fit(data['rows']) if a.command=='fit' else validate(read(a.model),data['rows'])
    write_new(Path(a.out),result);print(result.get('status','resource_model_frozen'), 'deployable=False')

if __name__=='__main__':main()
