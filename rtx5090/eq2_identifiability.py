"""Offline Eq.(2)-basis audit of existing board-energy windows; no GPU access."""
import argparse
import itertools
from pathlib import Path
import numpy as np
from .observe_core import GPU_UUIDS, digest
from .service_core import read, write_new, errors
from .host_reference import local_energy
from .nvapi_voltage import summarize_voltage


def features(data, uuid):
    samples = [s for frame in data['frames'] for s in frame if s['uuid'] == uuid]
    voltage = summarize_voltage(samples)
    start, end = data['host_window_start_ns'], data['host_window_end_ns']
    series = []
    for field in ('voltage', 'temperature_c', 'clock_mhz'):
        ts, values = [], []
        for s in samples:
            if field == 'voltage':
                q = s[field]
                a,b,value = q['query_start_ns'],q['query_end_ns'],q['reported_core_voltage_v']
            else:
                q = s['field_query_times_ns'][field]
                a,b,value = q['start'],q['end'],s[field]
            ts.append(((a+b)/2-start)/1e9)
            values.append(value)
        ts, values = np.asarray(ts), np.asarray(values, dtype=float)
        if not np.isfinite(values).all() or (np.diff(ts) <= 0).any() or (np.diff(ts) > .35).any():
            raise ValueError('Invalid feature samples: '+field)
        if ts[0] > 0 or ts[-1] < (end-start)/1e9:
            raise ValueError('Feature boundary not bracketed: '+field)
        if (values <= 0).any(): raise ValueError('Nonpositive feature: '+field)
        series.append((ts, values))
    duration = (end-start)/1e9
    grid = np.unique(np.concatenate(([0., duration], *[t[(t>0)&(t<duration)] for t,v in series])))
    v,t,f = [np.interp(grid, a,b) for a,b in series]
    basis = np.column_stack((v*v*f*1e6, t*v, v))
    # Trapezoid over the union of per-field query midpoints; never multiply means.
    integrals = np.sum((basis[:-1]+basis[1:])*.5*np.diff(grid)[:,None], axis=0)
    return dict(mean_basis=(integrals/duration).tolist(), integral_basis=integrals.tolist(),
                duration_s=duration, voltage=voltage,
                ranges=dict(voltage_v=[float(v.min()),float(v.max())],
                            temperature_c=[float(t.min()),float(t.max())],
                            frequency_hz=[float(f.min()*1e6),float(f.max()*1e6)]),
                feature_alignment='linear_interpolation_at_query_midpoints_not_sensor_update_certificate')


def audit_matrix(x, y, relative_budget):
    """Training only. Column normalization removes unit-dependent conditioning."""
    x,y = np.asarray(x,dtype=float),np.asarray(y,dtype=float)
    budget = np.asarray(relative_budget,dtype=float)
    if x.ndim != 2 or x.shape[1] != 3 or not 3 <= len(x) <= 5 or y.shape != (len(x),) or budget.shape != (len(x),):
        raise ValueError('Require three to five training windows and three basis terms')
    if not all(np.isfinite(v).all() for v in (x,y,budget)) or (x<=0).any() or (y<=0).any() or (budget<0).any():
        raise ValueError('Nonfinite/nonpositive regression inputs')
    scale = np.linalg.norm(x,axis=0)
    z = x/scale
    _,s,vh = np.linalg.svd(z,full_matrices=False)
    rank = int(np.linalg.matrix_rank(z))
    condition = float(s[0]/s[-1]) if rank == 3 else None
    result = dict(n_independent_windows=len(x), n_parameters=3, residual_degrees_of_freedom=len(x)-3,
                  column_norms=scale.tolist(), singular_values=s.tolist(), rank=rank,
                  normalized_condition_number=condition,
                  weakest_scaled_parameter_direction=vh[-1].tolist(),
                  parameters=None, blockers=[], coefficient_uncertainty_test='all_energy_budget_corners_fixed_features')
    if rank != 3: result['blockers'].append('rank_deficient')
    if condition is not None and condition > 1000:
        result['blockers'].append('normalized_condition_exceeds_1000')
    if result['blockers']: return result
    scaled = np.linalg.lstsq(z,y,rcond=None)[0]
    coefficients = scaled/scale
    candidates = np.asarray([np.linalg.lstsq(z,y*(1+budget*np.array(signs)),rcond=None)[0]/scale
                             for signs in itertools.product((-1,1),repeat=len(x))])
    deviations = np.max(np.abs(candidates-coefficients),axis=0)
    relative = deviations/np.maximum(np.abs(coefficients),1e-30)
    result.update(candidate_parameters=dict(zip(('alpha','k1','k2'), coefficients.tolist())),
                  coefficient_relative_sensitivity=relative.tolist(),
                  coefficient_budget_ranges=np.column_stack((candidates.min(axis=0),candidates.max(axis=0))).tolist())
    if (relative > .2).any(): result['blockers'].append('coefficient_energy_budget_sensitivity_exceeds_20_percent')
    if coefficients[0] < 0 or coefficients[1] < 0:
        result['blockers'].append('negative_dynamic_or_temperature_coefficient')
    if not result['blockers']: result['parameters'] = result['candidate_parameters']
    return result


def run(source, out, extra=None):
    out.mkdir(parents=True,exist_ok=False)
    reference = read(source/'reference.json')
    if reference['status'] != 'host_envelope_reference_complete': raise ValueError('Incomplete collection')
    protocol = read(source/'protocol.json')
    if not protocol.get('voltage_requested'): raise ValueError('Simultaneous voltage was not requested')
    if (protocol['requests'],protocol['input'],protocol['output'],protocol['concurrency']) != (20,512,64,1):
        raise ValueError('Workload changed')
    result = dict(status='offline_eq2_audit_complete', identity_fingerprint=reference['identity_fingerprint'],
                  source_reference_sha256=digest(reference), protocol_sha256=digest(protocol),
                  formula='P=alpha*V^2*f_Hz+k1*T_C*V+k2*V',
                  scope='conditional_board_window_basis_not_operator_power',
                  equation_2_parameters_validated=False, clock_writes=False, ranks=[],
                  unresolved=['core_voltage_vs_whole_board_power_domain', 'sensor_update_period_unknown',
                              'constant_activity_factor_over_mixed_workload_assumed',
                              'few_independent_windows'],
                  implementation_choices=dict(condition_limit=1000, coefficient_sensitivity_limit=.2,
                       fit='unconstrained_solve_then_physical_and_sensitivity_checks_no_regularization',
                       feature_units=['V^2*Hz','C*V','V'], pilot_excluded=True,
                       temperature='NVML_Celsius_not_independently_verified_junction_temperature',
                       feature_error_budget_known=False))
    def load(name, base=None):
        folder=(base or source)/name
        window=read(folder/'window.json')
        summary=read(folder/'summary.json')
        if window['identity_fingerprint'] != reference['identity_fingerprint'] or summary['status'] != 'valid_host_envelope':
            raise ValueError('Identity/window mismatch')
        data=read(folder/'windows-samples.json')
        energy=local_energy(data)
        if energy['quality_errors']: raise ValueError(str(energy['quality_errors']))
        return {u:dict(name=name, **features(data,u), energy_j=energy['by_uuid'][u]['energy_j'],
                       energy_budget=energy['by_uuid'][u]['relative_uncertainty'],
                       samples_sha256=digest(data)) for u in GPU_UUIDS}
    training = [load('reference-'+str(i)) for i in (1,2,3)]
    if extra is not None:
        manifest=read(extra/'collection.json')
        if manifest['status'] != 'complete' or manifest['identity_fingerprint'] != reference['identity_fingerprint']:
            raise ValueError('Supplemental collection incomplete or wrong service')
        if not manifest['restoration']['verified'] or manifest['windows'] != ['requested-2527','requested-2250']:
            raise ValueError('Require restored two-condition diagnostic')
        training += [load(n,extra) for n in manifest['windows']]
        result['extra_manifest_sha256']=digest(manifest)
        result['unresolved'].append('holdouts_only_cover_original_current_policy_not_new_frequency_conditions')
    for rank,uuid in enumerate(GPU_UUIDS):
        rows=[w[uuid] for w in training]
        audit=audit_matrix([r['mean_basis'] for r in rows],
                           [r['energy_j']/r['duration_s'] for r in rows], [r['energy_budget'] for r in rows])
        if audit['parameters']:
            c=audit['parameters']
            if any(c['k1']*t+c['k2'] < 0 for r in rows for t in r['ranges']['temperature_c']):
                audit['blockers'].append('negative_leakage_in_training_range'); audit['parameters']=None
        result['ranks'].append(dict(rank=rank,uuid=uuid,training=rows,audit=audit,validation=None))
    # Persist the decision and coefficients BEFORE reading holdout observations.
    write_new(out/'training-audit.json',result)
    if any(r['audit']['parameters'] for r in result['ranks']):
        held=[load('holdout-'+str(i)) for i in (1,2,3)]
        for r in result['ranks']:
            c=r['audit']['parameters']
            if not c: continue
            coeff=np.array([c[k] for k in ('alpha','k1','k2')])
            rows=[h[r['uuid']] for h in held]
            prediction=[float(np.dot(v['integral_basis'],coeff)) for v in rows]
            if min(prediction) <= 0 or any(c['k1']*t+c['k2'] < 0 for v in rows for t in v['ranges']['temperature_c']):
                r['validation']=dict(status='nonphysical_holdout_prediction',passed=False); continue
            es=errors([v['energy_j'] for v in rows],prediction)
            ps=errors([v['energy_j']/v['duration_s'] for v in rows], [p/v['duration_s'] for p,v in zip(prediction,rows)])
            r['validation']=dict(status='conditional_board_holdout_scored',
                energy_scores=es,power_scores=ps,predicted_energy_j=prediction,
                actual_energy_j=[v['energy_j'] for v in rows],
                passed=all(s['wape']<=.1 and s['p90_relative_error']<=.2 for s in (es,ps)),
                interpretation='conditional_on_measured_voltage_temperature_frequency_and_duration_not_ahead_of_time_prediction')
    write_new(out/'audit.json',result)
    lines=['Eq.(2) offline identifiability audit; board scope only.']
    for r in result['ranks']:
        a=r['audit']
        lines.append('rank %s: rank=%s condition=%s blockers=%s' % (r['rank'],a['rank'],a['normalized_condition_number'],a['blockers']))
        lines.append('parameters: '+str(a['parameters']))
        lines.append('validation: '+str(r['validation']))
    lines.append('Eq.(2) operator parameters validated: False; see unresolved scope/sensor assumptions in audit.json')
    (out/'REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines)); print('Saved:',out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True); p.add_argument('--out',required=True)
    p.add_argument('--extra',help='Completed two-condition supplemental collection')
    a=p.parse_args(); run(Path(a.source),Path(a.out),Path(a.extra) if a.extra else None)


if __name__ == '__main__': main()
