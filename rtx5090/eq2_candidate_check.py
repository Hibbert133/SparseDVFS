"""Offline conditional holdout predictions from unvalidated Eq2 candidates."""
import argparse
import itertools
from pathlib import Path
import numpy as np
from .eq2_identifiability import features
from .host_reference import local_energy
from .observe_core import GPU_UUIDS, digest
from .service_core import read, write_new, errors


def candidates(rank):
    rows=rank['training']
    if len(rows)!=5: raise ValueError('Require five-window supplemental audit')
    x=np.array([r['mean_basis'] for r in rows],float)
    y=np.array([r['energy_j']/r['duration_s'] for r in rows],float)
    budget=np.array([r['energy_budget'] for r in rows],float)
    scale=np.linalg.norm(x,axis=0); z=x/scale
    if np.linalg.matrix_rank(z)!=3: raise ValueError('Rank deficient training design')
    supplied=rank['audit']['candidate_parameters']
    center=np.array([supplied[k] for k in ('alpha','k1','k2')])
    reconstructed=np.linalg.lstsq(z,y,rcond=None)[0]/scale
    if not np.allclose(center,reconstructed,rtol=1e-8,atol=1e-18):
        raise ValueError('Candidate does not match frozen training evidence')
    corners=np.array([np.linalg.lstsq(z,y*(1+budget*np.array(s)),rcond=None)[0]/scale
                      for s in itertools.product((-1,1),repeat=5)])
    return center,corners


def prediction(center,corners,feature,observed,budget):
    basis=np.array(feature['integral_basis'])
    p=float(basis@center); values=corners@basis
    lo,hi=float(values.min()),float(values.max())
    duration=feature['duration_s']
    return dict(predicted_energy_j=p, prediction_range_j=[lo,hi], observed_energy_j=observed,
                predicted_power_w=p/duration, prediction_range_w=[lo/duration,hi/duration],
                observed_power_w=observed/duration,
                observed_within_prediction_range=lo<=observed<=hi,
                observed_budget_range_j=[observed*(1-budget),observed*(1+budget)],
                intervals_overlap=lo<=observed*(1+budget) and hi>=observed*(1-budget),
                max_corner_relative_error=float(np.max(np.abs(values-observed))/observed),
                relative_half_width=(hi-lo)/(2*abs(p)) if p else None,
                all_corner_predictions_positive=bool((values>0).all()))


def run(a):
    audit=read(a.audit); source=Path(a.source); out=Path(a.out)
    reference=read(source/'reference.json')
    if digest(reference)!=audit['source_reference_sha256'] or reference['identity_fingerprint']!=audit['identity_fingerprint']:
        raise ValueError('Reference provenance mismatch')
    if {r['uuid'] for r in audit['ranks']}!=set(GPU_UUIDS) or len(audit['ranks'])!=2:
        raise ValueError('Require both GPU ranks')
    frozen=[candidates(r) for r in audit['ranks']] # BEFORE reading any holdout.
    out.mkdir(parents=True,exist_ok=False)
    report=dict(status='candidate_holdout_diagnostic_complete',audit_sha256=digest(audit),
                identity_fingerprint=audit['identity_fingerprint'],parameters_updated=False,
                equation_2_parameters_validated=False,scope='original_policy_board_windows_only',
                range_semantics='32_training_energy_budget_corners_fixed_features_not_confidence_interval',
                limitations=['holdouts_previously_seen','V_T_f_duration_measured_not_forecast',
                             'feature_and_sensor_uncertainty_not_included','no_operator_or_new_frequency_validation'],ranks=[])
    for rank,(center,corners) in zip(audit['ranks'],frozen):
        rows=[]
        for i in (1,2,3):
            folder=source/('holdout-'+str(i))
            window=read(folder/'window.json')
            if window['identity_fingerprint']!=audit['identity_fingerprint'] or window['status']!='loaded_host_sampling_passed':
                raise ValueError('Holdout identity/status mismatch')
            data=read(folder/'windows-samples.json'); energy=local_energy(data)
            if energy['quality_errors']: raise ValueError(str(energy['quality_errors']))
            f=features(data,rank['uuid']); e=energy['by_uuid'][rank['uuid']]
            row=prediction(center,corners,f,e['energy_j'],e['relative_uncertainty'])
            row.update(name=folder.name,samples_sha256=digest(data))
            rows.append(row)
        valid=all(r['all_corner_predictions_positive'] and r['predicted_energy_j']>0 for r in rows)
        scores={}
        if valid:
            for name in ('energy_j','power_w'):
                scores[name]=errors([r['observed_'+name] for r in rows],[r['predicted_'+name] for r in rows])
        result=dict(rank=rank['rank'],uuid=rank['uuid'],rows=rows,center_scores=scores,
                    center_parameters=center.tolist(),corner_parameters=corners.tolist(),
                    negative_alpha_or_k1_corners=int(np.sum((corners[:,0]<0)|(corners[:,1]<0))),
                    all_predictions_positive=valid,
                    center_numeric_limits_met=valid and all(s['wape']<=.1 and s['p90_relative_error']<=.2 for s in scores.values()),
                    worst_corner_relative_error=max(r['max_corner_relative_error'] for r in rows))
        report['ranks'].append(result)
    write_new(out/'candidate-check.json',report)
    lines=['Unvalidated Eq2 candidates; conditional board holdout diagnostic, no refit.',
           'rank/window | predicted J | observed J | budget-corner range J | worst corner error %']
    for r in report['ranks']:
        for w in r['rows']:
            lines.append('%s/%s | %.2f | %.2f | [%.2f, %.2f] | %.2f' %
                (r['rank'],w['name'],w['predicted_energy_j'],w['observed_energy_j'],
                 *w['prediction_range_j'],100*w['max_corner_relative_error']))
        lines.append('rank %s center scores: %s' % (r['rank'],r['center_scores']))
        lines.append('negative alpha/k1 corners retained: '+str(r['negative_alpha_or_k1_corners']))
    lines.append('Eq2 parameters validated: False; intervals are sensitivity ranges, not confidence intervals.')
    (out/'REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines)); print('Saved:',out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('audit','source','out'): p.add_argument('--'+name,required=True)
    run(p.parse_args())


if __name__=='__main__': main()
