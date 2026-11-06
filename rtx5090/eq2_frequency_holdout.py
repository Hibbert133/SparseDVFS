"""Freeze existing Eq2 candidates, then score two fresh requested-clock windows."""
import argparse
from pathlib import Path
import numpy as np
from .eq2_candidate_check import candidates, prediction
from .eq2_identifiability import features
from .host_reference import local_energy
from .observe_core import digest, GPU_UUIDS
from .service_core import read, write_new, errors

NAMES=['requested-2527','requested-2250']


def request_hashes(folder):
    rows=read(folder/'requests.json')
    hashes=[r['prompt_hash'] for r in rows]
    if len(rows)!=20 or len(set(hashes))!=20: raise ValueError('Require 20 unique requests')
    return set(hashes)


def freeze(a):
    audit=read(a.audit); source=Path(a.source); extra=Path(a.training_extra)
    if digest(read(source/'reference.json'))!=audit['source_reference_sha256']:
        raise ValueError('Original reference changed')
    if digest(read(extra/'collection.json'))!=audit['extra_manifest_sha256']:
        raise ValueError('Supplemental training changed')
    if [r['uuid'] for r in audit['ranks']]!=list(GPU_UUIDS): raise ValueError('Wrong rank order')
    used=set()
    for folder in [source/'pilot']+[source/(s+'-'+str(i)) for s in ('reference','holdout') for i in (1,2,3)]+[extra/n for n in NAMES]:
        used.update(request_hashes(folder))
    ranks=[]
    for r in audit['ranks']:
        center,corners=candidates(r)
        ranks.append(dict(rank=r['rank'],uuid=r['uuid'],center=center.tolist(),corners=corners.tolist()))
    value=dict(schema='eq2-frozen-candidate-holdout-v1',identity_fingerprint=audit['identity_fingerprint'],
               audit_sha256=digest(audit),ranks=ranks,excluded_prompt_hashes=sorted(used),
               equation_2_parameters_validated=False,requested_mhz=[2527,2250],requests_per_condition=20,
               limits=dict(wape=.1,p90=.2,per_condition_relative_error=.1),
               scope='conditional_board_energy_given_measured_V_T_f_duration')
    write_new(a.out,value)
    print('Frozen candidates:',a.out); print('SHA256:',digest(value))


def require_manifest(manifest,frozen,names=None):
    if (manifest['status']!='complete' or not manifest['restoration']['verified'] or
        manifest['identity_fingerprint']!=frozen['identity_fingerprint'] or
        manifest.get('frozen_sha256')!=digest(frozen) or manifest['windows']!=(names or NAMES)):
        raise ValueError('Require complete fresh holdout bound to frozen parameters and verified restoration')


def check(a):
    frozen=read(a.frozen); source=Path(a.collection); manifest=read(source/'collection.json')
    require_manifest(manifest,frozen)
    replacement=Path(a.replacement_2527) if getattr(a,'replacement_2527',None) else None
    replacement_manifest=read(replacement/'collection.json') if replacement else None
    if replacement: require_manifest(replacement_manifest,frozen,['requested-2527'])
    excluded=set(frozen['excluded_prompt_hashes'])
    windows=[]
    for name in NAMES:
        folder=(replacement if replacement and name=='requested-2527' else source)/name
        hashes=request_hashes(folder)
        if hashes & excluded: raise ValueError('Holdout repeats previous prompts')
        excluded.update(hashes)
        window=read(folder/'window.json')
        if window['identity_fingerprint']!=frozen['identity_fingerprint'] or window['status']!='loaded_host_sampling_passed' or window['cache_status']!='verified_miss':
            raise ValueError('Invalid holdout window')
        data=read(folder/'windows-samples.json'); energy=local_energy(data)
        if energy['quality_errors']: raise ValueError(str(energy['quality_errors']))
        windows.append((name,data,energy))
    report=dict(status='frozen_frequency_holdout_scored',frozen_sha256=digest(frozen),
                replacement_2527_manifest_sha256=digest(replacement_manifest) if replacement else None,
                collection_sha256=digest(manifest),parameters_updated=False,
                equation_2_parameters_validated=False,restoration_verified=True,ranks=[],
                caveat='One window per condition; measured V/T/f/duration inputs, board scope only; ranges are not confidence intervals')
    lines=['Frozen Eq2 candidate: fresh two-condition holdout; no refit.',
           'rank/condition | prediction J | measured J | error % | corner range J | worst corner error %']
    for r in frozen['ranks']:
        rows=[]
        for name,data,energy in windows:
            f=features(data,r['uuid']); e=energy['by_uuid'][r['uuid']]
            row=prediction(np.array(r['center']),np.array(r['corners']),f,e['energy_j'],e['relative_uncertainty'])
            row.update(name=name,samples_sha256=digest(data),feature_ranges=f['ranges'])
            row['relative_error']=abs(row['predicted_energy_j']/row['observed_energy_j']-1)
            rows.append(row)
            lines.append('%s/%s | %.2f | %.2f | %.2f | [%.2f, %.2f] | %.2f' %
                (r['rank'],name,row['predicted_energy_j'],row['observed_energy_j'],row['relative_error']*100,
                 *row['prediction_range_j'],row['max_corner_relative_error']*100))
        scores={}
        if all(w['predicted_energy_j']>0 for w in rows):
            for k in ('energy_j','power_w'):
                scores[k]=errors([w['observed_'+k] for w in rows],[w['predicted_'+k] for w in rows])
        passed=bool(scores) and all(w['relative_error']<=.1 for w in rows) and all(s['wape']<=.1 and s['p90_relative_error']<=.2 for s in scores.values())
        report['ranks'].append(dict(rank=r['rank'],uuid=r['uuid'],rows=rows,scores=scores,
                                    center_numeric_limits_met=passed))
    report['center_numeric_limits_met']=all(r['center_numeric_limits_met'] for r in report['ranks'])
    out=Path(a.out); out.mkdir(parents=True,exist_ok=False)
    write_new(out/'validation.json',report)
    lines.append('Center numeric limits met: '+str(report['center_numeric_limits_met']))
    lines.append('Dual-card restoration verified: True; Eq2 parameters validated: False')
    (out/'REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines)); print('Saved:',out)
    if not report['center_numeric_limits_met']: raise SystemExit(2)


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    f=sub.add_parser('freeze')
    for name in ('audit','source','training-extra','out'): f.add_argument('--'+name,required=True)
    c=sub.add_parser('check')
    c.add_argument('--replacement-2527',help='Use a separately restored fresh window instead of duplicated original 2527 window')
    for name in ('frozen','collection','out'): c.add_argument('--'+name,required=True)
    a=p.parse_args(); (freeze if a.command=='freeze' else check)(a)


if __name__=='__main__': main()
