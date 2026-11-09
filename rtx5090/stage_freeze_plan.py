"""Freeze offline-stage artifacts and produce a non-deployable planning audit."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
from .observe_core import digest, GPU_UUIDS
from .service_core import write_new, positive
from .eq2_frequency_holdout import require_manifest
from .partition_dryrun import partition, demo
from .eager_core import UNITS


def screen(train,holdout):
    """Whole-window descriptive screen; never allocate board energy to units."""
    base=mean(w['latency_s'] for w in train['current'])
    cells={k:dict(mean_request_s=mean(w['latency_s'] for w in rows),
                  window_energy_j=mean(w['energy_j'] for w in rows),windows=len(rows)) for k,rows in train.items()}
    feasible=[k for k in ('2527','2250') if cells[k]['mean_request_s']<=1.05*base]
    selected=min(feasible,key=lambda k:(cells[k]['window_energy_j'],cells[k]['mean_request_s'],k)) if feasible else None
    validation=[]
    baseline_time=mean(w['latency_s'] for w in holdout['current'])
    baseline_energy=mean(w['energy_j'] for w in holdout['current'])
    for condition in ('2527','2250'):
        row=holdout[condition][0]
        validation.append(dict(requested_setting_mhz=int(condition),
            latency_change=row['latency_s']/baseline_time-1,
            energy_change=row['energy_j']/baseline_energy-1,
            latency_within_5_percent=row['latency_s']<=1.05*baseline_time,
            selected_using_training=condition==selected))
    return dict(status='whole_window_retrospective_screen',cells=cells,selected_requested_setting_mhz=int(selected) if selected else None,
                holdout_comparisons=validation,selection_uses_holdout_labels=False,
                protocol_chosen_after_results_seen=True,conditions_not_request_paired=True,
                reference_policy='current_not_independently_verified_AUTO',
                exact_lock_verified=False,deployable=False,
                scope='20_request_window_only_not_sparse_operator_frequency_selection')


def run(a):
    out=Path(a.out); out.mkdir(parents=True,exist_ok=False)
    evidence={}
    def load(path):
        path=Path(path); raw=path.read_bytes()
        evidence[str(path.resolve())]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
        return json.loads(raw)
    latency,power,train,hold,repair=map(Path,(a.latency,a.power,a.training,a.holdout,a.repair))
    model=load(latency/'model/model.json')
    observation=load(latency/'train/observation.json')
    check=load(latency/'check/observation.json')
    latency_diagnostic=load(a.latency_diagnostic)
    identity=model['identity_fingerprint']
    if digest(observation)!=model['source_sha256'] or check['identity_fingerprint']!=identity:
        raise ValueError('Latency evidence provenance mismatch')
    if (latency_diagnostic['model_sha256']!=digest(model) or
        latency_diagnostic['observation_sha256']!=digest(check) or not latency_diagnostic['numeric_limits_met']):
        raise ValueError('Latency diagnostic not bound to frozen model/new request')
    coverage=observation['coverage']
    if not coverage['passed'] or coverage['units']!=UNITS or coverage['identity_fingerprint']!=identity:
        raise ValueError('Real module coverage missing')
    reference=load(power/'reference.json')
    power_model=load(power/'power-model.json'); power_validation=load(power/'power-validation.json')
    frozen=load(hold/'frozen.json'); audit=load(train/'audit/audit.json')
    validation=load(repair/'validation/validation.json')
    manifests=[load(p/'collection/collection.json') for p in (train,hold,repair)]
    if any(m['identity_fingerprint']!=identity or m['status']!='complete' or not m['restoration']['verified'] for m in manifests):
        raise ValueError('Collection identity/restoration mismatch')
    require_manifest(manifests[1],frozen); require_manifest(manifests[2],frozen,['requested-2527'])
    if any(v['identity_fingerprint']!=identity for v in (reference,power_model,frozen,audit)):
        raise ValueError('Cross-service model mixing')
    if (reference['status']!='host_envelope_reference_complete' or not power_validation['passed'] or
        power_validation['model_sha256']!=digest(power_model) or frozen['audit_sha256']!=digest(audit) or
        validation['frozen_sha256']!=digest(frozen) or validation['collection_sha256']!=digest(manifests[1]) or
        validation.get('replacement_2527_manifest_sha256')!=digest(manifests[2]) or
        not validation['center_numeric_limits_met'] or not validation['restoration_verified']):
        raise ValueError('Frozen model/holdout certificate mismatch')
    def window(folder):
        w=load(folder/'window.json'); s=load(folder/'summary.json'); req=load(folder/'requests.json')
        if w['identity_fingerprint']!=identity or w['status']!='loaded_host_sampling_passed' or w['cache_status']!='verified_miss':
            raise ValueError('Invalid request window')
        if s['status']!='valid_host_envelope' or len(w['replies'])!=20 or len(req)!=20 or len({r['prompt_hash'] for r in req})!=20:
            raise ValueError('Incomplete request set')
        for r in w['replies']:
            if r['usage']['prompt_tokens']!=512 or r['usage']['completion_tokens']!=64: raise ValueError('Usage mismatch')
            positive(r['latency_s'],'request latency')
        if set(s['energy_by_uuid_j'])!=set(GPU_UUIDS): raise ValueError('Missing board energy')
        energy=sum(positive(v,'energy') for v in s['energy_by_uuid_j'].values())
        return dict(latency_s=mean(r['latency_s'] for r in w['replies']),energy_j=energy,
                    actual_readbacks=s.get('actual_readbacks',s.get('clock_ranges')),
                    prompt_hashes=[r['prompt_hash'] for r in req])
    train_rows={'current':[window(power/('reference-'+str(i))) for i in (1,2,3)]}
    for rate in ('2527','2250'): train_rows[rate]=[window(train/'collection'/('requested-'+rate))]
    test_rows={'current':[window(power/('holdout-'+str(i))) for i in (1,2,3)],
               '2527':[window(repair/'collection/requested-2527')],
               '2250':[window(hold/'collection/requested-2250')]}
    training_hashes={h for rows in train_rows.values() for w in rows for h in w['prompt_hashes']}
    if training_hashes & {h for rows in test_rows.values() for w in rows for h in w['prompt_hashes']}:
        raise ValueError('Training/holdout prompts overlap')
    selection=screen(train_rows,test_rows)
    write_new(out/'window-selection.json',selection)
    nodes=list(coverage['units'])
    template=dict(schema='pair-partition-input-v1',synthetic=False,
        identity_fingerprint=identity,scope='observed_module_order_projection_not_kernel_DAG',
        contexts=coverage['contexts'],nodes=nodes,edges=list(zip(nodes,nodes[1:])),
        candidates={n:{'2250':None,'2527':None} for n in nodes},switch_s=None,
        note='One-context template; require separate phase/step tables, whole-block timing and measured switch evidence before real planning')
    write_new(out/'module-input-template.json',template)
    blocked=partition(template); write_new(out/'module-partition-audit.json',blocked)
    software=partition(demo()); write_new(out/'algorithm1-software-example.json',software)
    freeze=dict(schema='offline-stage-freeze-v1',identity_fingerprint=identity,evidence=evidence,
        models=dict(linear_latency=model,fixed_workload_power=power_model,eq2_candidate=frozen),
        status=dict(latency='new_request_timing_diagnostic_kernel_unverified',
                    window_power='three_holdout_windows_passed',
                    eq2_candidate='two_fresh_requested_setting_holdouts_passed_conditionally',
                    eq2_parameters_validated=False),
        deployable=False,clock_writes=False)
    write_new(out/'stage-freeze.json',freeze)
    lines=['Stage frozen; offline selection/partition audit only, no live service access.',
           'identity: '+identity,'frozen evidence files: '+str(len(evidence)),
           'whole-window setting selected from training: '+str(selection['selected_requested_setting_mhz']),
           'Retrospective unpaired screen; current reference is not a verified AUTO certificate.']
    for r in selection['holdout_comparisons']:
        lines.append('%s requested MHz: latency change %.2f%%, energy change %.2f%%, <=5%%: %s' %
            (r['requested_setting_mhz'],r['latency_change']*100,r['energy_change']*100,r['latency_within_5_percent']))
    lines += ['Real module partition: '+blocked['status'],
              'Missing: per-unit two-frequency latency/energy + measured switching bound.',
              'Algorithm 1 software example: %s blocks, Eq3 all passed=%s, tail short=%s' %
              (len(software['blocks']),software['eq3_all_blocks_passed'],software['tail_short_or_equal']),
              'No executable hardware plan emitted. Existing models and original evidence unchanged.']
    (out/'REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines)); print('Saved:',out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('latency','latency-diagnostic','power','training','holdout','repair','out'): p.add_argument('--'+n,required=True)
    run(p.parse_args())


if __name__=='__main__': main()
