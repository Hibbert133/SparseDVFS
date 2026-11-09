"""Offline resource-to-Qwen transfer diagnostics; no fit, service access or writes.

Historical labels are retrospective checks, not prospective independent holdout.
Never converts repeated-primitive energy to measured Qwen operator energy.
"""
import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from .observe_core import digest, GPU_UUIDS
from .service_core import read, write_new, errors
from .operator_parameter_gaps import linear_work, elementwise_work
from .operator_memory_work import dense_tensor


def binding(meta,rank):
    op=meta.get('overload','');inputs=meta.get('inputs',{});args=inputs.get('args',[])
    family={'aten.linear.default':'linear','aten.add.Tensor':'add','aten.mul.Tensor':'mul'}.get(op)
    if not family:
        reason=('requires_paired_communication_model' if any(s in op for s in ('all_reduce','all_gather')) else
                'requires_state_and_fused_path_evidence' if any(s in op for s in ('attention','cache','gdn')) else
                'conversion_copy_alias_or_other_path_not_bound')
        return dict(family=None,work=None,blockers=[reason])
    tensors=args[:2]+[meta.get('outputs')]
    if len(tensors)!=3 or any(not dense_tensor(t) or t.get('dtype')!='torch.bfloat16'
                              or t.get('device')!=f'cuda:{rank}' for t in tensors):
        return dict(family=family,work=None,blockers=['dtype_layout_or_rank_device_mismatch'])
    if family=='linear' and inputs.get('kwargs'):
        return dict(family=family,work=None,blockers=['linear_kwargs_not_audited'])
    work=linear_work(meta) if family=='linear' else elementwise_work(meta)
    if work is None:return dict(family=family,work=None,blockers=['workload_definition_unresolved'])
    blockers=['kernel_equivalence_unverified','cache_state_unverified','primitive_to_single_call_timing_scope']
    if family!='linear' and tensors[0]['shape']!=tensors[1]['shape']:
        blockers.append('broadcast_not_in_primitive_calibration')
    return dict(family=family,work=work,blockers=blockers)


def prepare(collection):
    if collection.get('status')!='two_setting_operators_timing_complete':
        raise ValueError('Require complete operator collection, not module or linear summaries')
    if collection.get('restoration',{}).get('verified') is not True:
        raise ValueError('Source restoration unverified')
    windows=collection['windows']
    if len(windows)!=6 or Counter(w['requested_mhz'] for w in windows)!={2250:3,2527:3}:
        raise ValueError('Require all six prescribed windows')
    keys=set(windows[0]['operators'])
    if not keys or any(set(w['operators'])!=keys for w in windows):raise ValueError('Missing or extra occurrences')
    records=[];counts=Counter()
    for key in sorted(keys):
        rank=int(key.split(':',1)[0]);context=key.split(':',1)[1].split('/',1)[0]
        if rank not in (0,1):raise ValueError('Invalid rank')
        source=windows[0]['operators'][key]
        if not isinstance(source['shape'],dict):raise ValueError('Dispatcher contract required')
        match=binding(source['shape'],rank)
        for window in windows:
            row=window['operators'][key]
            if row['uuid']!=GPU_UUIDS[rank] or any(row[k]!=source[k] for k in ('shape','parent')):
                raise ValueError('Rank/shape/parent changed across source windows')
        observations={}
        for rate in (2250,2527):
            group=[w for w in windows if w['requested_mhz']==rate]
            labels=[w['operators'][key]['median_s'] for w in group]
            if any(type(t) not in (int,float) or not math.isfinite(t) or t<0 for t in labels):raise ValueError('Invalid event label')
            clocks=[w['actual_readbacks'][GPU_UUIDS[rank]] for w in group]
            if any(not 0<c['min_mhz']<=c['median_mhz']<=c['max_mhz'] for c in clocks):raise ValueError('Invalid clock range')
            observations[str(rate)]=dict(labels_s=labels,median_s=median(labels),
                frequency_hz=median(c['median_mhz'] for c in clocks)*1e6,
                frequency_range_hz=[min(c['min_mhz'] for c in clocks)*1e6,max(c['max_mhz'] for c in clocks)*1e6],
                frequency_scope='loaded_request_readback_not_per_operator_clock')
        counts[match['family'] or 'unmapped']+=1
        records.append(dict(key=key,rank=rank,context=context,parent=source['parent'],
            overload=source['shape']['overload'],contract=source['shape'],contract_sha256=digest(source['shape']),
            **match,observations=observations))
    return dict(schema='qwen-resource-map-v1',source_sha256=digest(collection),
        source_identity_fingerprint=collection['identity_fingerprint'],records=records,counts=dict(counts),
        status='qwen_resource_mapping_prepared',deployable=False,clock_writes=False,
        power_transfer='blocked_no_Qwen_operator_power_labels',
        scope='historical_transformer_dispatcher_occurrences_not_full_request_DAG')


def evaluate(mapping,model):
    if mapping['schema']!='qwen-resource-map-v1' or model['schema']!='resource-model-v1':raise ValueError('Wrong schema')
    scores=defaultdict(lambda:[[],[]]);rows=[];blocked=Counter();total=0
    for op in mapping['records']:
        if digest(op['contract'])!=op['contract_sha256'] or binding(op['contract'],op['rank'])['work']!=op['work']:
            raise ValueError('Mapping contract or work changed')
        m=model['models'].get(f"{op['rank']}:{op['family']}")
        for rate,obs in op['observations'].items():
            total+=1;reasons=list(op['blockers'])
            row=dict(key=op['key'],requested_mhz=int(rate),predicted_latency_s=None,
                     observed_latency_s=obs['median_s'],predicted_power_w=None,predicted_energy_j=None,
                     blockers=reasons)
            rows.append(row)
            if mapping['source_identity_fingerprint']!=model['identity_fingerprint']:
                reasons.append('historical_service_identity_differs')
            if not m or not op['work']:
                reasons.append('resource_model_or_work_missing');blocked.update(reasons);continue
            t=m['latency'];reasons.extend(t.get('blockers',[]))
            shape=([op['work'][k] for k in ('M','N','K')] if op['family']=='linear'
                   else op['contract']['outputs']['shape'])
            if str(shape) not in m['training_shapes']:reasons.append('shape_transfer_requires_validation')
            if t['status']!='candidate':
                reasons.append('resource_latency_parameters_unidentified');blocked.update(reasons);continue
            f=obs['frequency_hz'];lo,hi=m['frequency_range_hz']
            if not lo<=f<=hi:
                reasons.append('frequency_extrapolation_refused');blocked.update(reasons);continue
            if obs['frequency_range_hz'][0]<lo or obs['frequency_range_hz'][1]>hi:
                reasons.append('observed_frequency_range_leaves_calibration')
            work=op['work']
            predicted=t['overhead_s']+max(t['a_s_cycles_per_flop']*work['work_flop']/f,t['b_s_per_byte']*work['data_byte'])
            if not math.isfinite(predicted) or predicted<=0:raise ValueError('Nonphysical Eq1 prediction')
            row['predicted_latency_s']=predicted
            if obs['median_s']<=0 or any(v<=0 for v in obs['labels_s']):
                reasons.append('zero_event_label_unresolved');blocked.update(reasons);continue
            row['absolute_relative_error']=abs(predicted-obs['median_s'])/obs['median_s']
            group=f"{op['rank']}:{op['context']}:{op['family']}:{rate}"
            scores[group][0].extend(obs['labels_s']);scores[group][1].extend([predicted]*len(obs['labels_s']))
            blocked.update(reasons)
    summaries={k:errors(*v) for k,v in scores.items()}
    return dict(status='retrospective_Qwen_latency_transfer_complete',model_sha256=digest(model),
        mapping_sha256=digest(mapping),rows=rows,scores=summaries,total_operator_settings=total,
        scored_labels=sum(v['n'] for v in summaries.values()),blocker_counts=dict(blocked),
        parameters_updated=False,independent_holdout=False,parameters_validated=False,deployable=False,
        power_and_energy_validation='unavailable_no_operator_labels',
        numeric_limits_for_scored_groups=bool(summaries) and all(s['wape']<=.1 and s['p90_relative_error']<=.2 for s in summaries.values()),
        note='Scores compare historical single-call event envelopes to repeated-resource Eq1; not same-path certification')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','validate'])
    p.add_argument('--input',required=True);p.add_argument('--model');p.add_argument('--out',required=True)
    a=p.parse_args()
    if a.command=='validate' and not a.model:p.error('--model required')
    result=prepare(read(a.input)) if a.command=='prepare' else evaluate(read(a.input),read(a.model))
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    write_new(out/('mapping.json' if a.command=='prepare' else 'validation.json'),result)
    print(result['status'])
    for name in ('counts','scores','blocker_counts','numeric_limits_for_scored_groups'):
        if name in result:print(name+':',result[name])
    print('parameters_updated=False deployable=False');print('Saved:',out)

if __name__=='__main__':main()
