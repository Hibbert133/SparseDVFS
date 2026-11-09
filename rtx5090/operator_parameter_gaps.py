"""Offline exact-signature reuse inventory and deduplicated calibration tasks."""
import argparse
from collections import Counter,defaultdict
import math
from pathlib import Path
from .observe_core import GPU_UUIDS,digest
from .service_core import read,write_new
from .operator_equations import audit
from .eager_matrix_replay import case_spec
from .operator_memory_work import memory_work


def normalize(value, root=True):
    if isinstance(value, dict):
        return {k: ('cuda:rank' if value.get('tensor') is True and k == 'device'
                    and v in ('cuda:0', 'cuda:1') else normalize(v, False))
                for k, v in value.items() if not (root and k in ('rank', 'kernel_names'))}
    if isinstance(value, list): return [normalize(v, False) for v in value]
    return value


def differences(a, b, path='$'):
    if type(a) is not type(b): return [dict(path=path, rank0=a, rank1=b)]
    if isinstance(a, dict):
        rows=[]
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                rows.append(dict(path=path+'.'+key, rank0=a.get(key), rank1=b.get(key), missing=True))
            else: rows.extend(differences(a[key], b[key], path+'.'+key))
        return rows
    if isinstance(a, list):
        if len(a)!=len(b): return [dict(path=path+'.length', rank0=len(a), rank1=len(b))]
        return [row for i,(x,y) in enumerate(zip(a,b)) for row in differences(x,y,f'{path}[{i}]')]
    return [] if a==b else [dict(path=path, rank0=a, rank1=b)]


def embedding_shard_pair(left, right, delta):
    """Recognize only the reviewed out-first TP original-vocabulary ABI.

    This is a structural contract, not evidence of token values or all-reduce.
    """
    op='_C.vocab_parallel_embedding.default'
    if left.get('overload')!=op or right.get('overload')!=op: return None
    allowed={'$.signature.inputs.args[3]', '$.signature.inputs.args[4]'}
    if not delta or any(d['path'] not in allowed for d in delta): return None
    rows=[]
    for meta in (left,right):
        inputs=meta.get('inputs',{});args=inputs.get('args',[])
        if len(args)!=8 or inputs.get('kwargs'): return None
        out,ids,weight=args[:3]
        if not all(isinstance(t,dict) and t.get('tensor') is True for t in (out,ids,weight)): return None
        shapes=[t.get('shape') for t in (out,ids,weight)]
        if not all(isinstance(sh,list) and all(type(d) is int and d>0 for d in sh) for sh in shapes): return None
        os,ins,ws=shapes
        if len(os)!=2 or len(ins)!=1 or len(ws)!=2 or os!=[ins[0],ws[1]]: return None
        if ids.get('dtype') not in ('torch.int32','torch.int64'): return None
        if out.get('dtype')!=weight.get('dtype'): return None
        if any(type(v) is not int or v<0 for v in args[3:]): return None
        start,end,padding,added_start,added_end=args[3:]
        if end<=start or added_start!=added_end: return None  # added vocab needs its own reviewed contract
        if end-start+padding>ws[0]: return None
        rows.append(dict(original_range=[start,end],padding_rows=padding,
                         added_range=[added_start,added_end],weight_shape=ws))
    if rows[0]['original_range'][0]!=0 or rows[0]['original_range'][1]!=rows[1]['original_range'][0]: return None
    return dict(rule='vllm_out_first_original_vocab_tp2_v1',ranks=rows,
                scope='structural_shard_arguments_only',collective_verified=False,
                source='user_supplied_installed_vllm_custom_ops_wrapper',
                signature_sha256=[digest(left),digest(right)])


def elementwise_work(meta):
    op=meta.get('overload')
    if op not in ('aten.add.Tensor', 'aten.sub.Tensor', 'aten.mul.Tensor'): return None
    args=meta.get('inputs',{}).get('args',[])
    kwargs=meta.get('inputs',{}).get('kwargs',{})
    if len(args)!=2 or set(kwargs)-{'alpha'}: return None
    if kwargs.get('alpha',1)!=1 or (op=='aten.mul.Tensor' and kwargs): return None
    tensors=args+[meta.get('outputs')]
    widths={'torch.bfloat16':2,'torch.float16':2,'torch.float32':4,'torch.float64':8}
    for t in tensors:
        if not isinstance(t,dict) or t.get('tensor') is not True or t.get('dtype') not in widths: return None
        if not isinstance(t.get('shape'),list) or any(type(d) is not int or d<0 for d in t['shape']): return None
        if t.get('layout')!='torch.strided': return None
    a,b,out=tensors
    dims=max(len(a['shape']),len(b['shape']))
    shape=[]
    for x,y in zip([1]*(dims-len(a['shape']))+a['shape'],[1]*(dims-len(b['shape']))+b['shape']):
        if x!=y and x!=1 and y!=1: return None
        shape.append(y if x==1 else x)
    if out['shape']!=shape: return None
    return dict(work_flop=math.prod(shape),
                data_byte=sum(math.prod(t['shape'])*widths[t['dtype']] for t in tensors))


def work_class(node):
    op=node['operator']
    if 'all_reduce' in op or 'all_gather' in op: return 'collective'
    if 'attention' in op or 'cache' in op: return 'stateful_attention_or_cache'
    if not node['kernels']: return 'no_kernel_unresolved_not_zero_cost'
    if op=='aten.linear.default': return 'linear'
    if op in ('aten.add.Tensor','aten.sub.Tensor','aten.mul.Tensor'): return 'floating_arithmetic_candidate'
    if 'copy_' in op or '.to.' in op: return 'copy_or_conversion'
    return 'other_kernel_family'


def linear_work(meta):
    if meta.get('overload')!='aten.linear.default':return None
    args=meta.get('inputs',{}).get('args',[]);output=meta.get('outputs',{})
    if len(args) not in (2,3) or (len(args)==3 and args[2] is not None):return None
    a,b=args[:2]
    if not all(isinstance(x,dict) and x.get('tensor') and x.get('dtype')=='torch.bfloat16' for x in (a,b,output)):
        return None
    shape=a['shape'];weight=b['shape']
    if not shape or len(weight)!=2 or shape[-1]!=weight[1] or output['shape']!=shape[:-1]+[weight[0]]:
        return None
    if any(type(x) is not int or x<=0 for x in shape+weight):return None
    m=math.prod(shape[:-1]);n,k=weight
    return dict(work_flop=2*m*n*k,data_byte=2*(m*k+n*k+m*n),M=m,N=n,K=k)


def build(inventories,linear=None):
    if len(inventories)!=2:raise ValueError('Two rank inventories required')
    for rank,inv in enumerate(inventories):
        if inv['rank']!=rank or inv['identity']['gpu_uuid']!=GPU_UUIDS[rank]:raise ValueError('Inventory UUID mismatch')
        ids=[n['id'] for n in inv['nodes']]
        if len(ids)!=len(set(ids)):raise ValueError('Duplicate source occurrences')
    if linear and (linear.get('status')!='two_setting_linear_timing_complete' or not linear.get('restoration',{}).get('verified')):
        raise ValueError('Historical linear collection is incomplete or restoration unverified')
    bundle=dict(schema='operator-equations-v1',scope='historical_operator_inventory_contexts_only',
        identity_fingerprint=None,current_identity_verified=False,rank_join_verified=False,
        rank_join_evidence_sha256=None,
        units_contract=dict(time='s',work='FLOP',data='byte',frequency='Hz',voltage='V',power='W'),
        operators=[],families={},thermal={},states={},switch_s=None,switch_bound_verified=False,
        graph=dict(edges=None,timing_model=None,dependency_evidence_sha256=None),
        source_inventory_sha256=[digest(i) for i in inventories])
    tasks={};source_rows={};prior_matches=[]
    for rank,inv in enumerate(inventories):
        for node in inv['nodes']:
            meta=inv['signatures'][node['signature_id']]
            grouping=dict(rank=rank,phase=node['context'].split('.')[0],operator=node['operator'],
                          kernels=sorted(node['kernels']))
            # Grouping proposes capacity reuse; exact signature validation is still required.
            family_id=f"rank{rank}-"+digest(grouping)[:20]
            if family_id not in tasks:
                family=dict(rank=rank,signature_ids=[],same_path_verified=False,path_evidence_sha256=None,
                    independent_capacity_calibration=False,capacity_evidence_sha256=None,
                    alpha=None,overhead_s=None,power_scope=None,paired_window_verified=False,
                    power_evidence_sha256=None,validation=None,
                    capacities={str(f):dict(peak_flops=None,bandwidth_Bps=None,frequency_range_hz=None) for f in (2250,2527)})
                bundle['families'][family_id]=family
                op=node['operator']
                policy=('paired_ranks_only' if 'all_reduce' in op or 'all_gather' in op else
                        'original_once_or_verified_written_state_restore' if 'attention' in op or 'cache' in op else
                        'classify_metadata_copy_or_kernel_first' if not node['kernels'] else
                        'original_once_until_replay_adapter_validated')
                tasks[family_id]=dict(id=family_id,**grouping,occurrences=0,signature_ids=[],
                    representative=node['id'],representative_context=node['context'],
                    representative_parent=node['parent_label'],representative_layer=node['layer'],
                    work_class=work_class(node),replay_policy=policy,control_settings=[2250,2527],reuse_validated=False,
                    required=['path_state_kernel_evidence','independent_capacity_bandwidth_overhead',
                              'activity_alpha_at_effective_sparsity_zero','independent_holdout'],
                    static_work_definitions_missing=0)
            family=bundle['families'][family_id];task=tasks[family_id]
            if node['signature_id'] not in family['signature_ids']:
                family['signature_ids'].append(node['signature_id']);task['signature_ids'].append(node['signature_id'])
            task['occurrences']+=1
            priority=lambda layer: {1:0,3:1,0:2}.get(layer,3)
            if priority(node['layer'])<priority(task['representative_layer']):
                task.update(representative=node['id'],representative_context=node['context'],
                            representative_parent=node['parent_label'],representative_layer=node['layer'])
            work=linear_work(meta) or elementwise_work(meta) or memory_work(meta,node['kernels'])
            if work is None or work['work_flop'] is None:task['static_work_definitions_missing']+=1
            row=dict(rank=rank,uuid=GPU_UUIDS[rank],signature_id=node['signature_id'],family_id=family_id,
                work_flop=work['work_flop'] if work else None,data_byte=work['data_byte'] if work else None,
                workload_provenance=(inv.get('source_sha256') or digest(inv)) if work else None,
                data_semantics='logical_tensor_bytes_not_measured_DRAM' if work else None,s_comp=0,s_mem=0,
                workload_derivation=work.get('derivation', 'linear_2MNK_v1' if meta.get('overload')=='aten.linear.default' else 'floating_binary_broadcast_v1') if work else None,
                unresolved_workload=work.get('unresolved',[]) if work else ['workload_definition'])
            source_rows[(rank,node['id'])]=row
            if linear and work:
                try:
                    spec=case_spec(node['parent_label'],node['context'].split('.')[0],node['operator'],
                                   meta['inputs']['args'],meta['outputs'],rank)
                except (ValueError,KeyError,TypeError):spec=None
                if spec:
                    name=f"{rank}:{spec['phase']}/{spec['module']}"
                    rows=[w['operators'].get(name) for w in linear['windows']]
                    if rows and all(r and r['uuid']==GPU_UUIDS[rank] and r['shape']==[spec['M'],spec['N'],spec['K']] for r in rows):
                        prior_matches.append(dict(node=node['id'],rank=rank,linear_case=name,
                            use='historical_repeated_linear_diagnostic_only',parameters_authorized=False))
    by_context=[defaultdict(list),defaultdict(list)]
    for r,inv in enumerate(inventories):
        for node in inv['nodes']:by_context[r][node['context']].append(node)
    mismatches=[]; mismatch_details=[]; accepted_shards=[]
    if set(by_context[0])!=set(by_context[1]):raise ValueError('Rank contexts differ')
    for context,left in by_context[0].items():
        right=by_context[1][context]
        if len(left)!=len(right):raise ValueError('Rank occurrence counts differ: '+context)
        for index,(a,b) in enumerate(zip(left,right)):
            ma=inventories[0]['signatures'][a['signature_id']];mb=inventories[1]['signatures'][b['signature_id']]
            unit_id=f'{context}/{index:06d}'
            delta=differences(dict(parent=a['parent_label'],signature=normalize(ma)),
                              dict(parent=b['parent_label'],signature=normalize(mb)))
            shard=embedding_shard_pair(ma,mb,delta)
            if shard:
                accepted_shards.append(dict(id=unit_id,source_nodes=[a['id'],b['id']],
                    original_differences=delta,**shard))
            if delta and not shard:
                mismatches.append(unit_id)
                mismatch_details.append(dict(id=unit_id,source_nodes=[a['id'],b['id']],
                    operators=[a['operator'],b['operator']],differences=delta))
            bundle['operators'].append(dict(id=unit_id,context=context,
                source_nodes=[a['id'],b['id']],ranks=[source_rows[(0,a['id'])],source_rows[(1,b['id'])]]))
    for r in (0,1):
        bundle['thermal'][str(r)]=dict(k1=None,k2=None,identifiability_passed=False,
                                       evidence_sha256=None,temperature_range_c=None,voltage_range_v=None)
        bundle['states'][str(r)]={str(f):dict(effective_frequency_hz=None,voltage_v=None,temperature_c=None,
            known_before_prediction=False,evidence_sha256=None) for f in (2250,2527)}
    report=dict(status='calibration_tasks_ready_not_parameters_validated',clock_writes=False,deployable=False,
        source_inventory_sha256=bundle['source_inventory_sha256'],pair_occurrences=len(bundle['operators']),
        candidate_families=len(tasks),rank_contract_mismatches=mismatches,
        rank_mismatch_details=mismatch_details,recognized_embedding_shards=accepted_shards,
        family_class_counts=dict(Counter(t['work_class'] for t in tasks.values())),
        static_work_coverage=dict(rank_occurrences=len(source_rows),
            derived=sum(r['work_flop'] is not None for r in source_rows.values()),
            missing=sum(r['work_flop'] is None for r in source_rows.values()),
            logical_bytes_known=sum(r['data_byte'] is not None for r in source_rows.values()),
            bytes_only=sum(r['data_byte'] is not None and r['work_flop'] is None for r in source_rows.values())),
        workload_derivation_counts=dict(Counter(r['workload_derivation'] or 'unresolved' for r in source_rows.values())),
        linear_diagnostic_matches=len(prior_matches),linear_matches=prior_matches,
        linear_source_sha256=digest(linear) if linear else None,
        tasks=list(tasks.values()),missing_evidence_counts=dict(Counter(x['reason'] for x in audit(bundle))),
        shared_tasks=[dict(rank=r,uuid=GPU_UUIDS[r],required=['identifiable_shared_k1_k2',
            'pre_prediction_voltage_frequency_temperature_mapping','dual_rank_dependency_and_boundary_evidence']) for r in (0,1)],
        note='Family grouping is a calibration proposal, not permission to share unvalidated parameters')
    return bundle,report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inventory',required=True)
    p.add_argument('--linear');p.add_argument('--out',required=True);a=p.parse_args()
    root=Path(a.inventory);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    bundle,report=build([read(root/f'rank-{r}.json') for r in (0,1)],read(a.linear) if a.linear else None)
    write_new(out/'equations-template.json',bundle);write_new(out/'calibration-tasks.json',report)
    lines=[report['status'],f"pair occurrences: {report['pair_occurrences']}",
        f"candidate parameter families: {report['candidate_families']}",
        f"historical linear matches: {report['linear_diagnostic_matches']}",
        f"rank contract mismatches: {len(report['rank_contract_mismatches'])}",
        f"recognized embedding shard pairs: {len(report['recognized_embedding_shards'])} (structural only; graph join unverified)",
        'Static workload coverage: '+str(report['static_work_coverage']),
        'Family classes: '+str(report['family_class_counts']),
        'Workload derivations: '+str(report['workload_derivation_counts']),
        'Missing (occurrence/family/shared counts, not collection jobs): '+str(report['missing_evidence_counts']),
        'No service access, restart, new sampling or clock writes.']
    for item in report['rank_mismatch_details'][:10]:
        lines.append('Rank difference: '+str(dict(id=item['id'],operators=item['operators'],differences=item['differences'][:10])))
    (out/'REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines));print('Saved:',out)


if __name__=='__main__':main()
