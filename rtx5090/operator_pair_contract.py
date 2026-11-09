"""Offline dual-rank sequence contract for a later no-write boundary experiment."""
import argparse
from collections import defaultdict
from pathlib import Path
from .service_core import read,write_new
from .observe_core import GPU_UUIDS,digest
from .operator_parameter_gaps import normalize,differences
from .linear_frequency_probe import compare

CONTEXTS=('prefill','decode.0','decode.31','decode.62')

def audit(data):
    if data.get('status')!='two_setting_operators_timing_complete' or not data.get('restoration',{}).get('verified'):
        raise ValueError('Complete operator collection and verified restoration required')
    compare(data['windows']) # all six windows, exact per-rank contracts retained
    reference=data['windows'][0]['operators'];groups=[defaultdict(dict),defaultdict(dict)]
    for key,row in reference.items():
        rank_text,tail=key.split(':',1);context,index=tail.split('/')
        rank=int(rank_text);index=int(index)
        if rank not in (0,1) or context not in CONTEXTS or row['uuid']!=GPU_UUIDS[rank]:
            raise ValueError('Unexpected rank/context/UUID')
        if index in groups[rank][context]:raise ValueError('Duplicate position')
        groups[rank][context][index]=(key,row)
    mismatches=[];summary=[];candidates=[]
    for context in CONTEXTS:
        left,right=groups[0][context],groups[1][context]
        if not left or set(left)!=set(right) or sorted(left)!=list(range(len(left))):
            raise ValueError('Missing/reordered rank positions: '+context)
        communication=[];candidate=None
        for index in sorted(left):
            ka,a=left[index];kb,b=right[index]
            # shape here is dispatcher metadata, not a tensor shape list.
            da=dict(parent=a['parent'],contract=normalize(a['shape'],False))
            db=dict(parent=b['parent'],contract=normalize(b['shape'],False))
            delta=differences(da,db)
            if delta:mismatches.append(dict(context=context,index=index,differences=delta))
            op=a['shape']['overload']
            if 'all_reduce' in op or 'all_gather' in op:
                communication.append(dict(index=index,operator=op,matched=not delta))
            if (candidate is None and not delta and a.get('layer')==1 and
                op=='aten.linear.default' and a['parent'].endswith('.mlp.down_proj')):
                candidate=dict(context=context,before_keys=[ka,kb],
                    source_ordinals=[a['source_ordinal'],b['source_ordinal']],
                    contract_sha256=[digest(a['shape']),digest(b['shape'])],
                    parent=a['parent'],operator=op,
                    status='candidate_only_requires_device_completion_and_dual_rank_rendezvous_test')
        summary.append(dict(context=context,pair_calls=len(left),communication=communication))
        if candidate:candidates.append(candidate)
    return dict(status='paired_dispatch_contract_matched' if not mismatches else 'rank_contract_mismatch',
        identity_fingerprint=data['identity_fingerprint'],source_sha256=digest(data),
        contexts=summary,mismatches=mismatches,candidates=[] if mismatches else candidates,
        clock_writes=False,deployable=False,graph_join_verified=False,
        runtime_boundary_verified=False,
        scope='transformer_layer_dispatch_sequence_only_not_kernel_DAG_or_collective_completion')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--collection',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();result=audit(read(a.collection));write_new(a.out,result)
    print(result['status'])
    for c in result['contexts']:
        print(c['context'],'paired calls:',c['pair_calls'],'communication calls:',len(c['communication']))
    print('candidate boundaries:',len(result['candidates']),'mismatches:',len(result['mismatches']))
    print('runtime_boundary_verified: False; no service access or clock writes')
    print('Saved:',a.out)

if __name__=='__main__':main()
