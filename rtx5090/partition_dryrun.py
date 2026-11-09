"""Offline GPU-pair selection/Algorithm 1 trace. Never a deployment plan."""
import argparse
import math
from pathlib import Path
from sparsedvfs.paper_core import OrderedDAG, greedy_partition
from .eager_model import PairFrequency
from .service_core import read, write_new, positive
from .observe_core import digest


def partition(data, choices=None):
    if data.get('schema')!='pair-partition-input-v1': raise ValueError('Wrong dry-run schema')
    nodes=data['nodes']
    graph=OrderedDAG(tuple(nodes),tuple(tuple(e) for e in data['edges'])).validate()
    table=data['candidates']
    if set(table)!=set(nodes): raise ValueError('Missing/extra node table')
    rates=(2250,2527)
    missing=[]
    for node in nodes:
        if set(table[node])!={str(f) for f in rates}: raise ValueError('Missing/extra frequency candidates')
        for f in rates:
            row=table[node][str(f)]
            if row is None or row.get('pair_latency_s') is None or row.get('pair_energy_j') is None:
                missing.append(node+':'+str(f)); continue
            positive(row['pair_latency_s'],'paired latency')
            positive(row['pair_energy_j'],'paired energy')
    result=dict(schema='pair-partition-dryrun-v1',input_sha256=digest(data),
                synthetic=data.get('synthetic') is True,scope=data['scope'],
                deployable=False,clock_writes=False,N=5,similarity=.05,latency_budget=1.05,
                choices={},blocks=[],decisions=[],blockers=[],
                interpretation='GPU pair projection; no CPU/EMC control, no hardware authorization')
    switch=data.get('switch_s')
    if missing: result['blockers'].append(dict(reason='missing_unit_frequency_predictions',count=len(missing),examples=missing[:8]))
    if switch is None: result['blockers'].append(dict(reason='missing_measured_switch_bound'))
    else: positive(switch,'switch bound')
    if result['blockers']:
        result['status']='blocked_missing_inputs'; return result
    if choices is not None:
        if set(choices) != set(nodes) or any(type(f) is not int or f not in rates for f in choices.values()):
            raise ValueError('Explicit selection does not cover all nodes/rates')
        result['choices'] = dict(choices)
        result['selection_source'] = 'explicit_request_budget_selector'
    else:
        result['selection_source'] = 'legacy_per_unit_5pct_screen_not_request_budget'
        for node in nodes:
            feasible=[f for f in rates if table[node][str(f)]['pair_latency_s']<=1.05*table[node]['2527']['pair_latency_s']]
            result['choices'][node]=min(feasible,key=lambda f:(table[node][str(f)]['pair_energy_j'],table[node][str(f)]['pair_latency_s'],f))
    def duration(members,frequency):
        return sum(table[n][str(frequency.values[0])]['pair_latency_s'] for n in members)
    blocks,decisions=greedy_partition(graph,lambda n:(result['choices'][n],)*2,duration,
        lambda a,b:all(abs(x-y)<=.05*max(x,y) for x,y in zip(a.values,b.values)),switch,5,
        frequency_factory=PairFrequency.of)
    if [n for b in blocks for n in b['nodes']]!=list(nodes): raise ValueError('Partition coverage/order changed')
    for index,block in enumerate(blocks):
        block['boundary_sequence']=index
        block['rank_0_nodes']=block['nodes']; block['rank_1_nodes']=block['nodes']
        block['model_energy_j']=sum(table[n][str(block['triplet'][0])]['pair_energy_j'] for n in block['nodes'])
    transition_count=sum(a['triplet']!=b['triplet'] for a,b in zip(blocks,blocks[1:]))
    compute=sum(b['estimated_s'] for b in blocks)
    baseline=sum(table[n]['2527']['pair_latency_s'] for n in nodes)
    result.update(status='offline_partition_complete',blocks=blocks,decisions=decisions,
        eq3_all_blocks_passed=all(b['amortized_strict_gt'] for b in blocks),
        tail_short_or_equal=not blocks[-1]['amortized_strict_gt'],
        total_model_energy_j=sum(b['model_energy_j'] for b in blocks),
        transition_count=transition_count,compute_s=compute,
        modeled_serial_latency_s=compute+transition_count*switch,
        latency_budget_met_before_unmodeled_overheads=compute+transition_count*switch<=1.05*baseline,
        unmodeled_costs=['initial_setting','final_restore','dual_rank_coordination','readback_wait',
                         'switch_energy','HTTP_and_scheduler'],
        source='literal_Algorithm1_with_separate_strict_Eq3_audit')
    return result


def demo():
    # A deliberate low->high->low sequence including a short tail. Software only.
    nodes=['a','b','c','tail']
    table={}
    for n,t,low in [('a',.002,True),('b',.020,False),('c',.010,True),('tail',.001,False)]:
        table[n]={'2250':dict(pair_latency_s=t*(1.02 if low else 1.2),pair_energy_j=t*100),
                  '2527':dict(pair_latency_s=t,pair_energy_j=t*150)}
    return dict(schema='pair-partition-input-v1',synthetic=True,scope='synthetic_software_fixture',
                nodes=nodes,edges=list(zip(nodes,nodes[1:])),candidates=table,switch_s=.001)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True); g.add_argument('--input'); g.add_argument('--demo',action='store_true')
    p.add_argument('--out',required=True); a=p.parse_args()
    value=partition(demo() if a.demo else read(a.input))
    write_new(a.out,value)
    print(value['status']); print('blocks:',len(value['blocks']),'deployable:',False)
    if value['blockers']: print('blockers:',value['blockers'])
    print('Saved:',a.out)


if __name__=='__main__': main()
