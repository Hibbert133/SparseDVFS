"""Pure offline Algorithm 1 adapter: frequencies and time, never energy selection."""
import argparse
from sparsedvfs.paper_core import OrderedDAG, greedy_partition, relative_frequency_similarity
from .eager_model import PairFrequency
from .service_core import read, write_new, positive
from .observe_core import digest


def partition_predictions(graph, predict_frequency, predict_block_time, switch_s,
                          *, factor=5, similarity=.05, scope='operator_predictions'):
    """No energy callback or optimizer; use the given upstream predictions unchanged."""
    graph.validate()
    blocks, decisions=greedy_partition(graph,predict_frequency,predict_block_time,
        relative_frequency_similarity(similarity),switch_s,factor,
        frequency_factory=PairFrequency.of,compact_decisions=True)
    if [n for b in blocks for n in b['nodes']]!=list(graph.nodes):
        raise ValueError('Partition coverage/order changed')
    for i,block in enumerate(blocks):
        block.update(boundary_sequence=i,rank_0_nodes=block['nodes'],rank_1_nodes=block['nodes'])
    return dict(schema='pair-partition-dryrun-v2',status='offline_partition_complete',
        scope=scope,deployable=False,clock_writes=False,N=factor,similarity=similarity,
        blocks=blocks,decisions=decisions,blockers=[],
        eq3_all_blocks_passed=all(b['amortized_strict_gt'] for b in blocks),
        tail_short_or_equal=not blocks[-1]['amortized_strict_gt'],
        transition_count=sum(a['triplet']!=b['triplet'] for a,b in zip(blocks,blocks[1:])),
        source='literal_Algorithm1_no_energy_input_no_local_latency_constraint')


def partition(data, choices=None):
    if data.get('schema') not in ('pair-partition-input-v1','pair-partition-input-v2'):
        raise ValueError('Wrong partition input schema')
    nodes=data['nodes'];graph=OrderedDAG(tuple(nodes),tuple(tuple(e) for e in data['edges'])).validate()
    table=data['candidates'];rates=(2250,2527)
    if set(table)!=set(nodes):raise ValueError('Missing/extra node table')
    missing=[]
    for n in nodes:
        if set(table[n])!={str(f) for f in rates}:raise ValueError('Missing/extra frequency candidates')
        for f in rates:
            row=table[n][str(f)]
            if row is None or row.get('pair_latency_s') is None:missing.append(n+':'+str(f))
            else:positive(row['pair_latency_s'],'predicted latency')
    blockers=[]
    if missing:blockers.append(dict(reason='missing_unit_time_predictions',count=len(missing),examples=missing[:8]))
    switch=data.get('switch_s')
    if switch is None:blockers.append(dict(reason='missing_switch_time'))
    else:positive(switch,'switch time')
    choices=data.get('selected_frequencies') if choices is None else choices
    if choices is None:blockers.append(dict(reason='missing_upstream_frequency_predictions'))
    elif set(choices)!=set(nodes) or any(type(f) is not int or f not in rates for f in choices.values()):
        raise ValueError('Upstream selection must cover all nodes with supported frequencies')
    timing=data.get('timing_model','legacy_serial_projection')
    if timing not in ('legacy_serial_projection','verified_serial','resource_dag'):
        raise ValueError('Unknown block timing model')
    if timing in ('verified_serial','resource_dag') and not data.get('dependency_evidence_sha256'):
        raise ValueError('Verified timing requires dependency/resource evidence')
    if timing=='verified_serial' and not data.get('serial_nonoverlap_verified'):
        raise ValueError('Serial addition requires no-overlap evidence')
    if data['schema']=='pair-partition-input-v2' and timing=='legacy_serial_projection':
        blockers.append(dict(reason='missing_block_timing_contract'))
    base=dict(input_sha256=digest(data),synthetic=data.get('synthetic') is True,
              timing_model=timing,switch_bound_verified=data.get('switch_bound_verified') is True)
    if blockers:
        return dict(base,schema='pair-partition-dryrun-v2',status='blocked_missing_inputs',
                    deployable=False,clock_writes=False,blocks=[],decisions=[],choices={},blockers=blockers)
    predecessors={n:[] for n in nodes}
    for a,b in graph.edges:predecessors[b].append(a)
    def duration(members,freq):
        f=str(freq.values[0])
        if timing!='resource_dag':return sum(table[n][f]['pair_latency_s'] for n in members)
        # Resource serialization and collective dependencies must be included in edges.
        finish={}
        for n in members:
            finish[n]=table[n][f]['pair_latency_s']+max((finish[p] for p in predecessors[n] if p in finish),default=0)
        return max(finish.values())
    result=partition_predictions(graph,lambda n:(choices[n],)*2,duration,switch,
        factor=data.get('N',5),similarity=data.get('similarity',.05),scope=data['scope'])
    compute=sum(b['estimated_s'] for b in result['blocks'])
    result.update(base,choices=dict(choices),selection_source='upstream_only',compute_s=compute,
                  modeled_serial_latency_s=compute+result['transition_count']*switch,
                  request_latency_budget_verified=False,
                  unmodeled_costs=['initial_setting','restore','coordination','uncovered_request_work'])
    return result


def demo(include_energy=False):
    nodes=['a','b','c','tail'];table={}
    for n,t,low in [('a',.002,True),('b',.020,False),('c',.010,True),('tail',.001,False)]:
        table[n]={'2250':dict(pair_latency_s=t*(1.02 if low else 1.2)),
                  '2527':dict(pair_latency_s=t)}
        if include_energy:
            table[n]['2250']['pair_energy_j']=t*100
            table[n]['2527']['pair_energy_j']=t*150
    return dict(schema='pair-partition-input-v1',synthetic=True,scope='synthetic_software_fixture',
        nodes=nodes,edges=list(zip(nodes,nodes[1:])),candidates=table,switch_s=.001,
        selected_frequencies=dict(a=2250,b=2527,c=2250,tail=2527))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True);g.add_argument('--input');g.add_argument('--demo',action='store_true')
    p.add_argument('--out',required=True);a=p.parse_args()
    result=partition(demo() if a.demo else read(a.input));write_new(a.out,result)
    print(result['status'],'blocks:',len(result['blocks']),'deployable:',False)
    if result['blockers']:print('blockers:',result['blockers'])
    print('Saved:',a.out)


if __name__=='__main__':main()
