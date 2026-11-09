"""Measured-event Algorithm 1 sensitivity; no Eq1/Eq2 or executable plan claim."""
import argparse
import json
from pathlib import Path
from .service_core import read,write_new
from .observe_core import digest
from .operator_pair_contract import audit,CONTEXTS
from .linear_frequency_probe import compare
from .partition_dryrun import partition_predictions
from sparsedvfs.paper_core import OrderedDAG


def timing(root,result):
    if result.get('status')!='no_write_operator_rendezvous_passed' or result.get('disarm_error'):
        raise ValueError('Successful disarmed boundary experiment required')
    rows=[]
    for rank in (0,1):
        events=[json.loads(s) for s in (root/f'rank-{rank}-boundary.jsonl').read_text().splitlines() if s.strip()]
        for context in CONTEXTS:
            group=[e for e in events if e.get('context')==context]
            values={name:[e for e in group if e['event']==name] for name in ('gpu_wait_begin','gpu_complete','released')}
            if any(len(v)!=1 for v in values.values()) or any(e['event'] in ('failed','gpu_wait_timeout') for e in group):
                raise ValueError('Incomplete or failed boundary timing')
            begin,done,end=[values[k][0]['time_ns'] for k in values]
            if not begin<=done<=end or any(e['rank']!=rank for e in group):raise ValueError('Invalid timing identity/order')
            rows.append(dict(rank=rank,context=context,gpu_wait_ms=(done-begin)/1e6,
                             rendezvous_ms=(end-done)/1e6,total_observed_ms=(end-begin)/1e6))
    return rows


def experiment(data):
    paired=audit(data)
    if paired['mismatches']:raise ValueError('Rank contracts mismatch')
    table={r['key']:r for r in compare(data['windows'])};outputs=[]
    for context in CONTEXTS:
        keys=sorted(k[2:] for k in table if k.startswith('0:'+context+'/'))
        costs={k:{f:max(table[str(rank)+':'+k]['settings'][str(f)]['median_s'] for rank in (0,1))
                  for f in (2250,2527)} for k in keys}
        if any(t<=0 for c in costs.values() for t in c.values()):raise ValueError('Zero event time unresolved')
        chosen={k:min((2250,2527),key=lambda f:(costs[k][f],f)) for k in keys}
        graph=OrderedDAG(tuple(keys),tuple(zip(keys,keys[1:])))
        # Prefix sums avoid quadratic calc_time while preserving the explicitly assumed serial projection.
        positions={k:i for i,k in enumerate(keys)};prefix={f:[0.] for f in (2250,2527)}
        for f in prefix:
            for k in keys:prefix[f].append(prefix[f][-1]+costs[k][f])
        def duration(nodes,frequency):
            f=frequency.values[0]
            return prefix[f][positions[nodes[-1]]+1]-prefix[f][positions[nodes[0]]]
        for switch_ms in (5,10,20,50):
            result=partition_predictions(graph,lambda k:(chosen[k],chosen[k]),duration,switch_ms/1000,
                scope='diagnostic_measured_events_assumed_serial_order_not_verified_DAG')
            # Explicit experiment assumptions, never fed to the live plan loader.
            result.update(context=context,assumed_switch_ms=switch_ms,
                frequency_source='fastest_measured_event_time_ties_2250_NOT_energy_optimal',
                cost_source='max_rank_event_medians_NOT_Eq1',graph_verified=False,
                parameters_validated=False,switch_bound_verified=False)
            result.pop('decisions')
            outputs.append(result)
    return outputs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('collection','boundary-run','out'):p.add_argument('--'+arg,required=True)
    p.add_argument('--diagnostic-fastest-time',action='store_true',required=True)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    data=read(a.collection);root=Path(a.boundary_run);boundary=read(root/'boundary-test.json')
    if boundary['source_collection_sha256']!=digest(data):raise ValueError('Boundary source collection differs')
    rows=timing(root,boundary);results=experiment(data)
    write_new(out/'experiment.json',dict(status='offline_diagnostic_partition_complete',clock_writes=False,
        deployable=False,source_sha256=digest(data),boundary_sha256=digest(boundary),
        boundary_timings=rows,scenarios=results,
        limitations=['no_operator_energy_selection','assumed_serial_event_sum','assumed_switch_times',
                     'four_contexts_not_full_request','arbitrary_block_boundaries_not_runtime_verified']))
    print('rank | context | GPU wait ms | rendezvous ms | observed total ms')
    for r in rows:print(r['rank'],r['context'],round(r['gpu_wait_ms'],3),round(r['rendezvous_ms'],3),round(r['total_observed_ms'],3))
    print('context | assumed switch ms | blocks | transitions | Eq3 all passed')
    for r in results:print(r['context'],r['assumed_switch_ms'],len(r['blocks']),r['transition_count'],r['eq3_all_blocks_passed'])
    print('DIAGNOSTIC ONLY: no energy-optimal selection, no verified switch bound, no deployable plan')
    print('Saved:',out)

if __name__=='__main__':main()
