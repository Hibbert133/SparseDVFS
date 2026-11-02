#!/usr/bin/env python3
"""Synthetic descriptor/virtual-event demo. No torch, CUDA, vLLM or clock I/O."""
import argparse
import json
from itertools import product
from pathlib import Path
from sparsedvfs.paper_core import (Frequency, OrderedDAG, finite, greedy_partition,
                                   minimum_model_energy, relative_frequency_similarity)
from sparsedvfs.paper_calibration import predicted_candidates, coordination_vectors
from sparsedvfs.governor_reference import GovernorReference


def fixture(effective_sparse=True):
    """Invented numerical inputs, not Thor OPPs or paper experimental data."""
    names = ['matmul_a', 'vector_work', 'matmul_b', 'short_tail']
    vectors = [list(v) for v in product((1_000_000, 2_000_000), (500_000_000, 1_000_000_000), (1_000_000_000, 2_000_000_000))]
    profile = dict(schema=4, synthetic=True, dvfs_applied=False,
                   identity=dict(backend='synthetic_paper_core_v4'), units=names,
                   reference=vectors[-1], candidates=[dict(triplet=f, operators=[
                       dict(phase='synthetic', name=name) for name in names]) for f in vectors])
    opps = {}
    for cpu, gpu, emc in vectors:
        opps[f'{cpu},{gpu},{emc}'] = dict(
            peak_flops=(4e9 if gpu == 1_000_000_000 else 1e9) * (1 if cpu == 2_000_000 else 0.9),
            bandwidth_Bps=2e9 if emc == 2_000_000_000 else 5e8,
            voltage_v=0.7 + 0.05*(cpu == 2_000_000) + 0.1*(gpu == 1_000_000_000) + 0.05*(emc == 2_000_000_000),
            effective_frequency_hz=cpu*1000 + gpu + emc)
    specs = {}
    for name in names:
        sc = 0.75 if name == 'vector_work' and effective_sparse else 0.0
        specs['synthetic/' + name] = dict(
            work_flop=8e6 if name != 'short_tail' else 512,
            data_byte=532000 if name.startswith('matmul') else 16e6 if name == 'vector_work' else 512,
            s_comp=sc, s_mem=0.0, overhead_s=20e-6 if name == 'short_tail' else 0.0,
            effective_sparsity_verified=sc > 0,
            effective_sparsity_evidence='SYNTHETIC contract only; no actual sparse kernel' if sc else None,
            activity_factor=2e-9 if name.startswith('matmul') else 0.7e-9 if name == 'vector_work' else 0.2e-9,
            activity_s_comp=sc, workload_provenance='SYNTHETIC descriptor; no arrays allocated')
    calibration = dict(schema=4, synthetic=True,
        provenance='Invented fixture values, not measurements or author parameters',
        scalar_vf_scope='SYNTHETIC triplet-to-scalar mapping; no physical voltage rail is represented',
        temperature_c=50.0, temperature_range_c=[20.0, 80.0], temperature_sensor='virtual_junction',
        device_thermal=dict(k1=0.01, k2=0.1), opps=opps, operators=specs)
    graph = dict(nodes=names, edges=list(zip(names, names[1:])))
    shapes = dict(matmul_a=dict(lhs=[16, 500], rhs=[500, 500], output=[16, 500],
                               flop_convention='2*M*K*N', dtype='synthetic_BF16'),
                  vector_work=dict(input=[4_000_000], output=[4_000_000],
                                   traffic='synthetic read/write traffic; no data allocation'),
                  matmul_b='same descriptors as matmul_a', short_tail=dict(input=[128], output=[128]))
    return dict(profile=profile, calibration=calibration, graph=graph, shapes=shapes)


def run(data, *, max_slowdown=0.05, tolerance=0.05, switch_us=100.0, factor=5.0, mode='lookahead'):
    if not data['profile'].get('synthetic') or not data['calibration'].get('synthetic'):
        raise ValueError('This entry point accepts synthetic fixtures only; it never deploys a plan')
    slowdown = finite(max_slowdown, 'explicit demo slowdown', 0)
    predicted = predicted_candidates(data['profile'], data['calibration'], allow_synthetic=True)
    indexed = {tuple(c['triplet']): {(r['phase'], r['name']): r for r in c['operators']} for c in predicted}
    reference = tuple(data['profile']['reference'])
    choices, frequencies = [], {}
    graph = OrderedDAG(tuple(data['graph']['nodes']), tuple(tuple(e) for e in data['graph']['edges'])).validate()
    for name in graph.nodes:
        key = ('synthetic', name)
        record = indexed[reference][key]
        resource = ('effective_sparse_memory_bound' if record['s_comp'] > 0 and record['memory_ms'] >= record['compute_ms']
                    else 'compute_bound' if record['compute_ms'] >= record['memory_ms'] else 'dense_memory_bound')
        allowed, rule = coordination_vectors(indexed, key, resource)
        choice = minimum_model_energy([(f, indexed[f][key]['latency_ms']/1000, indexed[f][key]['power_w'])
                                       for f in allowed], record['latency_ms']/1000*(1+slowdown))
        choices.append(dict(choice, name=name, resource_class=resource, coordination_rule=rule,
                            effective_s_comp=record['s_comp'], effective_s_mem=record['s_mem']))
        frequencies[name] = Frequency.of(choice['triplet'])
    def duration(names, frequency):
        return sum(indexed[frequency.values][('synthetic', name)]['latency_ms'] for name in names)/1000
    raw, decisions = greedy_partition(graph, frequencies.__getitem__, duration,
        relative_frequency_similarity(tolerance), switch_us/1e6, factor)
    blocks = [dict(block_id=i, operators=b['nodes'], triplet=b['triplet'],
                   estimated_ms=b['estimated_s']*1000, amortized_strict_gt=b['amortized_strict_gt'])
              for i, b in enumerate(raw)]
    governor = GovernorReference(indexed, reference, mode)
    now = 0
    for i, block in enumerate(blocks):
        upcoming = blocks[i+1] if i+1 < len(blocks) else None
        ticket = governor.begin_submission(block, upcoming, max(f[0] for f in indexed), switch_us, now)
        governor.gpu_boundary(ticket, now+1000)
        governor.observe_start(ticket, now+2000)
        governor.end_submission(ticket, now+3000)
        state = governor.tickets[ticket]
        governor.poll(ticket, max(state.due_ns, now+3001), end_ready=False)
        end = state.start_seen_ns + state.estimated_ns + 2000
        governor.poll(ticket, end, end_ready=True)
        now = end + 1000
    recovery = governor.restore(now)
    return dict(schema=4, synthetic=True, deployable=False,
        evidence=dict(equations='executed with synthetic scalar inputs', partition='executed with supplied descriptor graph',
                      governor='executed with virtual host time and supplied event readiness',
                      hardware='NOT EXECUTED', paper_results='NOT REPRODUCED'),
        parameters=dict(max_slowdown=slowdown, similarity_tolerance=tolerance, switch_us=switch_us, N=factor),
        assumptions=['Input coefficients/OPPs/temperature/activity are invented fixtures.',
                     'Slowdown/tolerance/lead and scalar V/f mapping are demo choices, not reported paper parameters.',
                     'Block time is a serialized sum; arbitrary parallel DAG timing is not inferred.',
                     'Dense-memory coordination falls back to an explicitly proposed budgeted selector.',
                     'Virtual end-pending is not physical PLL overlap or measured energy.'],
        shapes=data.get('shapes'), operator_choices=choices, blocks=blocks, partition_decisions=decisions,
        virtual_events=governor.events, virtual_restore=recovery)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', help='Synthetic descriptor JSON; no models/datasets are loaded')
    p.add_argument('--scenario', choices=['dense', 'effective-sparse'], default='effective-sparse')
    p.add_argument('--max-slowdown', type=float, default=0.05, help='Explicitly proposed demo default, not a paper parameter')
    p.add_argument('--similarity-tolerance', type=float, default=0.05)
    p.add_argument('--switch-us', type=float, default=100.0, help='Virtual fixture lead, not Thor timing')
    p.add_argument('--factor', type=float, default=5)
    p.add_argument('--mode', choices=['serial', 'lookahead'], default='lookahead')
    p.add_argument('--output')
    args = p.parse_args()
    data = json.loads(Path(args.input).read_text(encoding='utf-8')) if args.input else fixture(args.scenario == 'effective-sparse')
    result = run(data, max_slowdown=args.max_slowdown, tolerance=args.similarity_tolerance,
                 switch_us=args.switch_us, factor=args.factor, mode=args.mode)
    output = json.dumps(result, indent=2, ensure_ascii=False) + '\n'
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output, encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
