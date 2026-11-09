"""Offline request-budget selector plus literal Algorithm 1 and post-merge audit.

Pareto dynamic programming is an explicit port choice, NOT a paper algorithm.
Input costs are additive paired predictions. Never grants live authorization.
"""
import argparse
import math
from pathlib import Path

from .partition_dryrun import partition, demo
from .service_core import read, write_new


def number(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError('Invalid '+name)
    return value


def select(nodes, table, deadline, switch_s, switch_energy, initial, max_states=50000):
    """Exact Pareto frontier for the unmerged additive sequence, indexed by last f."""
    frontier = {initial: [(0.0, 0.0, ())]}
    for node in nodes:
        next_frontier = {}
        for rate in (2250, 2527):
            cost = table[node][str(rate)]
            candidates = []
            for previous, states in frontier.items():
                for seconds, energy, path in states:
                    t = seconds + cost['pair_latency_s'] + (switch_s if rate != previous else 0)
                    e = energy + cost['pair_energy_j'] + (switch_energy if rate != previous else 0)
                    if t <= deadline:
                        candidates.append((t, e, path + (rate,)))
            candidates.sort()
            best = math.inf
            keep = []
            for state in candidates:
                if state[1] < best:
                    keep.append(state)
                    best = state[1]
            next_frontier[rate] = keep
        frontier = next_frontier
        if sum(map(len, frontier.values())) > max_states:
            raise ValueError('Pareto state cap exceeded; no approximate pruning or plan emitted')
    states = [s for values in frontier.values() for s in values]
    if not states:
        return None
    t, e, path = min(states, key=lambda s: (s[1], s[0], s[2]))
    return dict(choices=dict(zip(nodes, path)), latency_s=t, energy_j=e)


def plan(data):
    # Existing parser checks graph/table/switch completeness. Do not invent labels.
    validation = partition(data)
    result = dict(schema='request-budget-partition-v1', deployable=False, clock_writes=False,
                  selector='Pareto_DP_port_choice_not_specified_in_paper',
                  status='blocked_missing_inputs', blockers=list(validation['blockers']))
    budget = data.get('request_budget')
    if budget is None:
        result['blockers'].append(dict(reason='missing_request_budget_and_fixed_overhead'))
    if 'switch_energy_j' not in data:
        result['blockers'].append(dict(reason='missing_switch_energy_or_explicit_assumption'))
    if data.get('initial_mhz') not in (2250, 2527):
        result['blockers'].append(dict(reason='missing_initial_frequency'))
    if result['blockers']:
        return result
    reference = number(budget['reference_s'], 'reference', True)
    extra = number(budget['max_increase'], 'max increase')
    fixed = number(budget['fixed_overhead_s'], 'fixed overhead')
    switch_energy = number(data['switch_energy_j'], 'switch energy')
    deadline = reference * (1 + extra)
    result.update(deadline_s=deadline, fixed_overhead_s=fixed,
                  reference_scope=budget.get('reference_scope', 'unspecified_not_AUTO_certificate'),
                  synthetic=data.get('synthetic') is True)
    chosen = select(data['nodes'], data['candidates'], deadline-fixed,
                    data['switch_s'], switch_energy, data['initial_mhz'])
    if chosen is None:
        result['status'] = 'no_feasible_additive_sequence'
        return result
    result['before_partition'] = chosen
    merged = partition(data, choices=chosen['choices'])
    transitions = merged['transition_count'] + (merged['blocks'][0]['triplet'][0] != data['initial_mhz'])
    total = fixed + merged['compute_s'] + transitions * data['switch_s']
    energy = merged['total_model_energy_j'] + transitions * switch_energy
    result.update(partition=merged, modeled_request_s=total, modeled_energy_j=energy,
                  switches_including_initial=transitions,
                  request_budget_met=total <= deadline,
                  eq3_all_blocks_passed=merged['eq3_all_blocks_passed'],
                  status='offline_postmerge_audit_complete' if total <= deadline else 'postmerge_budget_failed',
                  postmerge_energy_optimal=False,
                  note='DP optimum applies before Algorithm1 merging only. Recheck time/energy after max-frequency merge. '
                       'Short tail retained. Fixed overhead must include unmodeled request work/final restore as applicable. '
                       'No hardware or full AUTO-relative validation.')
    return result


def fixture():
    value = demo()
    value.update(request_budget=dict(reference_s=.05, max_increase=.05, fixed_overhead_s=.005,
                                    reference_scope='synthetic_software_fixture'),
                 switch_energy_j=0.0, initial_mhz=2527)
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--input'); group.add_argument('--demo', action='store_true')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    result = plan(fixture() if a.demo else read(a.input))
    write_new(Path(a.out), result)
    print(result['status'])
    print('deployable:', False, 'budget:', result.get('request_budget_met'), 'Eq3:', result.get('eq3_all_blocks_passed'))
    if result['blockers']: print('blockers:', result['blockers'])
    print('Saved:', a.out)


if __name__ == '__main__': main()
