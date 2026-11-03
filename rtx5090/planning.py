"""Measured-latency candidates + Algorithm-1 reference partition, GPU domain only."""
from pathlib import Path
from .contracts import UNITS, digest_json, validate_plan
from sparsedvfs.graph_partitioner import partition


def candidate(profile, path, slowdown=.05, factor=5):
    if profile.get('synthetic') or profile.get('switch_source') != 'measured':
        raise ValueError('Real candidate generation requires real profiling and measured switching')
    table = {int(s['mhz']): s['latencies'] for s in profile['samples']}
    if len(table) < 2: raise ValueError('Need at least two measured clocks')
    operators = []
    for phase in ('prefill','decode'):
        for unit in UNITS:
            best = min(t[phase][unit] for t in table.values())
            if best <= 0: raise ValueError('Invalid module duration')
            frequency = min(f for f,t in table.items() if t[phase][unit] <= best*(1+slowdown))
            # Paper Frequency requires strictly positive domains. Unit placeholders are
            # used ONLY within the mathematical partitioner, then removed before deployment.
            operators.append(dict(phase=phase, name=unit, triplet=[1, frequency*1000000, 1]))
    def duration(names, phase, vector):
        return sum(table[vector[1]//1000000][phase][name] for name in names)
    blocks, decisions = partition(operators, duration, profile['switch_us'], factor, .05, include_decisions=True)
    for block in blocks: block['triplet'] = [0, block['triplet'][1], 0]
    value = dict(schema=5, identity=profile['identity'], synthetic=False, units=UNITS, granularity='module',
                 selector='latency_candidates_energy_selected_on_train', blocks=blocks,
                 profile_file=str(Path(path).resolve()), profile_sha256=digest_json(profile),
                 switch_us=profile['switch_us'], operator_slowdown=slowdown,
                 aggregation_factor_N=factor, partition_decisions=decisions,
                 similarity_tolerance=.05, energy_selection_pending=True,
                 caveat='GPU-only coarse modules; not full paper equations or sparse-kernel execution')
    validate_plan(value)
    return value


def select_rates(supported, observed):
    supported = sorted(set(supported))
    if not supported or not observed or observed <= 0: raise ValueError('Missing supported/observed clocks')
    rates = sorted({min(supported, key=lambda f: (abs(f-observed*r), f)) for r in (1, .9, .8, .7)})
    if len(rates) < 2: raise ValueError('Fewer than two distinct candidate frequencies')
    return rates
