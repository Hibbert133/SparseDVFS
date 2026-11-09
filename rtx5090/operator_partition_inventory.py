"""Offline operator occurrences for future Algorithm 1 inputs; never a DAG certificate."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

from .eager_core import context_key
from .observe_core import GPU_UUIDS, digest, signature, write_json


def inventory(source, rank, contexts):
    if source.get('status') != 'complete' or source.get('rank') != rank:
        raise ValueError('Need complete trace for the requested rank')
    identity = source['identity']
    if identity.get('gpu_uuid') != GPU_UUIDS[rank]:
        raise ValueError('Trace rank/UUID mismatch')
    nodes, catalog, patterns = [], {}, defaultdict(list)
    skipped = Counter()
    for ordinal, op in enumerate(source['operators']):
        try:
            context = context_key(op['context'])
        except (KeyError, ValueError) as exc:
            skipped[str(exc)] += 1
            continue
        if context not in contexts:
            continue
        if op.get('rank') != rank:
            raise ValueError('Operator rank differs from trace rank')
        key, meta = signature(op)
        catalog[key] = meta
        parent = op.get('parent', '__unknown__')
        match = re.search(r'(?:^|\.)layers\.(\d+)(?:\.|$)', parent)
        layer = int(match.group(1)) if match else None
        activities = op.get('gpu_activities', [])
        node = dict(id=f'{rank}:{ordinal}', source_ordinal=ordinal, context=context,
                    layer=layer, parent_label=parent, operator=op['overload'],
                    signature_id=key, opaque=op['opaque'], stream=op.get('stream'),
                    kernels=[a['name'] for a in activities if a.get('category') == 'kernel'],
                    gpu_activity_count=len(activities),
                    latency_by_frequency=None, energy_by_frequency=None,
                    legal_control_boundary_verified=False)
        nodes.append(node)
        if layer is not None:
            patterns[(context, layer)].append(key)
    if not nodes or set(contexts) - {n['context'] for n in nodes}:
        raise ValueError('Requested contexts absent from trace')
    groups = defaultdict(list)
    for (context, layer), sequence in patterns.items():
        groups[(context, digest(sequence))].append(layer)
    return dict(rank=rank, identity=identity, nodes=nodes, signatures=catalog,
                layer_template_candidates=[dict(context=c, sequence_hash=h, layers=ls)
                                           for (c, h), ls in groups.items()],
                skipped_context_reasons=dict(skipped),
                source_attribution=source.get('attribution'),
                sequence_semantics='observed_dispatch_order_not_dependency_DAG',
                template_reuse_validated=False, current_service_binding_verified=False,
                dependencies=None, deployable=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--contexts', nargs='+', default=['prefill', 'decode.0'])
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    summaries = []
    for rank in (0, 1):
        path = a.trace / f'rank-{rank}' / 'request.json'
        raw = path.read_bytes()
        source = json.loads(raw)
        source_hash = hashlib.sha256(raw).hexdigest()
        del raw
        result = inventory(source, rank, a.contexts)
        del source
        result['source_sha256'] = source_hash
        write_json(a.out / f'rank-{rank}.json', result)
        counts = Counter(n['operator'] for n in result['nodes'])
        summary = dict(rank=rank, occurrences=len(result['nodes']),
                       signatures=len(result['signatures']), operators=dict(counts),
                       occurrences_without_kernel=sum(not n['kernels'] for n in result['nodes']),
                       layer_template_candidates=result['layer_template_candidates'])
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False))
        del result
    write_json(a.out / 'summary.json', dict(status='operator_inventory_only', ranks=summaries,
               costs_validated=False, dependency_DAG_verified=False, deployable=False))
    print('Saved:', a.out)


if __name__ == '__main__':
    main()
