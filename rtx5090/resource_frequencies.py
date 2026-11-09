"""Frequency-grid and complete-training merge contracts; no hardware access."""
from collections import Counter
from .observe_core import digest


def resolve_rates(requested, common):
    if not requested or len(set(requested))!=len(requested) or any(type(v) is not int or v<=0 for v in requested):
        raise ValueError('Unique positive integer requested frequencies required')
    if not common:raise ValueError('No common supported GPU frequencies')
    mapping={str(f):min(common,key=lambda x:(abs(x-f),x)) for f in requested}
    if any(abs(mapping[str(f)]-f)>15 for f in requested):
        raise ValueError('Nearest common frequency farther than 15 MHz; explicit new protocol required')
    values=list(mapping.values())
    if len(set(values))!=len(values):raise ValueError('Requested frequencies collapse to same supported setting')
    return values,mapping


def holdout_rates(rates,training):
    if not training or any(f in training or not min(training)<f<max(training) for f in rates):
        raise ValueError('Holdout settings must be distinct from and inside frozen training range')


def merge_training(collections):
    if len(collections)<2:raise ValueError('At least two completed training collections required')
    identities={c.get('identity_fingerprint') for c in collections}
    if len(identities)!=1 or None in identities:raise ValueError('Cannot merge different resident identities')
    durations={c.get('requested_window_s') for c in collections}
    if len(durations)!=1 or None in durations:raise ValueError('Measurement window duration changed')
    rows=[];seen=set();settings=set();layout=None
    for c in collections:
        if c.get('status')!='resource_collection_complete' or c.get('collection_role')!='train' or c.get('restoration',{}).get('verified') is not True:
            raise ValueError('Only completed restored training collections; no pilot/holdout/partial data')
        source_rates={r['requested_mhz'] for r in c['rows']}
        if source_rates & settings:raise ValueError('Overlapping control conditions; do not silently reweight training')
        settings|=source_rates
        grids=[]
        for rate in source_rates:
            grid=Counter((r['rank'],r['family'],str(r['shape']),r['data_semantics'],r['memory_mhz'])
                         for r in c['rows'] if r['requested_mhz']==rate)
            if any(v!=3 for v in grid.values()):raise ValueError('Require three repeats of each rank/family/shape')
            if {k[0] for k in grid}!={0,1}:raise ValueError('Both ranks required')
            grids.append(grid)
        if not grids or any(g!=grids[0] for g in grids):raise ValueError('Unbalanced frequency grid')
        if layout is not None and grids[0]!=layout:raise ValueError('Resource/shape/memory protocol changed')
        layout=grids[0]
        for row in c['rows']:
            if row['split']!='train' or row['identity_fingerprint'] not in identities:raise ValueError('Row split/identity mismatch')
            if row['id'] in seen:raise ValueError('Repeated observation ID')
            seen.add(row['id']);rows.append(row)
    return dict(status='resource_collection_complete',collection_role='train',
        identity_fingerprint=next(iter(identities)),requested_window_s=next(iter(durations)),
        rows=rows,source_sha256=[digest(c) for c in collections],training_requested_mhz=sorted(settings),
        clock_writes=False,merge_only=True,restoration=dict(verified=True,basis='each_source_restoration_verified'))
