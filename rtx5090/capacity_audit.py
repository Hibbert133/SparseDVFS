"""Read-only audit of independent-capacity artifacts; no GPU, fitting or retries."""
import argparse
from pathlib import Path
from statistics import mean
from .service_core import read, write_new, positive, errors
from .observe_core import GPU_UUIDS, digest


def audit(collection, observation):
    if collection['identity_fingerprint'] != observation['identity_fingerprint']:
        raise ValueError('Replay/collection identities differ')
    if observation['status'] != 'complete' or not observation['coverage']['passed']:
        raise ValueError('Incomplete replay')
    result = dict(status='offline_audit_complete', clock_writes=False,
        identity_fingerprint=collection['identity_fingerprint'],
        collection_sha256=digest(collection), observation_sha256=digest(observation),
        equation_1_parameters_validated=False, equation_2_parameters_validated=False,
        ranks=[])
    replay = {r['rank']:r for r in observation['matrix_replay']}
    for measured in collection['ranks']:
        rank = measured['rank']; uuid = GPU_UUIDS[rank]
        if measured['uuid'] != uuid or replay[rank]['uuid'] != uuid:
            raise ValueError('Rank UUID mismatch')
        row = dict(rank=rank, status=measured['status'], primitives=[], linear_calls=[], phases={})
        result['ranks'].append(row)
        if measured['status'].startswith('blocked_'):
            row['memory_evidence'] = {k:measured[k] for k in
                ('before_context','after_context','cuda_free_mib') if k in measured}
            continue
        if measured['status'] != 'conditional_primitive_model_collected':
            raise ValueError('Failed collection cannot be audited as completed')
        for primitive in measured['rows']:
            ts = primitive['seconds']
            for t in ts: positive(t, 'primitive time')
            snapshots = [primitive['before'], primitive['after']]
            if any(s['uuid'] != uuid for s in snapshots): raise ValueError('Snapshot UUID mismatch')
            row['primitives'].append(dict(name=primitive['name'],
                batch_means_us=[v*1e6 for v in ts],
                spread_percent=100*(max(ts)-min(ts))/mean(ts),
                sm_before_after_mhz=[s['sm_mhz'] for s in snapshots],
                memory_before_after_mhz=[s['memory_mhz'] for s in snapshots],
                temperature_before_after_c=[s['temperature_c'] for s in snapshots],
                during_batch_clock_verified=False))
        lookup = {r['name']:r for r in measured['MLP_transfer_diagnostic']['rows']}
        for call in replay[rank]['rows']:
            comparison = lookup[call['phase']+'/'+call['module']]
            row['linear_calls'].append(dict(module_source=call['parent'], phase=call['phase'],
                pytorch_operator='aten.linear.default', math='Y=X@W.T (no bias)',
                shape_MNK=[call[k] for k in ('M','N','K')],
                operation_kind='matrix_vector' if call['M']==1 else 'matrix_matrix',
                cuda_kernel_names=None, cuda_kernel_multiplicity=None,
                kernel_names_verified=call['kernel_names_verified'],
                cache_state_matched=call['cache_state_matched'],
                timing_scope=call['timing_scope'],
                observed_us=comparison['observed_s']*1e6,
                predicted_us=comparison['predicted_s']*1e6,
                error_percent=comparison['relative_error']*100))
        for phase in ('prefill','decode'):
            calls = [c for c in row['linear_calls'] if c['phase']==phase]
            row['phases'][phase] = errors([c['observed_us'] for c in calls], [c['predicted_us'] for c in calls])
        row['primitive_holdout_scores'] = measured['primitive_holdout']['scores']
        scores = row['primitive_holdout_scores']
        row['primitive_numeric_gate_passed'] = scores['wape'] <= .1 and scores['p90_relative_error'] <= .2
    result['interpretation'] = ('Boundary snapshots cannot prove in-batch frequency stability. '
        'Same-phase error is diagnostic, not a new independent validation set. '
        'Unknown kernel names remain unknown; old-process traces are not attached as proof.')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection', required=True)
    p.add_argument('--replay', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    result = audit(read(args.collection), read(Path(args.replay)/'observation.json'))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    write_new(out/'audit.json', result)
    lines = ['Offline audit; parameters remain unvalidated.']
    for row in result['ranks']:
        lines.append('rank %d: %s'%(row['rank'],row['status']))
        for v in row['primitives']:
            lines.append('%s | batches us=%s | spread=%.2f%% | SM=%s | MEM=%s | C=%s'%(
                v['name'],','.join('%.3f'%t for t in v['batch_means_us']),v['spread_percent'],
                v['sm_before_after_mhz'],v['memory_before_after_mhz'],v['temperature_before_after_c']))
        for phase, score in row['phases'].items():
            lines.append('%s transfer: WAPE=%.2f%% P90=%.2f%% (diagnostic)'%(
                phase,100*score['wape'],100*score['p90_relative_error']))
        for v in row['linear_calls']:
            lines.append('%s | %s | %s | M,N,K=%s | observed=%.3fus predicted=%.3fus | kernel=unverified'%(
                v['phase'],v['module_source'],v['pytorch_operator'],v['shape_MNK'],v['observed_us'],v['predicted_us']))
        if 'memory_evidence' in row: lines.append('memory: '+str(row['memory_evidence']))
    (out/'REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines)); print('Saved:',out)


if __name__ == '__main__': main()
