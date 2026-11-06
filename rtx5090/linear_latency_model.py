"""Practical fixed-shape latency table; not SparseDVFS Eq1/Eq2 or DVFS model."""
import argparse
from pathlib import Path
from statistics import median
from .service_core import read, write_new, errors, positive
from .observe_core import digest, GPU_UUIDS


def fit(observation):
    if observation['status'] != 'complete' or not observation.get('kernel_mapping_passed') or not observation.get('kernel_launch_geometry_passed'):
        raise ValueError('Require completed kernel-matched collection')
    model = dict(schema='linear-latency-table-v1', identity_fingerprint=observation['identity_fingerprint'],
        source_sha256=digest(observation), training_count=32, check_count=128,
        selection='median_of_all_three_N32_batch_means', operators={},
        scope='same_shape_real_tensor_repeated_linear_latency_only',
        retrospective_protocol=True, independent_validation=False,
        frequency_scaling_supported=False, energy_prediction_supported=False,
        equation_1_parameters_validated=False, equation_2_parameters_validated=False,
        block_selection_authorized=False)
    measured = []
    keys=set()
    for rank in observation['matrix_replay']:
        r=rank['rank']
        if r not in (0,1) or rank['uuid'] != GPU_UUIDS[r]: raise ValueError('Wrong rank/UUID')
        for row in rank['rows']:
            key='%s:%s/%s'%(r,row['phase'],row['module'])
            if key in keys: raise ValueError('Duplicate operator')
            keys.add(key)
            if row['collection_protocol'] != 'kernel_verified_batch_counts_v1': raise ValueError('Wrong batch protocol')
            if not row['kernel_names_verified']: raise ValueError('Kernel not verified')
            train=[b['mean_call_s'] for b in row['batches'] if b['repeats']==32]
            check=[b['mean_call_s'] for b in row['batches'] if b['repeats']==128]
            if len(train)!=3 or len(check)!=3: raise ValueError('Require all three train/check batches')
            for v in train+check: positive(v,'batch mean')
            value=median(train)
            model['operators'][key]=dict(rank=r,uuid=rank['uuid'],phase=row['phase'],
                module=row['parent'], pytorch_operator='aten.linear.default',
                shape_MNK=[row[k] for k in ('M','N','K')], dtype='torch.bfloat16',
                kernel_sequence=row['kernel_comparison']['source_sequence'],
                latency_s=value, training_values_s=train,
                use_limit='continuous_repeated_calls_not_isolated_call_or_end_to_end_request')
            measured.append(dict(key=key,prediction_s=value,check_values_s=check,
                scores=errors(check,[value]*len(check))))
    expected={'%s:%s/%s'%(r,p,m) for r in (0,1) for p in ('prefill','decode') for m in ('gate_up_proj','down_proj')}
    if keys!=expected: raise ValueError('Missing eight dual-rank linear cases')
    overall=errors([v for r in measured for v in r['check_values_s']],
                   [r['prediction_s'] for r in measured for _ in r['check_values_s']])
    report=dict(status='retrospective_check_complete',model_sha256=digest(model),rows=measured,
        overall=overall, independent_validation=False,
        all_cases_within_project_error_limits=all(r['scores']['wape']<=.1 and
            r['scores']['p90_relative_error']<=.2 for r in measured),
        note='Already-seen same-request batches; no independent accuracy guarantee. '
             'Counts 1/8 remain in source and are outside continuous-batch model scope.')
    return model,report


def validate(model, training, observation):
    """Score every prescribed new-request batch without changing predictions."""
    if digest(training) != model.get('source_sha256') or digest(fit(training)[0]) != digest(model):
        raise ValueError('Frozen model does not match training evidence')
    if observation.get('identity_fingerprint') != model['identity_fingerprint']:
        raise ValueError('Worker identity changed')
    for field in ('prompt_hash', 'request_id'):
        value = observation.get('reply', {}).get(field)
        if not value or value == training.get('reply', {}).get(field):
            raise ValueError('Require a new request and prompt: '+field)
    for evidence in (training, observation):
        if evidence.get('status') != 'complete' or not evidence.get('kernel_mapping_passed') or not evidence.get('kernel_launch_geometry_passed'):
            raise ValueError('Require completed kernel/geometry-matched collection')
        if not evidence.get('cache_isolation', {}).get('miss_verified'):
            raise ValueError('Require verified cache miss')
        usage = evidence.get('reply', {}).get('usage', {})
        if usage.get('prompt_tokens') != 512 or usage.get('completion_tokens') != 64:
            raise ValueError('Require actual 512/64 usage')
    original = {('%s:%s/%s' % (rank['rank'], row['phase'], row['module'])): row
                for rank in training['matrix_replay'] for row in rank['rows']}
    rows, seen = [], set()
    for rank in observation['matrix_replay']:
        for row in rank['rows']:
            key = '%s:%s/%s' % (rank['rank'], row['phase'], row['module'])
            if key in seen or key not in model['operators']:
                raise ValueError('Duplicate or unknown operator')
            seen.add(key)
            op, old = model['operators'][key], original[key]
            if rank['uuid'] != op['uuid'] or row['parent'] != op['module']:
                raise ValueError('Rank/UUID/module changed')
            for field in ('M', 'N', 'K', 'input_metadata', 'output_metadata'):
                if field not in row or row[field] != old[field]:
                    raise ValueError('Tensor metadata changed: '+field)
            if row.get('collection_protocol') != 'kernel_verified_batch_counts_v1' or not row.get('kernel_names_verified'):
                raise ValueError('Unverified collection protocol')
            if row['kernel_comparison']['source_sequence'] != op['kernel_sequence']:
                raise ValueError('Kernel sequence/geometry changed')
            values = []
            for count in (32, 128):
                batch = [b['mean_call_s'] for b in row['batches'] if b['repeats'] == count]
                if len(batch) != 3:
                    raise ValueError('Require all three batches at each count')
                for v in batch: positive(v, 'validation mean')
                values.extend(batch)
            prediction = op['latency_s']
            rows.append(dict(key=key, prediction_s=prediction, check_values_s=values,
                             scores=errors(values, [prediction]*len(values))))
    if seen != set(model['operators']):
        raise ValueError('Missing dual-rank cases')
    passed = all(r['scores']['wape'] <= .1 and r['scores']['p90_relative_error'] <= .2 for r in rows)
    return dict(status='new_request_check_passed' if passed else 'new_request_check_failed',
                model_sha256=digest(model), source_sha256=digest(observation), rows=rows,
                all_cases_within_project_error_limits=passed, parameters_updated=False,
                validation_counts=[32, 128], batches_per_count=3,
                scope=model['scope'], frequency_scaling_supported=False,
                energy_prediction_supported=False, equation_1_parameters_validated=False,
                equation_2_parameters_validated=False,
                note='Fresh prompt, same resident workers and weights; not cross-frequency or end-to-end validation.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay',required=True); p.add_argument('--out',required=True)
    p.add_argument('--model'); p.add_argument('--training-replay')
    args=p.parse_args()
    if bool(args.model) != bool(args.training_replay):
        p.error('--model and --training-replay must be supplied together')
    observation=read(Path(args.replay)/'observation.json')
    if args.model:
        model=read(Path(args.model))
        report=validate(model,read(Path(args.training_replay)/'observation.json'),observation)
    else:
        model,report=fit(observation)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=False)
    if not args.model: write_new(out/'model.json',model)
    write_new(out/('validation.json' if args.model else 'check.json'),report)
    lines=[report['status'],
           'Fixed-shape continuous-linear latency; no frequency or energy generalization.',
           'rank:phase/module | prediction us | check WAPE % | check P90 %']
    for r in report['rows']:
        lines.append('%s | %.3f | %.2f | %.2f'%(r['key'],r['prediction_s']*1e6,
            r['scores']['wape']*100,r['scores']['p90_relative_error']*100))
    lines.append('Per-case numeric limits met: '+str(report['all_cases_within_project_error_limits']))
    (out/'REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines)); print('Saved:',out)
    if args.model and not report['all_cases_within_project_error_limits']:
        raise SystemExit(2)


if __name__=='__main__': main()
