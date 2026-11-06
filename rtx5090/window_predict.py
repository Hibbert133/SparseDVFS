"""Restricted constant-mean empirical predictor; explicitly NOT SparseDVFS Eq1/2."""
import argparse
import statistics
from pathlib import Path
from .service_core import read, write_new, positive, quantile
from .service_client import sha
from .observe_core import digest

TARGETS = ('mean_request_s', 'ttft_s', 'tpot_s', 'host_window_s', 'host_envelope_energy_j')


def rows_at(root, split):
    root = Path(root)
    protocol = read(root/'protocol.json')
    if (protocol['input'], protocol['output'], protocol['requests'], protocol['concurrency']) != (512, 64, 20, 1):
        raise ValueError('Only measured 512/64, 20 requests, concurrency one is supported')
    result, seen = [], set()
    for index in (1, 2, 3):
        folder = root/(split+'-'+str(index))
        row = read(folder/'summary.json')
        raw = read(folder/'window.json')
        if row['status'] != 'valid_host_envelope' or raw['energy']['quality_errors'] or raw.get('cache_status') != 'verified_miss':
            raise ValueError('Invalid source window')
        if row['requests'] != 20 or len(raw['replies']) != 20:
            raise ValueError('Incomplete window')
        hashes = {r['prompt_hash'] for r in read(folder/'requests.json')}
        if len(hashes) != 20 or seen & hashes: raise ValueError('Requests overlap')
        seen.update(hashes)
        for name in TARGETS: positive(row[name], name)
        result.append(dict(summary=row, identity=raw['identity_fingerprint'],
                           evidence_sha256=sha(folder/'window.json')))
    if len({r['identity'] for r in result}) != 1: raise ValueError('Service identity changed')
    return result, seen, protocol


def fit(root):
    rows, hashes, protocol = rows_at(root, 'reference')
    return dict(schema='fixed-config-empirical-mean-v1',
        method='arithmetic_mean_of_three_reference_windows_no_tuning',
        identity_fingerprint=rows[0]['identity'], protocol=protocol,
        predictions={name:statistics.mean(r['summary'][name] for r in rows) for name in TARGETS},
        train_prompt_hashes=sorted(hashes), train_evidence_sha256=[r['evidence_sha256'] for r in rows],
        allowed_scope='same_512_64_TP2_eager_service_current_policy_host_envelope_only',
        clock_response_model=False, equation_1=False, equation_2=False,
        holdout_previously_visible_to_user_and_developer=True,
        evaluation_design='retrospective_fixed_rule_not_prospective_blind_validation')


def validate(root, model):
    if model['schema'] != 'fixed-config-empirical-mean-v1': raise ValueError('Wrong model')
    rows, hashes, protocol = rows_at(root, 'holdout')
    if (hashes & set(model['train_prompt_hashes']) or protocol != model['protocol']
            or any(r['identity'] != model['identity_fingerprint'] for r in rows)):
        raise ValueError('Holdout request/config/identity mismatch')
    scores = {}
    for name in TARGETS:
        actual = [r['summary'][name] for r in rows]
        prediction = positive(model['predictions'][name], name)
        relative = [abs(prediction-x)/x for x in actual]
        wape = sum(abs(prediction-x) for x in actual)/sum(actual)
        p90 = quantile(relative, .9)
        scores[name] = dict(prediction=prediction, actual=actual, wape=wape,
                            p90_relative_error=p90, passed=wape <= .10 and p90 <= .20)
    return dict(status='passed' if all(s['passed'] for s in scores.values()) else 'not_passed',
        model_sha256=digest(model), scores=scores,
        thresholds=dict(wape=.10, p90=.20, source='project_standard_not_paper'),
        scope=model['allowed_scope'], evaluation_design=model['evaluation_design'],
        equation_1_validated=False, equation_2_validated=False,
        evidence_sha256=[r['evidence_sha256'] for r in rows])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('fit', 'validate', 'predict'))
    p.add_argument('--data')
    p.add_argument('--model')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    if args.command == 'fit':
        if not args.data: p.error('fit requires --data')
        value = fit(args.data)
    else:
        if not args.model: p.error('--model required')
        model = read(args.model)
        if model['schema'] != 'fixed-config-empirical-mean-v1': raise ValueError('Wrong model')
        if args.command == 'validate':
            if not args.data: p.error('validate requires --data')
            value = validate(args.data, model)
        else:
            value = dict(predictions=model['predictions'], scope=model['allowed_scope'],
                identity_fingerprint=model['identity_fingerprint'], model_sha256=digest(model),
                extrapolation_supported=False, equation_1=False, equation_2=False)
    write_new(args.out, value)
    import json
    print(json.dumps(value if args.command != 'fit' else {
        'status':'frozen', 'predictions':value['predictions'], 'scope':value['allowed_scope']}, indent=2))


if __name__ == '__main__': main()
