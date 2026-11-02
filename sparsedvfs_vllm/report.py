"""Evidence-based summaries; no expected speedup or energy saving is filled in."""
import csv
import json
import statistics
from collections import Counter
from pathlib import Path
from sparsedvfs.hardware import load_json, write_json
from .contracts import UNITS, REVISION, digest_json


def stats(values):
    values = [x for x in values if x is not None]
    if not values:
        return None
    return dict(n=len(values), mean=statistics.mean(values),
                std=statistics.stdev(values) if len(values) > 1 else None,
                min=min(values), max=max(values))


def controller_audit(directory):
    path = Path(directory)
    trace_path, cuda_path = path / 'fuse_trace.csv', path / 'cuda_timing.csv'
    gpu_ticket, violations, kinds, cuda_kinds = 0, [], Counter(), Counter()
    callbacks, observation_lags, poll_lateness = [], [], []
    if trace_path.exists():
        with trace_path.open(newline='') as source:
            for row in csv.DictReader(source):
                ticket = int(row['ticket'])
                kinds[row['kind']] += 1
                if row['kind'] == 'gpu_stream_boundary':
                    if ticket <= gpu_ticket:
                        violations.append(row)
                    gpu_ticket = ticket
                if row['kind'] == 'gpu_event_lookahead' and ticket != gpu_ticket:
                    violations.append(row)
    if cuda_path.exists():
        with cuda_path.open(newline='') as source:
            for row in csv.DictReader(source):
                cuda_kinds[row['kind']] += 1
                if row['kind'] == 'stream_boundary_callback_observed':
                    callbacks.append(float(row['callback_duration_ns'])/1000)
                    observation_lags.append(float(row['start_observation_lag_ns'])/1000)
                if row['kind'].startswith('lookahead_written') and row.get('poll_late_ns'):
                    poll_lateness.append(float(row['poll_late_ns'])/1000)
    restore_file = path / 'restore_status.json'
    requested = load_json(path / 'requested_schedule.json') if (path / 'requested_schedule.json').exists() else {}
    blocks = [e for e in requested.get('events', []) if e['kind'] == 'cuda_block']
    return dict(core_kinds=dict(kinds), cuda_kinds=dict(cuda_kinds),
                dvfs_applied=requested.get('dvfs_applied'),
                reference_abi=requested.get('reference_abi'),
                lookahead_resource_requests_issued=kinds['gpu_event_lookahead'],
                lookahead_unchanged_target_noops=cuda_kinds['lookahead_noop_same_target'],
                cuda_blocks_requested=len(blocks), lookahead_requested=sum(e['lookahead'] for e in blocks),
                stale_write_violations=violations,
                boundary_callback_us=stats(callbacks), start_observation_lag_us=stats(observation_lags),
                lookahead_poll_late_us=stats(poll_lateness),
                boundary_queue_delay_caveat='queue-to-callback delay includes preceding GPU work; not an isolated bubble',
                restored=load_json(restore_file)['restored'] if restore_file.exists() else None,
                physical_transition_overlap_verified=False,
                lookahead_evidence='end-pending proves CUDA event pending after sysfs return; PLL completion is unmeasured')


def coverage(trace, output_tokens):
    if trace.get('granularity') == 'operator':
        from .operator_contracts import context_key, validate_certificate
        module = dict(trace, rows=trace['module_rows'])
        module.pop('granularity', None)
        result = coverage(module, output_tokens)
        try:
            cert = validate_certificate(trace['coverage_certificate'], trace['identity'])
            steps = trace['operator_steps']
            for step in steps:
                case = cert['cases'][context_key(step)]
                if [call['layout'] for call in step['calls']] != case['layout']:
                    raise ValueError('Runtime dispatcher layout changed')
            if len(steps) != len(trace['steps']):
                raise ValueError('Incomplete operator steps')
            result.update(granularity='operator', certified_contexts=len(cert['cases']),
                          operator_event_rows=len(trace['rows']), diagnostic_profiler_in_benchmark=False)
        except (KeyError, ValueError) as exc:
            result.update(valid=False, operator_error=str(exc))
        return result
    by_request = {}
    for step in trace['steps']:
        by_request.setdefault(step['request_id'], []).append(step)
    errors = []
    for rid, steps in by_request.items():
        if (len(steps) != output_tokens or [s['phase'] for s in steps] !=
                ['prefill'] + ['decode'] * (output_tokens-1) or any(s['order'] != UNITS for s in steps)):
            errors.append(rid)
    row_count = Counter(r['request_id'] for r in trace['rows'])
    errors += [rid for rid, steps in by_request.items() if row_count[rid] != len(steps)*len(UNITS)]
    streams = {r['stream'] for r in trace['rows']}
    return dict(requests=len(by_request), valid=bool(by_request) and not errors and len(streams) == 1,
                errors=sorted(set(errors)), streams=sorted(streams),
                steps=len(trace['steps']), event_rows=len(trace['rows']))


def report(root):
    root = Path(root)
    data = load_json(root / 'bench_latest.json')
    plan = load_json(root / 'plan.json')
    groups, checks = {}, []
    for mode in ('native', 'instrumented', 'fixed', 'serial', 'lookahead'):
        rows = [r for r in data['results'] if r['mode'] == mode]
        if not rows:
            continue
        power_valid = all(r['measurement']['power']['valid'] for r in rows)
        values = dict(ttft_ms=stats([r['client']['ttft_ms'] for r in rows]),
                      tpot_ms=stats([r['client']['tpot_ms'] for r in rows]),
                      request_ms=stats([r['client']['request_ms'] for r in rows]),
                      request_energy_j=stats([r['measurement']['power']['energy_j'] for r in rows]) if power_valid else None,
                      energy_per_output_token_j=stats([r['measurement']['power']['energy_j'] /
                          r['client']['usage']['completion_tokens'] for r in rows]) if power_valid else None,
                      power_valid=power_valid, readback_samples=sum(len(r['measurement']['frequency_readbacks']) for r in rows),
                      power_scope=sorted(set(r['measurement']['power'].get('scope', 'unavailable') for r in rows)))
        readbacks = [sample for r in rows for sample in r['measurement']['frequency_readbacks']]
        valid_readback = lambda sample: (bool(sample['cpu_khz']) and all(isinstance(v, int) and v > 0
                                         for v in sample['cpu_khz'] + [sample['gpu_hz'], sample['emc_hz']]))
        values['frequency_readback_valid'] = bool(readbacks) and all(valid_readback(s) for s in readbacks)
        values['observed_frequency_values'] = dict(
            cpu_khz=sorted({v for s in readbacks for v in s['cpu_khz'] if v is not None}),
            gpu_hz=sorted({s['gpu_hz'] for s in readbacks if s['gpu_hz'] is not None}),
            emc_hz=sorted({s['emc_hz'] for s in readbacks if s['emc_hz'] is not None}))
        checks.append(values['frequency_readback_valid'])
        path = Path(rows[0]['run_dir'])
        if mode != 'native':
            worker_file = path / 'worker/worker_trace.json'
            if not worker_file.exists():
                raise RuntimeError(f'Missing worker trace for {mode}; shutdown/export was incomplete')
            trace = load_json(worker_file)
            values['coverage'] = coverage(trace, rows[0]['workload']['output_tokens'])
            values['identity_matches_plan'] = trace['identity'] == plan['identity']
            per_request = Counter()
            for row in trace['rows']:
                if trace.get('granularity') == 'operator' and row['name'] not in trace['units'].get(row['phase'], []):
                    continue
                per_request[row['request_id']] += row['latency_ms']
            values['worker_main_stream_unit_sum_ms'] = stats(list(per_request.values()))
            checks += [values['coverage']['valid'], values['identity_matches_plan'],
                       values['coverage']['requests'] == len(rows)]
        if mode in ('fixed', 'serial', 'lookahead'):
            values['controller'] = controller_audit(path / 'worker/controller')
            checks += [values['controller']['restored'] is True,
                       not values['controller']['stale_write_violations'],
                       values['controller']['dvfs_applied'] is True]
            if mode in ('serial', 'lookahead'):
                control = values['controller']
                count = control['cuda_blocks_requested']
                checks += [count > 0, control['core_kinds'].get('gpu_stream_boundary', 0) == count,
                           control['core_kinds'].get('cpu_submission_boost', 0) == count,
                           control['cuda_kinds'].get('gpu_end_observed', 0) == count]
                if mode == 'lookahead':
                    values['lookahead_exercised'] = (control['lookahead_requested'] > 0 and
                                                     control['lookahead_resource_requests_issued'] > 0)
                    values['lookahead_status'] = 'resource_request_observed' if values['lookahead_exercised'] else 'not_exercised'
                    checks.append(values['lookahead_exercised'])
        groups[mode] = values
    scopes = {tuple(g['power_scope']) for g in groups.values()}
    rows = data['results']
    identity_consistent = (len({r['image_id'] for r in rows}) == 1 and
                           {r['model_revision'] for r in rows} == {REVISION} and
                           len({digest_json(r['workload']) for r in rows}) == 1 and
                           len({r['client']['output_sha256'] for r in rows}) == 1)
    comparable = (data['comparable'] and identity_consistent and len(scopes) == 1 and
                  all(g['power_valid'] for g in groups.values()) and bool(checks) and all(checks))
    baseline = groups['native']['request_energy_j']
    for mode, group in groups.items():
        group['energy_saving_vs_eager_native_pct'] = (
            100 * (1 - group['request_energy_j']['mean']/baseline['mean'])
            if comparable and baseline and baseline['mean'] > 0 else None)
    result = dict(schema=5, granularity=plan.get('granularity', 'module'),
                  comparable=comparable, mechanism_acceptance=bool(checks) and all(checks),
                  method_fidelity=plan.get('fidelity'), paper_experiments_reproduced=False,
                  groups=groups, blocks=plan['blocks'],
                  unamortized_blocks=[b['block_id'] for b in plan['blocks'] if not b['amortized_strict_gt']],
                  selector=plan['selector'], emc_grid=plan.get('emc_grid'), source_run=data['run_dir'],
                  caveats=['Whole-request power includes module/carrier according to the selected sensor.',
                           'Operator energy in profile is an average-power proxy, not per-operator power integration.',
                           'CUDA unit sums cover the current main stream, exclude gaps and cannot replace client latency.',
                           'Frequency sampling is coarse (20 ms); it cannot prove every short block reached its requested frequency.',
                           'Negative or zero energy savings are valid. production-native is a separate deployment reference.',
                           'Fixed mode pins the reference vector once; per-block CPU race-to-submit exists only in serial/lookahead.'])
    result['caveats'].append('Unchanged-target look-ahead is a no-op; it is not counted as an issued GPC/EMC request or transition.')
    write_json(root / 'report.json', result)
    def mean(g, key):
        return f"{g[key]['mean']:.3f}" if g.get(key) else 'N/A'
    lines = ['# Thor vLLM SparseDVFS 实测报告', '',
             f"机制验收：{result['mechanism_acceptance']}；能耗可比较：{comparable}；EMC 网格：{result['emc_grid']}。", '',
             '| 模式 | TTFT ms | TPOT ms/token | 请求 ms | J/request | J/output token | 相对 eager native 节能 % |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for mode, g in groups.items():
        saving = g['energy_saving_vs_eager_native_pct']
        lines.append('| ' + ' | '.join([mode, mean(g, 'ttft_ms'), mean(g, 'tpot_ms'), mean(g, 'request_ms'),
                     mean(g, 'request_energy_j'), mean(g, 'energy_per_output_token_j'),
                     f'{saving:.3f}' if saving is not None else 'N/A']) + ' |')
    lines += ['', '逐请求原始值、标准差、层覆盖、过期请求检查、CUDA look-ahead 状态、恢复读回及未摊销块见 report.json。', '',
              '本表不包含初始化和 warm-up。HTTP 指标采用 token ID/usage 核对；多个 token 合并到同一 chunk 时只知道该批的接收时间。', '',
              '模块级能量代理、整次请求功率积分与物理切频时间分别报告；本测试不证明论文原始 Orin 数值。', '']
    (root / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))
