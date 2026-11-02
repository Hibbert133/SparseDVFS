"""Offline vllm.LLM profiling/trace entry point, executed inside the Thor image."""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from sparsedvfs.hardware import load_json, write_json, wait_frequencies
from .contracts import engine_args, verify_model, digest_json, UNITS
from .telemetry import Window
from .operator_contracts import read_chrome, build_certificate, validate_certificate, context_key


def workload(path, count=512, output=128):
    path = Path(path)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('/model', local_files_only=True)
    text = (Path('/opt/sparsedvfs/configs/prompt_zh.txt')).read_text(encoding='utf-8')
    seed = tokenizer.encode(text, add_special_tokens=False)
    if not seed:
        raise ValueError('Empty prompt')
    ids = (seed * ((count + len(seed) - 1) // len(seed)))[:count]
    data = dict(input_tokens=count, output_tokens=output, input_token_ids=ids,
                prompt_sha256=digest_json(ids), temperature=0.0, seed=0,
                ignore_eos=True, add_special_tokens=False, warmups=2)
    if path.exists() and load_json(path) != data:
        raise ValueError('Existing workload differs; choose a separate results directory')
    write_json(path, data)
    return data


def public_workload(data):
    return {key: value for key, value in data.items() if key != 'input_token_ids'}


def rpc(llm, name, **kwargs):
    result = llm.collective_rpc(name, kwargs=kwargs)
    if len(result) != 1:
        raise RuntimeError('Exactly one worker RPC result is required')
    return result[0]


def generate(llm, data):
    from vllm import SamplingParams
    result = llm.generate([{'prompt_token_ids': data['input_token_ids']}],
                          SamplingParams(temperature=0.0, seed=0, max_tokens=data['output_tokens'],
                                         min_tokens=data['output_tokens'], ignore_eos=True), use_tqdm=False)[0]
    ids = list(result.outputs[0].token_ids)
    if result.prompt_token_ids != data['input_token_ids'] or len(ids) != data['output_tokens']:
        raise RuntimeError('Input/output token count changed')
    return dict(output_token_ids=ids, output_sha256=digest_json(ids),
                request_id=result.request_id, finish_reason=result.outputs[0].finish_reason)


def check_trace(trace, data):
    steps = trace['steps']
    if len(steps) != data['output_tokens'] or any(s['order'] != UNITS for s in steps):
        raise RuntimeError('Expected one fully covered worker step per output token')
    if [s['phase'] for s in steps] != ['prefill'] + ['decode'] * (len(steps)-1):
        raise RuntimeError('Scheduler-derived phase sequence is invalid')
    module_rows = trace.get('module_rows', trace['rows'])
    if len(module_rows) != len(UNITS) * len(steps):
        raise RuntimeError('CUDA event coverage is incomplete')
    if len({row['stream'] for row in trace['rows']}) != 1:
        raise RuntimeError('Trace crossed CUDA main streams; unsupported')
    if trace.get('granularity') == 'operator' and len(trace['operator_steps']) != len(steps):
        raise RuntimeError('Incomplete dispatcher step coverage')


def aggregate(rows, power, certificate=None):
    groups = {}
    for row in rows:
        groups.setdefault((row['phase'], row['name']), []).append(row['latency_ms'])
    keys = ([(phase, name) for phase in certificate['context_order'] for name in certificate['cases'][phase]['units']]
            if certificate else [(phase, name) for phase in ('prefill', 'decode') for name in UNITS])
    return [dict(phase=phase, name=name, latency_ms=statistics.mean(groups[(phase, name)]),
                 calls=len(groups[(phase, name)]), power_w=power, s_comp=0.0, s_mem=0.0,
                 sparsity_exact_zero=None, power_attribution='whole_request_average_proxy')
            for phase, name in keys]


def measure_switches(controller, hw, grid, out, repeats=10):
    import torch
    tensor = torch.ones((512, 512), device='cuda', dtype=torch.bfloat16)
    def keepalive():
        _ = tensor @ tensor
        torch.cuda.current_stream().synchronize()
    records = []
    if repeats < 1:
        raise ValueError('Switch repetitions must be positive')
    from sparsedvfs.hardware import observation
    # Every directed transition, explicitly settled at its source; no no-op samples.
    for source in grid['triplets']:
        for triplet in grid['triplets']:
            if source == triplet:
                continue
            for repeat in range(repeats):
                controller.set(source)
                wait_frequencies(hw, source, keepalive=keepalive)
                start = time.monotonic_ns()
                controller.set(triplet)
                returned = time.monotonic_ns()
                readback = wait_frequencies(hw, triplet, keepalive=keepalive)
                matched = time.monotonic_ns()
                records.append(dict(from_triplet=source, triplet=triplet, repeat=repeat,
                                    write_start_ns=start, write_return_ns=returned,
                                    software_match_ns=matched, syscall_return_us=(returned-start)/1000,
                                    request_to_matching_readback_us=(matched-start)/1000,
                                    hardware_observation=observation(hw), **readback))
    values = sorted(r['request_to_matching_readback_us'] for r in records)
    value = values[min(len(values)-1, int(0.95 * len(values)))]
    data = dict(records=records, repeats_per_directed_transition=repeats, excluded_noops=True,
                switch_us=value, statistic='global_p95_request_to_matching_readback',
                physical_pll_settle_verified=False,
                caveat='Includes read polling and keepalive; conservative software observation, not physical PLL time')
    write_json(out / 'switches.json', data)
    return data


def certify_opp(llm, data, out, certificate, triplet):
    """Diagnostic launch/stream coverage at every profiled OPP, outside timing."""
    folder = out/'torch_trace'
    before = {str(p):p.stat().st_mtime_ns for p in folder.rglob('*.pt.trace.json*')}
    rpc(llm, 'sdvfs_observe', enabled=True, run_id='opp_coverage')
    rpc(llm, 'sdvfs_diagnostic', enabled=True)
    try:
        llm.start_profile()
        try:
            outputs = [generate(llm,data) for _ in range(3)]
        finally:
            llm.stop_profile()
    finally:
        rpc(llm, 'sdvfs_diagnostic', enabled=False)
    trace = rpc(llm, 'sdvfs_export', reset=True)
    if {r['output_sha256'] for r in outputs} != {certificate['native_output_sha256']}:
        raise RuntimeError('Operator OPP diagnostic output differs from native')
    trace.update(output_sha256=outputs[0]['output_sha256'],
                 native_output_sha256=certificate['native_output_sha256'])
    files = [p for p in folder.rglob('*.pt.trace.json*') if before.get(str(p)) != p.stat().st_mtime_ns]
    if not files:
        raise RuntimeError('Missing fresh per-OPP profiler evidence')
    events = [event for path in files for event in read_chrome(path)]
    current = build_certificate(trace,events)
    for key,case in current['cases'].items():
        original = certificate['cases'][key]
        if case['layout_sha256'] != original['layout_sha256'] or case['units'] != original['units']:
            raise RuntimeError('OPP changed dispatcher/GPU attribution; operator DVFS is disabled')
    current['applied_triplet'] = triplet
    current['source_trace_files'] = [str(p) for p in files]
    current.pop('certificate_sha256')
    current['certificate_sha256'] = digest_json(current)
    write_json(out/('coverage_'+','.join(map(str,triplet))+'.json'),current)
    return current


def execute(args):
    from vllm import LLM
    verify_model('/model')
    root = Path(args.run_dir)
    root.mkdir(parents=True, exist_ok=True)
    data = workload(root / 'workload.json', args.input_tokens, args.output_tokens)
    if args.command == 'workload':
        return
    out = root / f'{args.command}_{time.time_ns()}'
    out.mkdir()
    cfg = dict(mode='instrumented', enabled=False, out_dir=str(out / 'worker'),
               granularity=args.granularity, coverage=args.coverage,
               switch_repeat=args.switch_repeat,
               hardware=args.hardware, workload=public_workload(data), run_id='warmup', apply=False)
    # Worker config must exist before LLM constructs its subprocesses.
    config_path = out / 'worker_config.json'
    write_json(config_path, cfg)
    os.environ['SPARSEDVFS_WORKER_CONFIG'] = str(config_path)
    os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '0'
    engine = engine_args()
    diagnostic = args.command == 'trace' and args.granularity == 'operator'
    if args.granularity == 'operator' and args.command in ('trace', 'profile'):
        engine['profiler_config'] = dict(profiler='torch', torch_profiler_dir=str(out/'torch_trace'),
                                        torch_profiler_record_shapes=True, torch_profiler_with_stack=False,
                                        torch_profiler_with_flops=False, ignore_frontend=True)
    llm = LLM(**engine)
    hw = load_json(args.hardware) if args.hardware else None
    try:
        if args.command == 'profile':
            if not args.apply:
                raise ValueError('A deployable profile requires explicit --apply on Thor')
            grid = load_json(args.grid_file)
            cfg.update(mode='fixed', apply=True, enabled=False, keep_fixed_during_warmup=True,
                       fixed_triplet=grid['reference'])
            rpc(llm, 'sdvfs_configure', config=cfg)
            certificate = load_json(args.coverage) if args.granularity == 'operator' else None
            if certificate:
                validate_certificate(certificate, rpc(llm, 'sdvfs_configure', config=cfg))
            rpc(llm, 'sdvfs_prepare')
            switches = rpc(llm, 'sdvfs_profile_switches', grid=grid)
            write_json(out / 'switches.json', switches)
            candidates, opp_certificates = [], {}
            for index, triplet in enumerate(grid['triplets']):
                readback = rpc(llm, 'sdvfs_profile_opp', triplet=triplet)
                microbench = rpc(llm, 'sdvfs_microbench')
                for _ in range(2):
                    generate(llm, data)
                if certificate:
                    opp_certificates[','.join(map(str,triplet))] = certify_opp(llm,data,out,certificate,triplet)
                rpc(llm, 'sdvfs_observe', enabled=True, run_id=f'opp_{index}')
                window = Window(hw).start()
                try:
                    generated = [generate(llm, data) for _ in range(args.repeat)]
                finally:
                    measured = window.stop()
                trace = rpc(llm, 'sdvfs_export', reset=True)
                # Check every request separately, not only the first decode chain.
                for rid in {s['request_id'] for s in trace['steps']}:
                    sub = dict(trace, steps=[s for s in trace['steps'] if s['request_id'] == rid],
                               rows=[r for r in trace['rows'] if r['request_id'] == rid])
                    if trace.get('granularity') == 'operator':
                        sub['module_rows'] = [r for r in trace['module_rows'] if r['request_id'] == rid]
                        sub['operator_steps'] = [r for r in trace['operator_steps'] if r['request_id'] == rid]
                    check_trace(sub, data)
                power = measured['power']['average_power_w']
                if not measured['power']['valid']:
                    raise RuntimeError('No valid request power integral; select a real total-power source')
                row = dict(triplet=triplet, operators=aggregate(trace['rows'], power, certificate),
                           measurement=measured, readback=readback, generated=generated,
                           microbench=microbench)
                candidates.append(row)
                write_json(out / f'opp_{index}.json', dict(candidate=row, worker_trace=trace))
            profile = dict(schema=5, granularity=args.granularity, identity=trace['identity'],
                           units=trace['units'], reference=grid['reference'],
                           candidates=candidates, synthetic=False, dvfs_applied=True, grid=grid,
                           degraded_no_emc=False, switch_us=switches['switch_us'],
                           power_attribution='whole_request_average_proxy', run_dir=str(out))
            if certificate:
                profile['coverage_certificate'] = certificate
                profile['opp_coverage_certificates'] = opp_certificates
            hashes = {r['output_sha256'] for c in candidates for r in c['generated']}
            if len(hashes) != 1:
                raise RuntimeError('Outputs differ across OPPs; profile is not comparable')
            rpc(llm, 'sdvfs_export', close=True)
            write_json(root / 'profile.json', profile)
        else:
            for _ in range(2):
                generate(llm, data)
            cfg.update(enabled=True, mode=args.mode, apply=args.apply, plan=args.plan,
                       diagnostic_profiler=diagnostic,
                       fixed_triplet=load_json(args.grid_file)['reference'] if args.grid_file else None,
                       run_id=args.command)
            rpc(llm, 'sdvfs_configure', config=cfg)
            rpc(llm, 'sdvfs_export', reset=True)
            rpc(llm, 'sdvfs_prepare')
            window = Window(hw).start() if hw else None
            started = time.monotonic_ns()
            try:
                if diagnostic:
                    llm.start_profile()
                results = [generate(llm, data) for _ in range(3 if args.command == 'trace' else 1)]
                result = results[-1]
            finally:
                if diagnostic:
                    llm.stop_profile()
                measured = window.stop() if window else {'wall_ms': (time.monotonic_ns()-started)/1e6}
            trace = rpc(llm, 'sdvfs_export', reset=False, close=True)
            for rid in {s['request_id'] for s in trace['steps']}:
                sub = dict(trace, steps=[s for s in trace['steps'] if s['request_id'] == rid],
                           rows=[r for r in trace['rows'] if r['request_id'] == rid])
                if args.granularity == 'operator':
                    sub['module_rows'] = [r for r in trace['module_rows'] if r['request_id'] == rid]
                    sub['operator_steps'] = [r for r in trace['operator_steps'] if r['request_id'] == rid]
                check_trace(sub, data)
            if args.command == 'trace':
                native_path = (root.parent if args.granularity == 'operator' else root) / 'smoke_latest.json'
                native = load_json(native_path)
                hashes = {r['client']['output_sha256'] for r in native['results']}
                if hashes != {r['output_sha256'] for r in results}:
                    raise RuntimeError('Trace output differs from native smoke reference')
                trace['native_output_sha256'] = next(iter(hashes))
            if diagnostic:
                trace['output_sha256'] = result['output_sha256']
                events = []
                files = list((out/'torch_trace').rglob('*.pt.trace.json*'))
                if not files:
                    raise RuntimeError('No worker profiler trace found; operator DVFS remains disabled')
                for file in files:
                    events.extend(read_chrome(file))
                certificate = build_certificate(trace, events)
                certificate['source_trace_files'] = [str(f) for f in files]
                certificate.pop('certificate_sha256')
                certificate['certificate_sha256'] = digest_json(certificate)
                write_json(root/'coverage.json', certificate)
            write_json(root / f'{args.command}.json', dict(result=result, measurement=measured,
                       worker_trace=trace, run_dir=str(out), command=args.command))
    finally:
        try:
            rpc(llm, 'sdvfs_export', close=True)
        finally:
            llm.llm_engine.engine_core.shutdown()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['workload', 'trace', 'profile', 'offline'])
    p.add_argument('--run-dir', required=True)
    p.add_argument('--hardware')
    p.add_argument('--input-tokens', type=int, default=512)
    p.add_argument('--output-tokens', type=int, default=128)
    p.add_argument('--repeat', type=int, default=3)
    p.add_argument('--grid-file')
    p.add_argument('--mode', choices=['instrumented', 'fixed', 'serial', 'lookahead'], default='instrumented')
    p.add_argument('--plan')
    p.add_argument('--granularity', choices=['module', 'operator'], default='module')
    p.add_argument('--coverage')
    p.add_argument('--switch-repeat', type=int, default=10)
    p.add_argument('--backend', choices=['torch'], default='torch')
    p.add_argument('--apply', action='store_true')
    args = p.parse_args()
    if args.input_tokens < 1 or args.input_tokens + args.output_tokens > 4096 or args.output_tokens < 2 or args.repeat < 1:
        p.error('Require input+output <=4096, output>=2 and repeat>=1')
    try:
        execute(args)
    except BaseException as exc:
        failure = dict(error=repr(exc), oom='out of memory' in str(exc).lower(),
                       action='Stopped; no quantization/model fallback was attempted')
        try:
            import torch
            if torch.cuda.is_available():
                failure['cuda_memory_summary'] = torch.cuda.memory_summary()
        except Exception as detail:
            failure['memory_report_error'] = str(detail)
        write_json(Path(args.run_dir) / 'failure.json', failure)
        raise


if __name__ == '__main__':
    main()
