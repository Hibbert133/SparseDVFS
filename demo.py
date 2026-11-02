#!/usr/bin/env python3
"""v5 CLI alias to the vLLM host; legacy pure helpers remain for regression tests."""
import argparse
import hashlib
import json
import math
import os
import signal
import statistics
import sys
import threading
import time
from pathlib import Path
from sparsedvfs.hardware import load_json, write_json

ROOT = Path(__file__).resolve().parent


def add_model(parser):
    parser.add_argument('--model', required=True, help='local Qwen3.5-4B directory or cached hub ID')
    parser.add_argument('--prompt-file', default=str(ROOT / 'configs/prompt_zh.txt'))
    parser.add_argument('--tokens', type=int, default=32)
    parser.add_argument('--dtype', choices=['bfloat16', 'float16', 'float32'], default='bfloat16')
    parser.add_argument('--attention', choices=['eager', 'sdpa'], default='eager')
    parser.add_argument('--allow-download', action='store_true', help='explicitly allow target-side HF downloads')
    parser.add_argument('--revision')
    parser.add_argument('--thinking', action='store_true')
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=3)


def add_control(parser):
    parser.add_argument('--hardware', required=True)
    parser.add_argument('--apply', action='store_true', help='write target frequency interfaces; otherwise no writes')
    parser.add_argument('--library', default=str(ROOT / 'build/libfuse_dvfs.so'))


def create_runner(args):
    from sparsedvfs.qwen_runner import Runner
    return Runner(args.model, Path(args.prompt_file).read_text(encoding='utf-8'), args.tokens,
                  args.dtype, args.attention, args.allow_download, args.revision, args.thinking)


def ensure_apply(args):
    if not args.apply:
        raise ValueError('This measurement requires real clock control. On the Thor target, explicitly add --apply; no clocks were written.')
    if not hasattr(os, 'geteuid') or os.geteuid() != 0:
        raise RuntimeError('--apply requires root on the Thor host')


def validate_grid(hardware, grid):
    domains = [hardware['cpu_common_khz'], hardware['gpu']['frequencies'], hardware['emc']['frequencies']]
    if not grid.get('triplets') or len({tuple(f) for f in grid['triplets']}) != len(grid['triplets']):
        raise ValueError('empty grid or duplicate triplets')
    for freq in grid['triplets']:
        if len(freq) != 3 or any(f not in allowed for f, allowed in zip(freq, domains)):
            raise ValueError(f'OPP outside probed hardware limits: {freq}')
    reference = [max(f[i] for f in grid['triplets']) for i in range(3)]
    if grid['reference'] != reference or reference not in grid['triplets']:
        raise ValueError('grid reference must be the componentwise maximum, present in the grid')


def command_probe(args):
    from sparsedvfs.hardware import probe, make_grid
    cuda = None
    if args.cuda_check:
        from sparsedvfs.qwen_runner import cuda_check
        cuda = cuda_check()
    rates = [int(v) for v in args.emc_rates_hz.split(',') if v] if args.emc_rates_hz else []
    hardware = probe(emc_rates=rates, gpu_path=args.gpu_path, allow_no_emc=args.allow_no_emc)
    if args.power_backend == 'jtop':
        hardware['power'] = {'kind': 'jtop', 'rail': 'tot', 'scope': 'jtop_total_driver_defined',
                             'units': 'mW', 'auto_install_or_service_restart': False}
    hardware['cuda'] = cuda
    write_json(args.output, hardware)
    print(f'Read-only hardware report: {args.output}')
    print(f'CPU policies: {len(hardware["cpu"])}; GPC: {hardware["gpu"]["path"]}; EMC: {hardware["emc"]["backend"]}')
    print('Power source: ' + json.dumps(hardware['power'], ensure_ascii=False))
    if args.grid_output:
        write_json(args.grid_output, make_grid(hardware, args.points))
        print(f'Cartesian frequency grid: {args.grid_output}')


def command_switch(args):
    ensure_apply(args)
    from sparsedvfs.qwen_runner import cuda_check
    from sparsedvfs.hardware import wait_frequencies
    from sparsedvfs.fuse import FUSE
    cuda = cuda_check()
    import torch
    hardware, grid = load_json(args.hardware), load_json(args.grid)
    validate_grid(hardware, grid)
    if args.repeats < 1 or len(grid['triplets']) < 2:
        raise ValueError('switch measurement requires >=2 OPPs and repeats >=1')
    out = Path(args.output)
    errors, transitions = [], []
    stop = threading.Event()
    def load_gpu():
        try:
            a = torch.ones((256, 256), device='cuda', dtype=torch.bfloat16)
            while not stop.is_set():
                _ = a @ a
                torch.cuda.synchronize()
        except Exception as exc:
            errors.append(str(exc)); stop.set()
    thread = threading.Thread(target=load_gpu, daemon=True)
    thread.start()
    try:
        with FUSE(hardware, out.parent / (out.stem + '_controller'), True, args.library) as controller:
            previous = grid['reference']
            controller.set(previous)
            wait_frequencies(hardware, previous, args.timeout)
            # Include every directed pair; the default 2x2x2 grid is still small.
            for rep in range(args.repeats):
                for source in grid['triplets']:
                    for target in grid['triplets']:
                        if source == target:
                            continue
                        controller.set(source)
                        wait_frequencies(hardware, source, args.timeout)
                        start = time.monotonic_ns()
                        controller.set(target)
                        result = wait_frequencies(hardware, target, args.timeout)
                        total = (time.monotonic_ns() - start) / 1000
                        transitions.append({'from': source, 'to': target, 'repeat': rep,
                                            'write_us': controller.lib.fuse_last_write_us(controller.handle),
                                            'observed_request_to_readback_us': total, **result})
                print(f'Switch calibration pass {rep + 1}/{args.repeats}', flush=True)
            if errors:
                raise RuntimeError('CUDA keepalive failed: ' + '; '.join(errors))
    finally:
        stop.set(); thread.join()
    write_json(out, {'schema': 1, 'dvfs_applied': True, 'cuda': cuda,
                     'triplets': grid['triplets'], 'degraded_no_emc': hardware['degraded_no_emc'],
                     'recommended_switch_us': max(r['observed_request_to_readback_us'] for r in transitions),
                     'definition': 'maximum host request-to-driver-readback duration under light GPU load; not physical PLL instrumentation',
                     'transitions': transitions})
    print(f'Switch calibration saved: {out}')


def command_profile(args):
    ensure_apply(args)
    from sparsedvfs.fuse import FUSE
    from sparsedvfs.qwen_runner import aggregate_profile
    hardware, grid = load_json(args.hardware), load_json(args.grid)
    validate_grid(hardware, grid)
    runner = create_runner(args)
    result = aggregate_profile(runner, hardware, grid,
                               lambda d: FUSE(hardware, d, True, args.library),
                               Path(args.output).parent / (Path(args.output).stem + '_raw'),
                               args.repeats, args.warmups, args.stability_s)
    result['hardware_context'] = hardware
    write_json(args.output, result)
    print(f'Measured profile: {args.output}')


def command_plan(args):
    from sparsedvfs.graph_partitioner import build_plan
    profile = load_json(args.profile)
    switch_us = args.switch_us
    if args.switch_bench:
        switch = load_json(args.switch_bench)
        if not switch.get('dvfs_applied'):
            raise ValueError('switch calibration was not measured with applied clocks')
        if {tuple(f) for f in switch['triplets']} != {tuple(c['triplet']) for c in profile['candidates']}:
            raise ValueError('switch calibration and profile use different OPP grids')
        switch_us = switch['recommended_switch_us']
    if switch_us is None:
        raise ValueError('provide --switch-bench measured.json or your own measured --switch-us')
    plan = build_plan(profile, switch_us, args.factor, args.tolerance, args.max_slowdown,
                      args.selector, load_json(args.calibration) if args.calibration else None,
                      args.allow_synthetic, args.memory_policy)
    if 'hardware_context' in profile:
        plan['hardware_context'] = profile['hardware_context']
    write_json(args.output, plan)
    print(f'{len(plan["blocks"])} super-blocks saved: {args.output}; synthetic={plan["synthetic"]}')


def command_run(args):
    from sparsedvfs.fuse import FUSE
    from sparsedvfs.hardware import wait_frequencies, frequencies
    from sparsedvfs.qwen_runner import Schedule
    if args.repeats < 1 or args.warmups < 0:
        raise ValueError('invalid repeats/warmups')
    hardware = load_json(args.hardware)
    if args.apply:
        ensure_apply(args)
    if args.mode == 'baseline' and args.apply:
        raise ValueError('baseline does not set clocks; omit --apply or select --mode fixed')
    plan = load_json(args.plan) if args.plan else None
    grid = load_json(args.grid) if args.grid else None
    if args.mode in ('serial', 'lookahead'):
        if plan is None or plan.get('synthetic'):
            raise ValueError('serial/lookahead requires a real, non-synthetic --plan')
        if plan.get('hardware_context') != hardware:
            raise ValueError('hardware report differs from the profiled board report; reprofile or use the original hardware.json')
        if args.mode == 'lookahead' and args.transport != 'cuda':
            raise ValueError('lookahead requires --transport cuda in v2')
    if (args.runtime_profile or args.thermal_calibration) and args.mode not in ('serial', 'lookahead'):
        raise ValueError('runtime partition/thermal feedback applies to serial/lookahead modes')
    runtime_policy = None
    if args.thermal_calibration and not args.runtime_profile:
        raise ValueError('--thermal-calibration requires --runtime-profile')
    if args.runtime_profile:
        from sparsedvfs.runtime_policy import RuntimePlanner
        runtime_profile = load_json(args.runtime_profile)
        if runtime_profile.get('hardware_context') != hardware:
            raise ValueError('runtime profile uses a different hardware report')
        runtime_policy = RuntimePlanner(runtime_profile, plan, hardware,
                                        load_json(args.thermal_calibration) if args.thermal_calibration else None,
                                        args.memory_policy)
    if args.mode == 'fixed':
        if not grid:
            raise ValueError('fixed mode requires --grid')
        validate_grid(hardware, grid)
    runner = create_runner(args)
    if plan and args.mode in ('serial', 'lookahead') and runner.identity != plan['identity']:
        raise ValueError('Model/prompt/tokens/dtype/backend identity differs from the profile. Reprofile this workload.')
    directory = Path(args.output).parent / (Path(args.output).stem + '_controller')
    results = []
    if args.mode == 'baseline':
        runner.warmup(args.warmups)
        results = [runner.generate(hardware) for _ in range(args.repeats)]
    else:
        native_dry = args.mode in ('serial', 'lookahead') and args.transport == 'cuda' and not args.apply
        with FUSE(hardware, directory, args.apply, args.library, native_dry=native_dry) as controller:
            if args.mode in ('serial', 'lookahead') and args.transport == 'cuda':
                controller.enable_cuda(runner.torch.cuda.current_device(), args.poll_us, runner.torch.version.cuda)
            if args.mode == 'fixed':
                controller.set(grid['reference'])
                if args.apply:
                    runner.settle(hardware, grid['reference'])
                runner.warmup(args.warmups)
                results = [runner.generate(hardware) for _ in range(args.repeats)]
            else:
                # Warm up at the fixed reference; timed runs include runtime control overhead.
                controller.boost_khz = plan['reference'][0]
                controller.set(plan['reference'])
                if args.apply:
                    runner.settle(hardware, plan['reference'])
                runner.warmup(args.warmups)
                for _ in range(args.repeats):
                    schedule = Schedule(plan, controller, args.mode, args.transport, runtime_policy)
                    results.append(runner.generate(hardware, schedule=schedule))
            observed = frequencies(hardware)
    write_json(args.output, {'schema': 1, 'identity': runner.identity, 'cuda': runner.cuda_info,
                             'mode': args.mode, 'dvfs_applied': args.apply,
                             'transport': args.transport if args.mode in ('serial', 'lookahead') else None,
                             'hardware_context': hardware, 'plan': plan,
                             'power_source': hardware.get('power'), 'runs': results,
                             'observed_at_end': frequencies(hardware) if args.mode == 'baseline' else observed})
    print(f'{args.mode}: mean TTFT={statistics.mean(r["ttft_ms"] for r in results):.3f} ms; '
          f'TPOT={statistics.mean(r["tpot_ms"] for r in results):.3f} ms; applied={args.apply}')
    print(f'Results: {args.output}')
    print(results[-1]['text'])


def summarize_runs(document):
    runs = document['runs']
    if not runs:
        raise ValueError('empty run document')
    summary = {k: statistics.median(r[k] for r in runs) for k in ('elapsed_ms', 'ttft_ms', 'tpot_ms')}
    values = [r['power']['energy_j'] for r in runs]
    summary['energy_j'] = statistics.median(values) if all(v is not None and math.isfinite(v) for v in values) else None
    return summary


def compare_documents(baseline, variant):
    if baseline['identity'] != variant['identity']:
        raise ValueError('Cannot compare different model/prompt/length/dtype/library identities')
    if baseline.get('power_source') != variant.get('power_source'):
        raise ValueError('Cannot compare different power rails or measurement scopes')
    for d in (baseline, variant):
        if d['mode'] != 'baseline' and not d.get('dvfs_applied'):
            raise ValueError('A control run did not apply DVFS. No hardware energy claim is valid.')
    a, b = summarize_runs(baseline), summarize_runs(variant)
    tokens = [r['generated_token_ids'] for d in (baseline, variant) for r in d['runs']]
    equal = all(t == tokens[0] for t in tokens)
    report = {'schema': 1, 'baseline_mode': baseline['mode'], 'variant_mode': variant['mode'],
              'baseline_medians': a, 'variant_medians': b, 'output_token_ids_equal': equal,
              'latency_change_pct': (b['elapsed_ms'] / a['elapsed_ms'] - 1) * 100,
              'energy_saving_pct': None, 'edp_change_pct': None,
              'valid_energy_comparison': equal and a['energy_j'] is not None and b['energy_j'] is not None}
    if report['valid_energy_comparison'] and a['energy_j'] > 0:
        report['energy_saving_pct'] = (1 - b['energy_j'] / a['energy_j']) * 100
        report['edp_change_pct'] = (b['energy_j'] * b['elapsed_ms'] / (a['energy_j'] * a['elapsed_ms']) - 1) * 100
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    probe = sub.add_parser('probe', help='read-only hardware/telemetry discovery')
    probe.add_argument('--output', required=True)
    probe.add_argument('--cuda-check', action='store_true')
    probe.add_argument('--gpu-path')
    probe.add_argument('--emc-rates-hz', help='comma-separated OPPs verified for this BSP/nvpmodel')
    probe.add_argument('--allow-no-emc', action='store_true')
    probe.add_argument('--power-backend', choices=['hwmon', 'jtop'], default='hwmon')
    probe.add_argument('--grid-output')
    probe.add_argument('--points', type=int, default=2)
    probe.set_defaults(func=command_probe)
    switch = sub.add_parser('switch-bench', help='measure directed OPP transitions on Thor')
    add_control(switch)
    switch.add_argument('--grid', required=True)
    switch.add_argument('--output', required=True)
    switch.add_argument('--repeats', type=int, default=3)
    switch.add_argument('--timeout', type=float, default=0.5)
    switch.set_defaults(func=command_switch)
    profile = sub.add_parser('profile', help='offline module profiling; sparsity measured separately')
    add_control(profile); add_model(profile)
    profile.add_argument('--grid', required=True)
    profile.add_argument('--output', required=True)
    profile.add_argument('--stability-s', type=float, default=0.2)
    profile.set_defaults(func=command_profile)
    plan = sub.add_parser('plan', help='offline selector and Algorithm 1 graph partitioner')
    plan.add_argument('--profile', required=True)
    plan.add_argument('--output', required=True)
    source = plan.add_mutually_exclusive_group()
    source.add_argument('--switch-bench')
    source.add_argument('--switch-us', type=float)
    plan.add_argument('--factor', type=float, default=5)
    plan.add_argument('--tolerance', type=float, default=0.05)
    plan.add_argument('--max-slowdown', type=float, default=0.05)
    plan.add_argument('--selector', choices=['measured', 'paper'], default='measured')
    plan.add_argument('--calibration')
    plan.add_argument('--allow-synthetic', action='store_true')
    plan.add_argument('--memory-policy', choices=['calibrated', 'paper'], default='calibrated')
    plan.set_defaults(func=command_plan)
    run = sub.add_parser('run', help='text generation with baseline/fixed/serial/lookahead control')
    add_control(run); add_model(run)
    run.add_argument('--mode', choices=['baseline', 'fixed', 'serial', 'lookahead'], default='baseline')
    run.add_argument('--plan')
    run.add_argument('--grid')
    run.add_argument('--transport', choices=['cuda', 'host'], default='cuda')
    run.add_argument('--poll-us', type=int, default=100)
    run.add_argument('--runtime-profile', help='repartition at each prefill/decode forward')
    run.add_argument('--thermal-calibration', help='complete calibration with sensor and valid temperature range')
    run.add_argument('--memory-policy', choices=['calibrated', 'paper'], default='calibrated')
    run.add_argument('--output', required=True)
    run.set_defaults(func=command_run)
    compare = sub.add_parser('compare')
    compare.add_argument('--baseline', required=True)
    compare.add_argument('--variant', required=True)
    compare.add_argument('--output', required=True)
    compare.set_defaults(func=lambda a: write_json(a.output, compare_documents(load_json(a.baseline), load_json(a.variant))))
    restore = sub.add_parser('restore', help='explicit emergency restoration from a fresh run snapshot')
    restore.add_argument('--snapshot', required=True)
    restore.add_argument('--apply', action='store_true')
    def do_restore(a):
        ensure_apply(a)
        from sparsedvfs.hardware import restore
        import fcntl
        with open('/run/lock/sparsedvfs-thor.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            restore(load_json(a.snapshot))
        print('Saved clock constraints restored.')
    restore.set_defaults(func=do_restore)
    return p


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, interrupted)
    try:
        from sparsedvfs_vllm.host import main
        main()
    except KeyboardInterrupt:
        print('Interrupted; controller context attempted restoration.', file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
