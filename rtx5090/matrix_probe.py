"""Single-GPU synthetic kernel-name probe; never calibration or same-path proof."""
import argparse
from collections import Counter
import os
from pathlib import Path
import subprocess

from .observe_core import GPU_UUIDS, digest
from .service_core import read, write_new
from .service_client import sha
from .performance import safe_error


def candidates(rows, rank):
    selected = {}
    for row in rows:
        modules = {p.rsplit('.', 1)[-1] for p in row['parents']
                   if p.endswith(('.mlp.gate_up_proj', '.mlp.down_proj'))}
        if not modules:
            continue
        s = row['signature']
        if len(modules) != 1 or s['rank'] != rank or row['signature_id'] != digest(s):
            raise ValueError('Signature rank/hash/module mismatch')
        module = next(iter(modules))
        phase = s['phase']
        if phase not in ('prefill', 'decode') or s['overload'] != 'aten.linear.default' or row['opaque']:
            raise ValueError('Only observed nonopaque aten.linear is supported')
        m = 512 if phase == 'prefill' else 1
        n, k = (17408, 5120) if module == 'gate_up_proj' else (5120, 8704)
        args = s['inputs']['args']
        if len(args) != 2 or s['inputs']['kwargs']:
            raise ValueError('Bias/extra arguments require separate audit')
        for meta, shape in zip([*args, s['outputs']], ([m, k], [n, k], [m, n])):
            if (meta.get('tensor') is not True or meta['shape'] != shape
                    or meta['stride'] != [shape[1], 1] or meta['dtype'] != 'torch.bfloat16'
                    or meta['device'] != f'cuda:{rank}' or meta['layout'] != 'torch.strided'):
                raise ValueError('Shape/stride/dtype/device contract differs')
        if not s['kernel_names'] or not all(isinstance(x, str) for x in s['kernel_names']):
            raise ValueError('No expected kernel names')
        key = (phase, module)
        if key in selected:
            raise ValueError('Multiple kernel variants; do not choose silently')
        selected[key] = dict(row=row, m=m, n=n, k=k, phase=phase, module=module)
    if len(selected) != 4:
        raise ValueError('Require both phases and both MLP projections')
    return [selected[k] for k in sorted(selected)]


def compare_kernels(events, expected):
    counts = Counter(e['name'] for e in events if e.get('cat') == 'kernel' and e.get('ph') == 'X')
    actual = set(counts)
    return dict(kernel_name_set_match=bool(actual) and actual == set(expected),
                actual_kernel_counts=dict(counts), missing=sorted(set(expected)-actual),
                unexpected=sorted(actual-set(expected)),
                launch_order_and_multiplicity_verified=False)


def free_mib(uuid):
    result = subprocess.run(['nvidia-smi', '-i', uuid, '--query-gpu=uuid,memory.free',
        '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=10)
    values = result.stdout.strip().split(',')
    if len(values) != 2 or values[0].strip() != uuid:
        raise ValueError('GPU UUID query mismatch')
    return int(values[1].strip())


def run(args):
    from . import eager
    if os.name != 'posix':
        raise ValueError('Use target WSL, not development Windows')
    import fcntl
    identity = read(args.identity)
    with (Path(identity['runtime'])/'client.lock').open('a') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = eager.checked(args.identity)
        trace = Path(args.trace_dir).resolve()
        observation = read(trace/'observation.json')
        if (observation['status'] != 'complete' or observation['identity_fingerprint'] != identity['fingerprint']
                or not observation['coverage']['passed']):
            raise ValueError('Trace belongs to a different/incomplete service observation')
        files = list(trace.glob(f'rank-{args.rank}-matrix-signatures-*.json'))
        if not files:
            raise ValueError('Missing saved matrix signatures')
        source = max(files, key=lambda p: p.stat().st_mtime_ns)
        jobs = candidates(read(source), args.rank)
        out = Path(args.out).resolve()
        out.mkdir(parents=True, exist_ok=False)
        result = dict(status='failed', rank=args.rank, uuid=GPU_UUIDS[args.rank],
            identity_fingerprint=identity['fingerprint'], source=str(source), source_sha256=sha(source),
            observation_sha256=sha(trace/'observation.json'), commit=eager.git('rev-parse', 'HEAD'),
            probe_sha256=sha(__file__), clock_writes=False, synthetic_tensor_values=True,
            cache_state_matched=False, same_path_verified=False, calibration_ready=False,
            scope='single_gpu_kernel_name_set_only_not_TP_execution_or_timing', cases=[])
        try:
            uuid = result['uuid']
            result['free_mib_before_context'] = free_mib(uuid)
            if result['free_mib_before_context'] < 1536:
                raise RuntimeError('Require 1536 MiB free before creating probe CUDA context; no allocation attempted')
            os.environ['CUDA_VISIBLE_DEVICES'] = uuid
            import torch
            if torch.cuda.device_count() != 1:
                raise RuntimeError('Probe must see exactly one UUID-selected GPU')
            props = torch.cuda.get_device_properties(0)
            actual_uuid = str(getattr(props, 'uuid', ''))
            if actual_uuid.removeprefix('GPU-').lower() != uuid.removeprefix('GPU-').lower():
                raise RuntimeError('PyTorch physical GPU UUID verification failed')
            result['torch'] = str(torch.__version__)
            result['cuda'] = torch.version.cuda
            worker = identity['ranks'][args.rank]
            if str(torch.__version__) != worker['torch_version'] or torch.version.cuda != worker['torch_cuda']:
                raise ValueError('Probe/worker PyTorch or CUDA version differs')
            result['free_mib_after_context'] = torch.cuda.mem_get_info()[0] // 2**20
            result['nvml_free_mib_after_context'] = free_mib(uuid)
            result['conservative_free_mib_after_context'] = min(result['free_mib_after_context'],
                                                               result['nvml_free_mib_after_context'])
            result['memory_sources_disagree'] = abs(result['free_mib_after_context']-
                                                     result['nvml_free_mib_after_context']) > 256
            if result['conservative_free_mib_after_context'] < 1024:
                raise RuntimeError('Require both NVML and CUDA to report 1024 MiB free after context initialization')
            torch.manual_seed(110601)
            for job in jobs:
                key = job['phase']+'-'+job['module']
                with torch.inference_mode():
                    x = torch.empty((job['m'], job['k']), dtype=torch.bfloat16, device='cuda').normal_(0, .02)
                    w = torch.empty((job['n'], job['k']), dtype=torch.bfloat16, device='cuda').normal_(0, .02)
                    # Warm library dispatch outside the one-call profiler capture.
                    y = torch.ops.aten.linear.default(x, w)
                    torch.cuda.synchronize()
                    del y
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                          torch.profiler.ProfilerActivity.CUDA]) as prof:
                        y = torch.ops.aten.linear.default(x, w)
                        torch.cuda.synchronize()
                    if list(y.shape) != [job['m'], job['n']] or list(y.stride()) != [job['n'], 1]:
                        raise ValueError('Probe output layout differs')
                    path = out/(key+'-trace.json')
                    prof.export_chrome_trace(str(path))
                    match = compare_kernels(read(path)['traceEvents'], job['row']['signature']['kernel_names'])
                    row = dict(phase=job['phase'], module=job['module'],
                               signature_id=job['row']['signature_id'], **match)
                    result['cases'].append(row)
                    write_new(out/(key+'.json'), row)
                    del prof, y, x, w
            eager.checked(args.identity)
            result['status'] = ('kernel_names_matched_only' if all(
                r['kernel_name_set_match'] for r in result['cases']) else 'kernel_name_mismatch')
        except BaseException as exc:
            result['error'] = safe_error(exc)
            raise
        finally:
            write_new(out/'probe.json', result)
        return result


def main():
    import json
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--identity', required=True)
    p.add_argument('--trace-dir', required=True)
    p.add_argument('--rank', type=int, choices=(0, 1), required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    try:
        print(json.dumps(run(args), ensure_ascii=False, indent=2))
    except Exception as exc:
        print(json.dumps(dict(status='failed', error=safe_error(exc)), ensure_ascii=False))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
