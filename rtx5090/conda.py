"""Manual, one-shot Conda entry. No installations or persistent agent needed."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from .native import ROOT, DEFAULT_GPU, configure_process, doctor, build, check_vllm_sources
from .run import write, read


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['doctor', 'build', 'cpu-test', 'download', 'dataset', 'smoke',
                   'baseline', 'chain', 'probe', 'calibrate', 'profile', 'plan', 'run', 'tune', 'compare', 'report', 'recover', '_window'])
    p.add_argument('--model', default='/mnt/d/models/Qwen3.5-27B-AWQ-4bit')
    p.add_argument('--gpu-uuid', default=DEFAULT_GPU)
    p.add_argument('--out', default='runs/conda')
    p.add_argument('--library-dir', default=str(ROOT/'build'))
    p.add_argument('--gpu', help='GPU backend JSON, UUID must equal --gpu-uuid')
    p.add_argument('--backend', choices=['nvml', 'bridge', 'mock'], default='nvml')
    p.add_argument('--apply', action='store_true')
    p.add_argument('--ack-default-auto', action='store_true')
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--cuda-include'); p.add_argument('--cudart')
    p.add_argument('--dataset'); p.add_argument('--calibration'); p.add_argument('--profile')
    p.add_argument('--plan'); p.add_argument('--selection'); p.add_argument('--baseline')
    p.add_argument('--mode', choices=['auto', 'instrumented', 'fixed', 'serial', 'lookahead'], default='serial')
    p.add_argument('--repeat', type=int, default=3)
    p.add_argument('--pairs', type=int, default=10)
    p.add_argument('--split', choices=['train', 'test'], default='train')
    p.add_argument('--requests', type=int, default=0)
    p.add_argument('--min-seconds', type=float, default=30)
    p.add_argument('--mhz', type=int)
    p.add_argument('--power-evidence', help='Reviewed telemetry accuracy evidence; needed only for power integration fallback')
    p.add_argument('--job', help=argparse.SUPPRESS)
    return p


def main():
    p = parser(); a = p.parse_args()
    out = Path(a.out).resolve(); a.out = str(out)
    required = {
        'baseline': ('dataset',), 'chain': ('dataset','baseline'),
        'calibrate': ('gpu',), 'profile': ('gpu','dataset','baseline','calibration'),
        'plan': ('profile',), 'run': ('gpu','dataset','baseline','calibration'),
        'tune': ('gpu','dataset','baseline','calibration','profile'),
        'compare': ('gpu','dataset','baseline','calibration','selection'), 'recover': ('gpu',),
        '_window': ('job','dataset','baseline'),
    }
    for key in required.get(a.command, ()):
        if not getattr(a,key): p.error(f'{a.command} requires --{key}')
    artifacts = dict(baseline='baseline.json',calibrate='calibration.json',profile='profile.json',
                     tune='tuning.json',compare='pairs.json',chain='result.json',run='result.json',_window='result.json')
    if a.command in artifacts and (out/artifacts[a.command]).exists():
        p.error('Output already contains evidence; use a fresh --out directory')
    if a.command == 'doctor':
        result = doctor(a.model); write(out/'doctor.json', result)
        print(json.dumps(result, indent=2)); return
    if a.command == 'cpu-test':
        build(a.library_dir)
        if Path(a.library_dir).resolve() != ROOT/'build':
            p.error('CPU regression suite requires the default project build directory')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
        for key in ('SPARSEDVFS_GPU_UUID','SPARSEDVFS_APPLY','SPARSEDVFS_WORKER_CONFIG'):
            env.pop(key, None)
        subprocess.run([sys.executable, '-B', str(ROOT/'run_virtual_validation.py'), '--output', str(out)], check=True, env=env)
        result=read(out/'summary.json')
        if not result['native_core_library_present'] or any(
                'test_cuda_fuse.' not in row['test'] and 'test_v4_native.NativeCudaABI.' not in row['test'] for row in result['skipped']):
            raise RuntimeError('A required CPU/native test was skipped')
        return
    if a.command == 'build':
        print(json.dumps(build(a.library_dir, a.cuda, a.cuda_include, a.cudart), indent=2)); return
    if a.command == 'download':
        from .run import download
        download(a.model); return
    configure_process(a.model, a.gpu_uuid, a.library_dir)
    if a.gpu:
        cfg = read(a.gpu)
        if cfg['uuid'] != a.gpu_uuid:
            p.error('--gpu UUID differs from --gpu-uuid')
    if a.command in ('dataset', 'smoke', 'baseline', 'chain', 'profile', 'run', 'tune', 'compare', '_window'):
        check_vllm_sources()
    from .experiment import dispatch
    dispatch(a)


if __name__ == '__main__':
    main()
