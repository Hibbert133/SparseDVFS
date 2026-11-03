"""Conda process setup, audited vLLM sources and local-only native builds."""
import ctypes
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GPU = 'GPU-22d1cd22-d198-45c4-2b54-974484026ab5'
V032_SOURCES = {
    'v1/worker/gpu_worker.py': 'e970aa29a738aa15cd963d9f9202ca6d3d504e98fd344959fd998e940472a5a0',
    'v1/worker/gpu_model_runner.py': '4cf95efe4ea8caa3110dcd231ff95ceb97e8136f5056e6bdd388300a484c4a81',
    'v1/worker/gpu/model_runner.py': '5240fe8d62d403e466a4aa07a28fb7af5c0f1e9b9c3bb08c4003185bde461f87',
    'entrypoints/llm.py': '283f5ed4d0744aef22a01fefd307565ef60ee172a8ec4805ece0bcd630d9568c',
    'config/vllm.py': '573692a80364d5c373321754c6213b4792ae292ba6aea7d6573ae013c2c71ae9',
    'v1/core/sched/output.py': '2a3454c78ba1e7ffcb96cb2839cea900a662829d495ed3d411a731aaccc1ba00',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment_record():
    packages = {}
    for name in ('vllm', 'torch', 'triton', 'transformers'):
        try:
            spec = importlib.util.find_spec(name)
            packages[name] = dict(version=importlib.metadata.version(name), path=spec.origin)
        except (ImportError, ValueError):
            packages[name] = None
    library = Path(os.environ.get('SPARSEDVFS_LIBRARY_DIR', ROOT/'build'))
    try:
        commit = subprocess.run(['git','-C',str(ROOT),'rev-parse','HEAD'],capture_output=True,text=True,timeout=5)
        commit = commit.stdout.strip() if commit.returncode == 0 else None
    except (OSError,subprocess.TimeoutExpired): commit = None
    sources = {str(p.relative_to(ROOT)):sha(p) for folder in ('rtx5090','cpp','sparsedvfs','sparsedvfs_vllm')
               for p in (ROOT/folder).glob('*') if p.suffix in ('.py','.cpp','.h')}
    return dict(python=sys.version, executable=sys.executable, packages=packages,
                git_commit=commit, sources_sha256=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest(),
                libraries={p.name: sha(p) for p in library.glob('libfuse*.so')},
                env={k: os.environ.get(k) for k in ('CONDA_PREFIX', 'PYTHONPATH', 'TRITON_CACHE_DIR',
                     'CUDA_VISIBLE_DEVICES', 'VLLM_USE_V2_MODEL_RUNNER', 'SPARSEDVFS_LIBRARY_DIR')})


def check_vllm_sources():
    version = importlib.metadata.version('vllm').split('+')[0]
    if version != '0.32.0':
        raise RuntimeError('Conda entry requires audited vLLM 0.32.0; no automatic downgrade')
    spec = importlib.util.find_spec('vllm')
    root = Path(spec.origin).parent
    actual = {name: sha(root/name) for name in V032_SOURCES}
    if actual != V032_SOURCES:
        raise RuntimeError('Installed vLLM source differs from audited 0.32.0: ' + json.dumps(actual))
    return actual


def configure_process(model, uuid, library_dir):
    if not uuid.startswith('GPU-') or ',' in uuid:
        raise ValueError('Select exactly one full GPU UUID')
    if '/tmp/triton-360-test' in os.environ.get('PYTHONPATH', ''):
        raise RuntimeError('Temporary Triton override is active; run from a clean terminal (do not replace Conda Triton)')
    os.environ.update(CUDA_VISIBLE_DEVICES=uuid, VLLM_USE_V2_MODEL_RUNNER='0',
                      SPARSEDVFS_GPU_UUID=uuid, SPARSEDVFS_MODEL=str(Path(model).resolve()),
                      SPARSEDVFS_LIBRARY_DIR=str(Path(library_dir).resolve()))
    # Spawned workers must import this checkout, not another installed copy.
    old = [p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) if p and p != str(ROOT)]
    os.environ['PYTHONPATH'] = os.pathsep.join([str(ROOT)] + old)
    sys.path.insert(0, str(ROOT))


def assert_cuda_uuid(torch, expected):
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one CUDA device must be visible')
    value = getattr(torch.cuda.get_device_properties(0), 'uuid', None)
    if value is None:
        raise RuntimeError('CUDA device UUID unavailable; cannot prove inference/control binding')
    actual = str(value)
    if not actual.startswith('GPU-'):
        actual = 'GPU-' + actual
    if actual.lower() != expected.lower():
        raise RuntimeError(f'Inference/control UUID mismatch: {actual} != {expected}')
    return actual


def cuda_paths():
    roots = [Path(sys.prefix)]
    roots += [Path(p) for p in sys.path if p and Path(p).is_dir() and 'site-packages' in p]
    roots += [Path(os.environ[k]) for k in ('CUDA_HOME', 'CUDA_PATH') if os.environ.get(k)]
    if Path('/usr/local/cuda').exists():
        roots.append(Path('/usr/local/cuda'))
    headers, libraries = [], []
    for root in roots:
        for pattern in ('include/cuda_runtime_api.h', 'targets/*/include/cuda_runtime_api.h',
                        'nvidia/*/include/cuda_runtime_api.h', 'triton/backends/nvidia/include/cuda_runtime_api.h'):
            headers.extend(p.parent for p in root.glob(pattern))
        for pattern in ('lib/libcudart.so*', 'lib64/libcudart.so*', 'targets/*/lib/libcudart.so*',
                        'nvidia/*/lib/libcudart.so*'):
            libraries.extend(p for p in root.glob(pattern) if p.is_file())
    return sorted(set(headers)), sorted(set(libraries))


def build(directory, cuda=False, include=None, cudart=None):
    if sys.platform != 'linux':
        raise RuntimeError('Run this build inside WSL/Linux, using the target Conda Python')
    compiler = shutil.which('g++')
    if not compiler:
        raise RuntimeError('g++ missing; no packages were installed')
    directory = Path(directory).resolve(); directory.mkdir(parents=True, exist_ok=True)
    commands = [[compiler, '-std=c++17', '-O2', '-Wall', '-Wextra', '-Wpedantic', '-fPIC',
                 '-shared', '-pthread', str(ROOT/'cpp/FUSE.cpp'), '-o', str(directory/'libfuse_dvfs.so')]]
    if cuda:
        headers, libraries = cuda_paths()
        header = Path(include) if include else (headers[0] if headers else None)
        runtime = Path(cudart) if cudart else (libraries[0] if libraries else None)
        if not header or not runtime:
            raise RuntimeError('CUDA headers/libcudart not found. Inspect doctor; supply --cuda-include and --cudart. Nothing installed.')
        # Check the chosen runtime against the active torch, not a guessed toolkit version.
        import torch
        lib = ctypes.CDLL(str(runtime)); value = ctypes.c_int()
        if lib.cudaRuntimeGetVersion(ctypes.byref(value)) or value.value != int(torch.version.cuda.split('.')[0])*1000 + int(torch.version.cuda.split('.')[1])*10:
            raise RuntimeError('Selected libcudart version differs from torch CUDA')
        commands.append([compiler, '-std=c++17', '-O2', '-fPIC', '-shared', '-pthread',
                         '-I'+str(header), str(ROOT/'cpp/CUDAFUSE.cpp'), '-L'+str(directory),
                         '-lfuse_dvfs', str(runtime), '-Wl,-rpath,$ORIGIN',
                         '-Wl,-rpath,'+str(runtime.parent), '-o', str(directory/'libfuse_cuda.so')])
    for command in commands:
        subprocess.run(command, check=True)
    core = ctypes.CDLL(str(directory/'libfuse_dvfs.so'))
    if core.fuse_reference_abi_version() != 5 or core.fuse_gpu_backend_abi_version() != 1:
        raise RuntimeError('Native ABI mismatch')
    if cuda and ctypes.CDLL(str(directory/'libfuse_cuda.so')).fuse_cuda_reference_abi_version() != 5:
        raise RuntimeError('CUDA ABI mismatch')
    report = dict(commands=commands, environment=environment_record(), cuda_built=cuda,
                  cuda_executed=False, libraries={p.name: sha(p) for p in directory.glob('libfuse*.so')})
    (directory/'build.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def doctor(model):
    headers, libraries = cuda_paths()
    result = dict(environment=environment_record(), cuda_headers=list(map(str, headers)),
                  cudart=list(map(str, libraries)), compiler=shutil.which('g++'))
    try:
        import torch
        result['torch_cuda'] = dict(version=torch.version.cuda, available=torch.cuda.is_available(),
                                    devices=[dict(index=i,name=torch.cuda.get_device_name(i),
                                                  uuid=str(getattr(torch.cuda.get_device_properties(i),'uuid',None)))
                                             for i in range(torch.cuda.device_count())])
    except Exception as exc: result['torch_cuda_error'] = str(exc)
    try:
        result['vllm_source_hashes'] = check_vllm_sources()
    except Exception as exc:
        result['vllm_error'] = str(exc)
    root = Path(model)
    if (root/'config.json').exists():
        cfg = json.loads((root/'config.json').read_text()); text = cfg.get('text_config', cfg)
        index = root/'model.safetensors.index.json'
        shards = set(json.loads(index.read_text())['weight_map'].values()) if index.exists() else set()
        result['model'] = dict(path=str(root), quantization=cfg.get('quantization_config', text.get('quantization_config')),
                              layers=text.get('num_hidden_layers'), lock=(root/'sparsedvfs_model_lock.json').exists(),
                              missing_shards=sorted(s for s in shards if not (root/s).is_file()))
    return result
