"""Proxy-free existing-service client. No engine construction, no worker hooks."""
from contextlib import asynccontextmanager
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import socket
import subprocess
import urllib.error
import urllib.request

from .observe_core import digest, GPU_UUIDS
from .service_core import SCHEMA, read, write_new, positive

BASE_URL = 'http://10.255.255.254:8000'
MODEL = '/mnt/d/models/Qwen3.5-27B'
TRITON = '/mnt/d/s00958243/triton-360'
PROXY_KEYS = ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy')
ENV_KEYS = ('CUDA_VISIBLE_DEVICES', 'PYTHONPATH', 'TRITON_CACHE_DIR', 'VLLM_USE_V2_MODEL_RUNNER',
            'NCCL_P2P_DISABLE', 'NCCL_SHM_DISABLE', 'NCCL_IB_DISABLE', 'NCCL_DEBUG',
            'NCCL_DEBUG_SUBSYS', 'NCCL_CUMEM_HOST_ENABLE')


def isolate_proxy():
    for key in PROXY_KEYS: os.environ.pop(key, None)
    for key in ('NO_PROXY', 'no_proxy'):
        os.environ[key] = '127.0.0.1,localhost,10.255.255.254'


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def http(base, route, body=None, timeout=10, text=False):
    request = urllib.request.Request(base.rstrip('/')+route,
        data=None if body is None else json.dumps(body).encode(),
        headers={'Content-Type': 'application/json'} if body is not None else {})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        value = response.read().decode('utf-8')
    return value if text else (json.loads(value) if value.strip() else {})


def profiler_config(directory):
    return dict(profiler='torch', torch_profiler_dir=str(Path(directory).resolve()),
        ignore_frontend=True, torch_profiler_with_stack=False, torch_profiler_record_shapes=False,
        torch_profiler_with_memory=False, torch_profiler_with_flops=False,
        capture_torch_profiler=False, max_iterations=3)


def process_record(pid):
    root = Path('/proc')/str(pid)
    stat = (root/'stat').read_text().rsplit(')', 1)[1].split()
    argv = (root/'cmdline').read_bytes().decode().split('\0')
    return dict(pid=int(pid), start_ticks=stat[19], argv=[x for x in argv if x])


def preflight(out, launch=False):
    """Validate the real installed types BEFORE the one permitted restart."""
    import torch
    import triton
    from vllm.config import ProfilerConfig
    from vllm.benchmarks.lib.endpoint_request_func import RequestFuncInput, RequestFuncOutput
    import vllm
    from .native import triton_record
    version = importlib.metadata.version('vllm')
    if version.split('+')[0] != '0.32.0':
        raise ValueError('Audit this installed vLLM version before launching; do not change dependencies')
    config = profiler_config(out)
    ProfilerConfig(**config)
    for cls, keys in ((RequestFuncInput, ('prompt', 'extra_body', 'ignore_eos', 'request_id')),
                      (RequestFuncOutput, ('prompt_len', 'output_tokens', 'ttft', 'latency'))):
        if not set(keys) <= set(cls.__dataclass_fields__):
            raise ValueError('Installed benchmark contract differs; stop before restart')
    source_root = Path(vllm.__file__).resolve().parent
    api_file = source_root/'entrypoints/serve/profile/api_router.py'
    tree = ast.parse(api_file.read_text(encoding='utf-8'))
    request_type = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'StartProfileRequest'), None)
    fields = {n.target.id for n in request_type.body if isinstance(n, ast.AnnAssign)
              and isinstance(n.target, ast.Name)} if request_type else set()
    if not {'profile_prefix', 'delay_iterations', 'max_iterations'} <= fields:
        raise ValueError('Installed profile API lacks short-window overrides; audit before restart')
    source_files = ('config/profiler.py', 'profiler/wrapper.py', 'v1/worker/gpu_worker.py',
                    'entrypoints/serve/profile/api_router.py', 'benchmarks/lib/endpoint_request_func.py')
    source_hashes = {p: sha(source_root/p) for p in source_files}
    tool_hashes = {p.name: sha(p) for p in Path(__file__).parent.glob('service_*.py')}
    tool_hashes.update({p: sha(Path(__file__).parent/p) for p in ('backends.py', 'bridge.py')})
    wrapper = (source_root/'profiler/wrapper.py').read_text(encoding='utf-8')
    if 'self._profiling_for_iters > self._max_iters' not in wrapper:
        raise ValueError('Profiler iteration-stop semantics changed; audit before restart')
    actual = triton_record(TRITON)
    env = {k: os.environ.get(k) for k in ENV_KEYS}
    if env['CUDA_VISIBLE_DEVICES'] != '0,1' or env['NCCL_CUMEM_HOST_ENABLE'] != '0':
        raise ValueError('Run the provided script to preserve the successful environment')
    if any(env[k] is not None for k in ENV_KEYS if k.startswith('NCCL_') and k != 'NCCL_CUMEM_HOST_ENABLE'):
        raise ValueError('Old NCCL restrictions/debug variables remain')
    result = dict(schema=SCHEMA, status='preflight_ready_not_worker_verification',
        profiler_config=config, triton=actual, torch_version=str(torch.__version__),
        torch_git_version=torch.version.git_version, cuda=torch.version.cuda,
        vllm_version=version, environment=env, model=MODEL,
        audited_interface_source_hashes=source_hashes,
        tool_source_hashes=tool_hashes,
        weight_mode='not_forced_must_verify_startup', dtype='not_forced_must_verify_startup')
    if launch:
        old = read(Path(out)/'preflight.json')
        if result != old: raise ValueError('Environment changed since preflight; inspect before stopping service')
        result['service_process'] = process_record(os.getppid())
        write_new(Path(out)/'launch.json', result)
    else: write_new(Path(out)/'preflight.json', result)
    return result


def startup_identity(log):
    # Keep provenance. This parser reads actual engine initialization logs, not defaults.
    lines = [line for line in log.splitlines() if 'Initializing a V' in line and 'config:' in line]
    config_line = re.sub(r'\x1b\[[0-9;]*m', '', lines[-1]) if lines else ''
    def value(key):
        m = re.search(r'\b'+key+r"['\"]?\s*[:=]\s*['\"]?([\w.]+)", config_line)
        return m.group(1) if m else None
    dtype, load_format = value('dtype'), value('load_format')
    loaded = bool(re.search(r'Loading model weights took|Model loading took', log))
    worker_pids = {}
    for rank, pid in re.findall(r'Worker_TP([01])\s+pid=(\d+)', log): worker_pids[rank] = int(pid)
    return dict(dtype=dtype, load_format=load_format, weights_loaded_log=loaded,
        resolved_weight_loader='not_directly_observed' if load_format == 'auto' else load_format,
        worker_pids_from_log=worker_pids,
        tensor_parallel_size=value('tensor_parallel_size'),
        prefix_caching=value('enable_prefix_caching'),
        engine_config_line=config_line, source='actual_startup_log',
        runner_selection='unknown_unless_explicit_in_log',
        enforce_eager=value('enforce_eager'), quantization=value('quantization'),
        enable_chunked_prefill=value('enable_chunked_prefill'),
        speculative_config=value('speculative_config'),
        communication_channels='unknown_no_detailed_NCCL_log',
        worker_triton_import='not_directly_observed_launch_preflight_only')


def inspect_service(base, launch_dir=None):
    isolate_proxy()
    models, spec = http(base, '/v1/models'), http(base, '/openapi.json')
    paths = spec.get('paths', {})
    routes = {p: 'post' in paths.get(p, {}) for p in ('/start_profile', '/stop_profile')}
    result = dict(schema=SCHEMA, base_url=base, models=models, profiler_routes=routes,
                  client=dict(hostname=socket.gethostname(), platform=os.name),
                  source='live_HTTP_and_local_launch_evidence', identity_ready=False,
                  unknown=['graph_capture_node_mapping', 'actual_collective_transport', 'worker_import_hashes'])
    schemas = spec.get('components', {}).get('schemas', {})
    properties = schemas.get('StartProfileRequest', {}).get('properties', {})
    result['profile_overrides_supported'] = {'profile_prefix', 'delay_iterations', 'max_iterations'} <= set(properties)
    if launch_dir:
        directory = Path(launch_dir).resolve()
        launch = read(directory/'launch.json')
        current = process_record(launch['service_process']['pid'])
        if current['start_ticks'] != launch['service_process']['start_ticks']:
            raise ValueError('PID reused: launch evidence does not identify this service')
        if 'serve' not in current['argv'] or MODEL not in current['argv']:
            raise ValueError('Recorded process is not the declared live vLLM service')
        log = (directory/'startup.log').read_text(encoding='utf-8', errors='replace')
        state = startup_identity(log)
        smi = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,driver_version', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, check=True, timeout=10)
        gpu_rows = [list(map(str.strip, line.split(','))) for line in smi.stdout.splitlines() if ',' in line]
        indexed = {row[0]: row[1] for row in gpu_rows}
        hardware = {row[1]: dict(name=row[2], driver_version=row[3]) for row in gpu_rows if len(row) == 4}
        if [indexed.get('0'), indexed.get('1')] != GPU_UUIDS:
            raise ValueError('CUDA_VISIBLE_DEVICES index-to-UUID mapping changed')
        model_entries = models.get('data', [])
        model_matches = any(x.get('root', x.get('id')) == MODEL for x in model_entries)
        ready = bool(state['dtype'] in ('torch.bfloat16', 'torch.float16', 'torch.float32',
                                        'bfloat16', 'float16', 'float32')
                     and state['load_format'] and state['weights_loaded_log']
                     and state['tensor_parallel_size'] == '2' and model_matches)
        if state['enforce_eager'] not in ('False', 'false'):
            ready = False  # Never certify the old eager path as the optimized baseline.
        ready = ready and all(routes.values()) and result['profile_overrides_supported']
        result.update(launch_dir=str(directory), startup_identity=state, launch=launch,
                      service_process=current, gpu_index_uuids=indexed,
                      gpu_hardware=hardware,
                      identity_ready=ready,
                      rank_uuid_basis='standard_TP_launch_visibility_not_direct_worker_RPC')
    else:
        result['unknown'].append('actual_dtype_weights_service_boot_and_GPU_binding')
    # Only same process lifetime models: no claim of portability across restarts.
    basis = {k: result.get(k) for k in ('base_url', 'client', 'launch',
                                       'service_process', 'startup_identity', 'gpu_index_uuids', 'gpu_hardware')}
    # OpenAI ModelCard.created may be generated anew by /v1/models. It is not
    # a service boot identifier and must not invalidate every subsequent check.
    basis['served_models'] = [{k: row.get(k) for k in ('id', 'root', 'parent', 'max_model_len')}
                              for row in models.get('data', [])]
    result['fingerprint'] = digest(basis)
    result['status'] = 'ready_for_service_window' if result['identity_ready'] else 'needs_identity_evidence'
    result['ready_for_operator_models'] = False
    return result


class UsageContent:
    """Tee usage metadata while the installed vLLM parser handles all timing."""
    def __init__(self, content, owner, parser):
        self.content, self.owner, self.parser = content, owner, parser

    async def iter_any(self):
        async for chunk in self.content.iter_any():
            for message in self.parser.add_chunk(chunk):
                if not message.startswith('data:'): continue
                payload = message[5:].strip()
                if payload == '[DONE]': continue
                value = json.loads(payload)
                if value.get('usage'): self.owner.usage = value['usage']
            yield chunk


class UsageSession:
    def __init__(self, session, parser_type):
        self.session, self.parser_type, self.usage = session, parser_type, None

    @asynccontextmanager
    async def post(self, *args, **kwargs):
        async with self.session.post(*args, **kwargs) as response:
            class Response:
                def __getattr__(self, name): return getattr(response, name)
            wrapped = Response()
            wrapped.content = UsageContent(response.content, self, self.parser_type())
            yield wrapped


class BenchClient:
    def __init__(self, base, model_id, tokenizer_path=MODEL):
        isolate_proxy()
        import aiohttp
        from vllm.benchmarks.lib import endpoint_request_func as endpoint
        from vllm.tokenizers import get_tokenizer
        self.aiohttp, self.endpoint = aiohttp, endpoint
        self.base, self.model_id = base, model_id
        self.source_sha256 = sha(endpoint.__file__)
        self.tokenizer = get_tokenizer(tokenizer_path, trust_remote_code=True)
        self.vocab = sorted(set(self.tokenizer.get_vocab().values())-set(self.tokenizer.all_special_ids))
        if not self.vocab: raise ValueError('No legal non-special token IDs')
        self.tokenizer_fingerprint = digest(dict(vocab=self.vocab,
            tokenizer_files={p.name: sha(p) for p in Path(tokenizer_path).glob('*token*') if p.is_file()}))
        self.session = None

    async def __aenter__(self):
        self.session = self.aiohttp.ClientSession(trust_env=False,
            timeout=self.aiohttp.ClientTimeout(total=600, connect=5))
        return self

    async def __aexit__(self, *exc):
        await self.session.close()

    def prepare(self, length, seed, request_id):
        rng = random.Random(seed)
        tokens = [rng.choice(self.vocab) for _ in range(length)]
        return dict(request_id=request_id, tokens=tokens, prompt_hash=digest(tokens), length=length, seed=seed)

    async def run(self, request):
        inp = self.endpoint.RequestFuncInput(prompt=request['tokens'], prompt_len=request['length'],
            output_len=64, model=self.model_id, api_url=self.base+'/v1/completions',
            ignore_eos=True, request_id=request['request_id'],
            extra_body=dict(temperature=0, add_special_tokens=False))
        session = UsageSession(self.session, self.endpoint.StreamedResponseHandler)
        output = await self.endpoint.ASYNC_REQUEST_FUNCS['openai'](inp, session=session)
        if not output.success: raise RuntimeError('Benchmark request failed: '+output.error)
        usage = session.usage or {}
        if usage.get('prompt_tokens') != request['length'] or usage.get('completion_tokens') != 64:
            raise ValueError(f'Actual token usage differs/missing: {usage}')
        if output.output_tokens != 64 or output.prompt_len != request['length']:
            raise ValueError('Installed benchmark result disagrees with raw usage')
        latency = positive(output.latency, 'HTTP streaming latency')
        ttft = positive(output.ttft, 'HTTP TTFT')
        tpot = positive((latency-ttft)/63, 'HTTP TPOT')
        return dict(request_id=request['request_id'], prompt_hash=request['prompt_hash'],
                    usage=usage, ttft_s=ttft, tpot_s=tpot, latency_s=latency,
                    output_text_sha256=hashlib.sha256(output.generated_text.encode()).hexdigest())
