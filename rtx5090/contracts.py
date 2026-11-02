"""Dependency-free version, scheduling and deployment contracts."""
import hashlib
import json
from pathlib import Path

IMAGE_TAG = 'vllm/vllm-openai:v0.19.0'
IMAGE_DIGEST = 'sha256:7a0f0fdd2771464b6976625c2b2d5dd46f566aa00fbc53eceab86ef50883da90'
IMAGE = 'vllm/vllm-openai@' + IMAGE_DIGEST
MODEL_ID = 'cyankiwi/Qwen3.5-27B-AWQ-4bit'
REVISION = '9d8d63632afafe90bdfcb70164d3e2a0955619f7'
WORKER = 'rtx5090.worker.SparseDVFSWorker'
PROTOCOL = 'rtx5090_v1_vllm019_int4_gpu_only'
UNITS = ['embed'] + [f'layer.{i}' for i in range(64)] + ['norm', 'logits', 'sample']
MODES = ('instrumented', 'fixed', 'serial', 'lookahead')


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def engine_args(model='/model', *, custom=True, production=False):
    args = dict(model=model, tokenizer=model, dtype='bfloat16', language_model_only=True,
                revision=REVISION, tokenizer_revision=REVISION, seed=0,
                tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1,
                max_model_len=4096, gpu_memory_utilization=0.70,
                generation_config='vllm')
    if not production:
        args.update(enforce_eager=True, max_num_seqs=1, max_num_batched_tokens=4096,
                    enable_prefix_caching=False, enable_chunked_prefill=False,
                    async_scheduling=False)
    if custom:
        if production:
            raise ValueError('production-native must use the native worker')
        args['worker_cls'] = WORKER
    return args


def server_args(*, custom=True, production=False, port=8000):
    values = engine_args(custom=custom, production=production)
    model = values.pop('model')
    result = ['serve', model, '--host', '0.0.0.0', '--port', str(port),
              '--served-model-name', 'qwen35', '--enable-request-id-headers']
    for key, value in values.items():
        flag = '--' + key.replace('_', '-')
        if isinstance(value, bool):
            result.append(flag if value else '--no-' + key.replace('_', '-'))
        else:
            result.extend([flag, str(value)])
    return result


def validate_engine(config, version, use_v2=False):
    if version.split('+')[0] != '0.19.0':
        raise RuntimeError(f'Expected vLLM 0.19.0; found {version}')
    if use_v2:
        raise RuntimeError('Set VLLM_USE_V2_MODEL_RUNNER=0; only the pinned V1 runner is supported')
    p, m, s, cache = (config.parallel_config, config.model_config,
                       config.scheduler_config, config.cache_config)
    if any(getattr(p, k, 1) != 1 for k in
           ('tensor_parallel_size', 'pipeline_parallel_size', 'data_parallel_size')):
        raise RuntimeError('SparseDVFS v3 requires TP=PP=DP=1')
    if getattr(p, 'enable_dbo', False) or getattr(p, 'use_ubatching', False):
        raise RuntimeError('Microbatching/DBO is unsupported')
    if not m.enforce_eager or s.max_num_seqs != 1:
        raise RuntimeError('SparseDVFS requires enforce_eager=True, max_num_seqs=1')
    if s.enable_chunked_prefill or s.async_scheduling or cache.enable_prefix_caching:
        raise RuntimeError('Disable chunked prefill, async scheduling and cross-request prefix caching')
    if config.speculative_config is not None:
        raise RuntimeError('Speculative decoding is unsupported')
    if getattr(m, 'quantization', None) != 'compressed-tensors' or 'bfloat16' not in str(m.dtype):
        raise RuntimeError('Expected locked compressed-tensors INT4 weights and BF16 activations')
    if m.hf_text_config.num_hidden_layers != 64:
        raise RuntimeError('Expected Qwen3.5-27B with 64 layers')


class PhaseTracker:
    """Uses scheduler counts, never the flattened input tensor length."""
    def __init__(self):
        self.requests = {}

    def step(self, output):
        for rid in output.finished_req_ids:
            self.requests.pop(rid, None)
        scheduled = {rid: n for rid, n in output.num_scheduled_tokens.items() if n > 0}
        if sum(scheduled.values()) != output.total_num_scheduled_tokens:
            raise RuntimeError('Inconsistent scheduler token counts')
        if not scheduled:
            return None
        if len(scheduled) != 1:
            raise RuntimeError('Mixed/batched requests are unsupported')
        if (output.scheduled_spec_decode_tokens or output.scheduled_encoder_inputs or
                getattr(output, 'preempted_req_ids', None)):
            raise RuntimeError('Speculation, multimodal encoder work or preemption is unsupported')
        cached = output.scheduled_cached_reqs
        if cached.resumed_req_ids:
            raise RuntimeError('Resumed/preempted requests require reprofiling; unsupported in v3')
        for req in output.scheduled_new_reqs:
            if req.req_id in self.requests or req.prompt_token_ids is None or req.mm_features:
                raise RuntimeError('Expected a new text-only token-ID request')
            sp = req.sampling_params
            if sp is not None and (sp.prompt_logprobs is not None or sp.logprobs is not None):
                raise RuntimeError('Prompt/output logprobs add unprofiled work; unsupported')
            self.requests[req.req_id] = dict(prompt=len(req.prompt_token_ids),
                                            computed=req.num_computed_tokens, step=0,
                                            prompt_sha256=digest_json(req.prompt_token_ids),
                                            sampling={k: getattr(sp, k, None) for k in
                                                ('temperature', 'seed', 'max_tokens', 'min_tokens', 'ignore_eos')})
        if len(cached.req_ids) != len(cached.num_computed_tokens):
            raise RuntimeError('Malformed cached request counters')
        for rid, count in zip(cached.req_ids, cached.num_computed_tokens):
            if rid not in self.requests:
                raise RuntimeError('Unknown cached request ID')
            self.requests[rid]['computed'] = count
        rid, count = next(iter(scheduled.items()))
        if rid not in self.requests or len(self.requests) > 1:
            raise RuntimeError('Only one live request is allowed')
        state = self.requests[rid]
        computed = state['computed']
        if computed < state['prompt']:
            if computed != 0 or count != state['prompt'] or state['step'] != 0:
                raise RuntimeError('Chunked/cached/partial prefill is unsupported')
            phase = 'prefill'
        else:
            if count != 1 or computed != state['prompt'] + state['step'] - 1:
                raise RuntimeError('Expected one sequential decode token without speculation')
            phase = 'decode'
        result = dict(request_id=rid, phase=phase, step=state['step'],
                      num_computed_tokens=computed, num_scheduled_tokens=count,
                      prompt_tokens=state['prompt'], prompt_sha256=state['prompt_sha256'],
                      sampling=state['sampling'])
        state['computed'] += count
        state['step'] += 1
        return result


def validate_plan(plan, identity=None):
    if plan.get('identity', {}).get('backend') != PROTOCOL:
        raise ValueError('Use a real v5 vLLM profile/plan; v2 Transformers, v3/v4 and synthetic plans are forbidden')
    if any(b.get('triplet', [1,1,1])[0] != 0 or b.get('triplet', [1,1,1])[2] != 0 for b in plan['blocks']):
        raise ValueError('5090 CPU and memory domains must be disabled')
    if plan.get('schema') != 5:
        raise ValueError('Use a v5 plan schema; old formats cannot authorize frequency control')
    granularity = plan.get('granularity', 'module')
    if granularity != plan['identity'].get('granularity', 'module'):
        raise ValueError('Plan granularity differs from the active profile')
    if plan.get('selector') == 'paper' and (plan.get('calibration_schema') != 4 or plan.get('calibration', {}).get('schema') != 4):
        raise ValueError('v5 paper-equation plans require schema-4 calibration; legacy free per-OPP alpha/thermal coefficients are not deployable')
    if plan.get('schema') == 5 and plan.get('selector') == 'paper':
        cal = plan['calibration']
        if (not cal.get('v5_validation', {}).get('deployable') or cal.get('synthetic') or
                cal.get('profile_sha256') != plan.get('profile_sha256')):
            raise ValueError('v5 strict paper plans require held-out validated real calibration for the same profile')
    if granularity == 'operator':
        from sparsedvfs_vllm.operator_contracts import validate_certificate
        certificate = validate_certificate(plan['coverage_certificate'], plan['identity'])
        expected = {k: c['units'] for k, c in certificate['cases'].items()}
        phases = certificate['context_order']
        used = {tuple(block['triplet']) for block in plan['blocks']}
        for vector in used:
            key = ','.join(map(str, vector))
            evidence = plan.get('opp_coverage_certificates', {}).get(key)
            if not evidence:
                raise ValueError('Operator plan lacks diagnostic GPU coverage at a used OPP')
            validate_certificate(evidence, plan['identity'])
            if evidence.get('applied_triplet') != list(vector) or any(
                    evidence['cases'][k]['layout_sha256'] != case['layout_sha256'] or
                    evidence['cases'][k]['units'] != case['units'] for k,case in certificate['cases'].items()):
                raise ValueError('Operator OPP evidence does not match the scheduled vector/layout')
    elif granularity == 'module':
        expected = {p: UNITS for p in ('prefill', 'decode')}
        phases = list(expected)
    else:
        raise ValueError('Unknown execution granularity')
    if plan.get('units') != (expected if granularity == 'operator' else UNITS):
        raise ValueError('Plan must cover embedding, 64 layers, norm, logits and sampling')
    if plan.get('calibration') and any(o['s_comp'] != 0 or o['s_mem'] != 0
                                      for o in plan['calibration']['operators'].values()):
        raise ValueError('Dense BF16 v5 requires zero effective sparsity coefficients')
    if identity is not None and plan['identity'] != identity:
        raise ValueError('Plan/profile identity differs from the active worker/workload')
    ids = []
    for phase in phases:
        blocks = [b for b in plan['blocks'] if b['phase'] == phase]
        if [u for b in blocks for u in b['operators']] != expected[phase]:
            raise ValueError(f'Incomplete, duplicate or unordered {phase} coverage')
        for block in blocks:
            ids.append(block['block_id'])
            if block['estimated_ms'] <= 0 or len(block['triplet']) != 3:
                raise ValueError('Invalid block duration/vector')
    if len(ids) != len(set(ids)) or plan['switch_us'] <= 0:
        raise ValueError('Duplicate block IDs or missing measured switch latency')
    if any(b['phase'] not in phases for b in plan['blocks']):
        raise ValueError('Unexpected plan context')


def verify_model(path, hashes=False):
    """The revision marker is checked together with the safetensors index/shards."""
    path = Path(path)
    marker = json.loads((path / 'sparsedvfs_model_lock.json').read_text())
    if marker.get('revision') != REVISION or marker.get('model_id') != MODEL_ID:
        raise ValueError('Model lock differs from the fixed repository/revision')
    config = json.loads((path / 'config.json').read_text())
    lock = json.loads((Path(__file__).resolve().parents[1]/'configs/5090.lock.json').read_text())
    if hashlib.sha256((path/'config.json').read_bytes()).hexdigest() != lock['config_sha256']:
        raise ValueError('Quantized config differs from locked revision')
    text = config['text_config']
    if config['model_type'] != 'qwen3_5' or text['num_hidden_layers'] != 64:
        raise ValueError('Unexpected model architecture')
    if sum(t == 'full_attention' for t in text['layer_types']) != 16:
        raise ValueError('Unexpected hybrid attention layout')
    index = json.loads((path / 'model.safetensors.index.json').read_text())
    shards = sorted(set(index['weight_map'].values()))
    sizes = {}
    for name in shards:
        file = path / name
        if not file.is_file() or file.stat().st_size != marker['files'][name]['size']:
            raise ValueError(f'Missing/incomplete weight shard: {name}')
        if hashes:
            with file.open('rb') as source:
                if hashlib.file_digest(source, 'sha256').hexdigest() != marker['files'][name]['sha256']:
                    raise ValueError(f'Weight hash mismatch: {name}')
        sizes[name] = file.stat().st_size
    for name, row in marker['files'].items():
        if name.endswith('.safetensors'):
            continue
        file = path / name
        if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError(f'Model metadata/tokenizer changed: {name}')
    return {'model_id': MODEL_ID, 'revision': REVISION, 'weight_bytes': sum(sizes.values()),
            'shards': len(shards), 'manifest_sha256': digest_json(marker)}


def identity(config, vllm, torch, cuda, device):
    import platform
    from .backends import backend
    lock = json.loads((Path(__file__).resolve().parents[1]/'configs/5090.lock.json').read_text())
    b=backend(config['gpu'])
    try:
        capabilities={k:v for k,v in b.probe().items() if k in ('uuid','driver','name','rates_mhz','synthetic')}
    finally: b.close()
    root=Path(__file__).resolve().parents[1]
    sources={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
             for folder in ('rtx5090','cpp') for p in (root/folder).glob('*') if p.suffix in ('.py','.cpp','.h')}
    return dict(backend=PROTOCOL, model=MODEL_ID, revision=REVISION,
                model_manifest_sha256=verify_model('/model')['manifest_sha256'],
                quantization_sha256=digest_json(lock['quantization']), image=lock['image'],
                vllm=vllm, torch=torch, cuda=cuda, device=device,
                gpu=config['gpu'], capabilities=capabilities, kernel=platform.release(), sources_sha256=digest_json(sources),
                granularity=config.get('granularity','module'),
                workload=config['workload'], eager=True, max_model_len=4096,
                gpu_memory_utilization=0.70, cpu='uncontrolled', memory='uncontrolled')
