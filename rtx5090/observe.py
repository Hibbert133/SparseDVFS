"""Read-only target-vLLM inventory: doctor -> collect -> summarize.

No bridge/token, frequency writes, FUSE, model downloads, or replay claims.
The 1024-token holdout is declared but deliberately not exposed here.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import time

from .native import check_vllm_sources
from .observe_core import (SCHEMA, DEVELOPMENT, HOLDOUT, GPU_UUIDS, canonical_uuid,
                           digest, inventory, read_json, validate_ranks, write_json)
from .performance import safe_error

WORKER = 'rtx5090.observe_worker.ObservationWorker'
NCCL_ENV = {'NCCL_P2P_DISABLE': '1', 'NCCL_SHM_DISABLE': '1', 'NCCL_IB_DISABLE': '1'}
CONFIG_DIFFERENCES = [
    'User working serve command: TP2/dummy/max_model_len4096/max_num_seqs1/memory0.90/reasoning_parser=qwen3',
    'Observation explicitly fixes BF16/eager/V1/language_model_only; these were not explicit in the supplied serve command',
    'Offline LLM token-ID generation bypasses HTTP/chat/reasoning parsing',
    'Prefix caching is disabled by default to observe full prefill after warmup; actual scheduler configuration is recorded',
    'NCCL P2P/SHM/IB disable values follow the user working script; profiler instrumentation changes execution timing',
]
OVERRIDES = {'gpu_memory_utilization', 'max_model_len', 'max_num_batched_tokens',
             'enable_prefix_caching', 'enable_chunked_prefill', 'async_scheduling',
             'attention_config', 'compilation_config', 'kv_cache_dtype',
             'disable_custom_all_reduce'}


def engine_options(args):
    model = Path(args.model).resolve()
    if not (model/'config.json').is_file():
        raise ValueError('Use the existing local model directory containing config.json')
    value = dict(model=str(model), tokenizer=str(model), dtype='bfloat16',
        load_format='dummy', tensor_parallel_size=2, pipeline_parallel_size=1,
        data_parallel_size=1, enforce_eager=True, language_model_only=True,
        max_model_len=4096, max_num_seqs=1, gpu_memory_utilization=0.9,
        enable_prefix_caching=False, seed=0,
        distributed_executor_backend='mp', worker_cls=WORKER)
    if args.engine_config:
        extra = read_json(args.engine_config)
        if not isinstance(extra, dict) or set(extra)-OVERRIDES:
            raise ValueError('Unsupported engine override; fixed BF16/TP2/dummy/eager fields cannot change')
        value.update(extra)
    if type(value['max_model_len']) is not int or value['max_model_len'] < max(DEVELOPMENT)+64:
        raise ValueError('max_model_len must accommodate the 2048+64 workload')
    if not 0 < value['gpu_memory_utilization'] < 1:
        raise ValueError('gpu_memory_utilization must be between 0 and 1')
    return value


def prepare(args):
    uuids = args.gpu_uuids
    if len(uuids) != 2 or len(set(x.lower() for x in uuids)) != 2 or any(
        not re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', x)
        for x in uuids):
        raise ValueError('Select two distinct full GPU UUIDs in TP rank order')
    if 'torch' in sys.modules and sys.modules['torch'].cuda.is_initialized():
        raise RuntimeError('Use a fresh process before selecting both GPUs')
    root = str(Path(__file__).resolve().parents[1])
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(uuids)
    os.environ['VLLM_USE_V2_MODEL_RUNNER'] = '0'
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ.update(NCCL_ENV)
    old = [p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) if p and p != root]
    os.environ['PYTHONPATH'] = os.pathsep.join([root] + old)
    if not os.environ.get('SPARSEDVFS_TRITON_PATH'):
        raise ValueError('Set the existing persistent SPARSEDVFS_TRITON_PATH first')


def environment(args):
    prepare(args)
    import torch
    import triton
    sources = check_vllm_sources()
    if triton.__version__ != '3.6.0' or Path(triton.__file__).resolve() != (
            Path(os.environ['SPARSEDVFS_TRITON_PATH']).resolve()/'triton/__init__.py'):
        raise RuntimeError('Imported Triton differs from the selected existing 3.6.0 package')
    if torch.cuda.device_count() != 2:
        raise RuntimeError('Observation requires exactly two visible CUDA GPUs')
    devices = []
    for index, expected in enumerate(args.gpu_uuids):
        prop = torch.cuda.get_device_properties(index)
        uuid = canonical_uuid(getattr(prop, 'uuid', 'unavailable'))
        if uuid.lower() != expected.lower():
            raise RuntimeError('CUDA device order differs from requested UUID order')
        devices.append(dict(index=index, uuid=uuid, name=prop.name, total_memory=prop.total_memory))
    files = ('observe.py', 'observe_core.py', 'observe_worker.py')
    return dict(python=sys.version, executable=sys.executable,
        packages={name: importlib.metadata.version(name) for name in ('vllm', 'torch', 'transformers')},
        torch_cuda=torch.version.cuda, triton=dict(version=triton.__version__, file=triton.__file__),
        cuda_devices=devices, vllm_sources_sha256=sources,
        observation_sources_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                    for name in files},
        selected_env={key: os.environ.get(key) for key in
            ('CUDA_VISIBLE_DEVICES', 'VLLM_USE_V2_MODEL_RUNNER', 'VLLM_WORKER_MULTIPROC_METHOD',
             'SPARSEDVFS_TRITON_PATH', 'TRITON_CACHE_DIR',
             'NCCL_P2P_DISABLE', 'NCCL_SHM_DISABLE', 'NCCL_IB_DISABLE')})


def doctor(args):
    out = Path(args.out)
    if (out/'doctor.json').exists():
        raise FileExistsError('doctor.json exists; preserve evidence and choose a new --out')
    report = dict(schema=SCHEMA, status='failed', clock_writes_performed=False,
                  target_worker_verified=False, configuration_differences=CONFIG_DIFFERENCES)
    try:
        report['engine_options'] = engine_options(args)
        report['environment'] = environment(args)
        report['status'] = 'ready'
    except BaseException as exc:
        report['error'] = safe_error(exc)
        raise
    finally:
        write_json(out/'doctor.json', report)
    return report


def scenarios():
    return [dict(prompt_tokens=n, output_tokens=64, repeats=3, warmups=1,
                 split='development', concurrency=1) for n in DEVELOPMENT]


def token_pool(tokenizer, vocab_size):
    special = set(tokenizer.all_special_ids)
    pool = sorted({int(v) for v in tokenizer.get_vocab().values()
                   if 0 <= int(v) < vocab_size and int(v) not in special})
    if len(pool) < 16:
        raise ValueError('Cannot construct valid deterministic non-special token IDs')
    return pool[:256]


def prompt_ids(pool, length):
    return [pool[(17*i + 11) % len(pool)] for i in range(length)]


def check_output(results, prompt):
    if len(results) != 1 or list(results[0].prompt_token_ids) != prompt:
        raise RuntimeError('Inference output request/prompt identity differs')
    if len(results[0].outputs) != 1 or len(results[0].outputs[0].token_ids) != 64:
        raise RuntimeError('Expected exactly 64 generated tokens, without early EOS')
    return dict(prompt_token_ids=prompt, output_token_ids=list(results[0].outputs[0].token_ids),
                request_id=results[0].request_id, finish_reason=results[0].outputs[0].finish_reason)


def collect(args, *, llm_factory=None):
    out = Path(args.out).resolve()
    if (out/'manifest.json').exists() or (out/'requests').exists() or (out/'inventory.json').exists():
        raise FileExistsError('Observation evidence exists; choose a new --out')
    prior = read_json(out/'doctor.json')
    options, env = engine_options(args), environment(args)
    if prior['status'] != 'ready' or prior['engine_options'] != options or prior['environment'] != env:
        raise ValueError('Doctor/config/source mismatch; run doctor in a new output directory')
    report = dict(schema=SCHEMA, status='collecting', split='development',
        engine_options=options, environment=env, scenarios=scenarios(), requests=[],
        configuration_differences=CONFIG_DIFFERENCES,
        holdout=dict(prompt_tokens=HOLDOUT, output_tokens=64, status='sealed_not_collected'),
        clock_writes_performed=False, equation_validated=False)
    write_json(out/'manifest.json', report)
    llm, error = None, None
    try:
        from vllm import LLM, SamplingParams
        llm = (llm_factory or LLM)(**options)
        identities = llm.collective_rpc('observation_identity', timeout=args.rpc_timeout)
        report['workers'] = validate_ranks(identities, args.gpu_uuids)
        for worker in report['workers']:
            if (worker['worker']['sha256'] != env['observation_sources_sha256']['observe_worker.py'] or
                    worker['vllm_sources_sha256'] != env['vllm_sources_sha256']):
                raise RuntimeError('Worker imported different observation/vLLM sources')
        config = llm.llm_engine.vllm_config.model_config
        pool = token_pool(llm.get_tokenizer(), int(config.hf_text_config.vocab_size))
        params = SamplingParams(temperature=0, seed=0, max_tokens=64, min_tokens=64,
                                ignore_eos=True, detokenize=False)
        for scenario in scenarios():
            count = scenario['prompt_tokens']
            tokens = prompt_ids(pool, count)
            print(f'Warmup: input={count}, output=64', flush=True)
            warm = llm.generate([dict(prompt_token_ids=tokens)], params, use_tqdm=False)
            warm_record = check_output(warm, tokens)
            write_json(out/'warmup'/f'{count}.json', warm_record)
            for repeat in range(3):
                key = f'input-{count}-repeat-{repeat+1}'
                directory = out/'requests'/key
                entry = dict(request_key=key, prompt_tokens=count, repeat=repeat+1,
                             split='development', status='collecting')
                report['requests'].append(entry)
                write_json(out/'manifest.json', report)
                started = False
                request_error = None
                try:
                    # Set before RPC: one rank might start even if the other raises.
                    started = True
                    llm.collective_rpc('observation_start', timeout=args.rpc_timeout, args=(dict(
                        directory=str(directory), request_key=key, prompt_tokens=count,
                        split='development', max_operators=args.max_operators),))
                    begin = time.perf_counter()
                    result = llm.generate([dict(prompt_token_ids=tokens)], params, use_tqdm=False)
                    entry['instrumented_request_wall_s'] = time.perf_counter()-begin
                    generation = check_output(result, tokens)
                    entry['generated_tokens_match_warmup'] = generation['output_token_ids'] == warm_record['output_token_ids']
                    write_json(directory/'generation.json', generation)
                except BaseException as exc:
                    request_error = exc
                    entry.update(status='failed', error=safe_error(exc))
                    raise
                finally:
                    if started:
                        try:
                            entry['rank_exports'] = llm.collective_rpc('observation_stop', timeout=args.rpc_timeout)
                        except BaseException as exc:
                            entry['stop_error'] = safe_error(exc)
                            if request_error is None:
                                raise
                if len(entry['rank_exports']) != 2 or any(x['status'] != 'complete' for x in entry['rank_exports']):
                    raise RuntimeError('Missing complete evidence from one or both ranks')
                entry['status'] = 'complete'
                write_json(out/'manifest.json', report)
                print(f'[{len(report["requests"])}/9] {key}: both ranks exported', flush=True)
        report['status'] = 'collected'
    except BaseException as exc:
        error = exc
        report.update(status='failed', error=safe_error(exc))
    finally:
        if llm is not None:
            try:
                llm.llm_engine.engine_core.shutdown()
                report['engine_shutdown_returned'] = True
            except BaseException as exc:
                report.update(status='failed', shutdown_error=safe_error(exc))
                error = error or exc
        write_json(out/'manifest.json', report)
    if error:
        raise error
    return report


def summarize(args):
    out = Path(args.out)
    if (out/'inventory.json').exists():
        raise FileExistsError('inventory.json exists; preserve the existing report')
    manifest = read_json(out/'manifest.json')
    if manifest['schema'] != SCHEMA or manifest['status'] != 'collected':
        raise ValueError('Require a complete development collection')
    validate_ranks(manifest['workers'], [x['uuid'] for x in manifest['environment']['cuda_devices']])
    requests = []
    expected = {f'input-{n}-repeat-{r}' for n in DEVELOPMENT for r in (1, 2, 3)}
    entries = manifest['requests']
    if len(entries) != 9 or {e['request_key'] for e in entries} != expected:
        raise ValueError('Require three requests per development scenario')
    for entry in entries:
        if entry['status'] != 'complete' or entry['split'] != 'development':
            raise ValueError('Incomplete or non-development request')
        for rank in (0, 1):
            path = out/'requests'/entry['request_key']/f'rank-{rank}'/'request.json'
            data = read_json(path)
            if data['rank'] != rank or data['identity'] != manifest['workers'][rank]:
                raise ValueError('Worker identity changed within the collection')
            if data['request_key'] != entry['request_key'] or data['prompt_tokens'] != entry['prompt_tokens']:
                raise ValueError('Request identity changed within the collection')
            requests.append(data)
    report = inventory(requests)
    report['manifest_sha256'] = digest(manifest)
    report['workers'] = manifest['workers']
    write_json(out/'inventory.json', report)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('doctor', 'collect', 'summarize'))
    p.add_argument('--out', required=True)
    p.add_argument('--model', default='/mnt/d/models/Qwen3.5-27B')
    p.add_argument('--gpu-uuids', nargs=2, default=GPU_UUIDS)
    p.add_argument('--engine-config', help='Optional allowlisted JSON overrides matching the working launch')
    p.add_argument('--rpc-timeout', type=float, default=600)
    p.add_argument('--max-operators', type=int, default=250000)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.rpc_timeout <= 0 or args.max_operators <= 0:
            raise ValueError('RPC timeout and operator limit must be positive')
        if args.command == 'doctor':
            result = doctor(args)
            print(json.dumps(dict(status=result['status'], target_worker_verified=False), indent=2))
        elif args.command == 'collect':
            result = collect(args)
            print(json.dumps(dict(status=result['status'], requests=len(result['requests']),
                                  holdout=result['holdout'], clock_writes_performed=False), indent=2))
        else:
            result = summarize(args)
            print(json.dumps(result['summary'], indent=2))
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print(safe_error(exc), file=sys.stderr)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1


if __name__ == '__main__':
    raise SystemExit(main())
