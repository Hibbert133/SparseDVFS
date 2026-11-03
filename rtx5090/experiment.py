"""One-shot offline experiments. All physical writes require explicit --apply."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
from .contracts import UNITS, engine_args, digest_json, verify_model, identity, validate_plan
from .run import read, write, generate, rpc, validate_trace, calibrate
from .native import ROOT, assert_cuda_uuid, environment_record
from .backends import backend
from .energy import Meter, acceptance, percentile


def dataset_contract(dataset):
    if dataset.get('input_tokens') != 512 or dataset.get('output_tokens') != 128:
        raise ValueError('First experiment is fixed at 512 input / 128 output tokens')
    groups = dataset['splits']
    hashes = []
    for name in ('train','test'):
        if not groups.get(name): raise ValueError('Both train and test must be nonempty')
        for ids in groups[name]:
            if len(ids) != 512 or any(type(i) is not int or i < 0 for i in ids):
                raise ValueError('Invalid token-ID workload')
            hashes.append(digest_json(ids))
    if len(set(hashes)) != len(hashes): raise ValueError('Train/test prompts must be distinct and disjoint')
    return dict(input_tokens=512, output_tokens=128, temperature=0.0, seed=0, ignore_eos=True,
                dataset_sha256=digest_json(dataset), allowed_prompt_sha256=hashes,
                sampling_contract='greedy_fixed_length', warmups=3)


def make_dataset(model, out):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    subjects = ['compiler correctness', 'sorting algorithms', 'queue scheduling', 'memory locality',
                'database indexing', 'network congestion', 'matrix multiplication', 'cache invalidation',
                'file system recovery', 'graph traversal', 'numerical stability', 'distributed consensus',
                'thread synchronization', 'data compression', 'hash table collisions', 'query optimization']
    rows = []
    for i, subject in enumerate(subjects):
        seed = tokenizer.encode(f'Case {i}: Explain {subject} with an example and discuss its limitations. ', add_special_tokens=False)
        rows.append((seed*(512//len(seed)+1))[:512])
    value = dict(schema=1, input_tokens=512, output_tokens=128, generated_prompts=True,
                 model_manifest_sha256=verify_model(model)['manifest_sha256'],
                 splits=dict(train=rows[:8], test=rows[8:]),
                 scope='Fixed-length engineering prompts; no general-workload claim')
    dataset_contract(value); write(out, value)


def gpu_config(a):
    if a.gpu:
        config = read(a.gpu)
        if config['uuid'] != a.gpu_uuid: raise ValueError('GPU UUID mismatch')
        return config
    return dict(backend=a.backend, uuid=a.gpu_uuid, default_auto_ack=a.ack_default_auto,
                rates_mhz=[900,1500] if a.backend == 'mock' else [])


def require_apply(a, cfg):
    if not a.apply: raise ValueError('Physical clock writes require explicit --apply')
    if cfg['backend'] == 'mock': raise ValueError('Mock is not a real clock experiment')
    if not a.ack_default_auto or not cfg.get('default_auto_ack'):
        raise ValueError('Start with default automatic clocks and pass --ack-default-auto')


def audit_worker(worker):
    import torch
    import vllm
    from .contracts import validate_engine
    validate_engine(worker.vllm_config, vllm.__version__, worker.use_v2_model_runner)
    uuid = assert_cuda_uuid(torch, os.environ['SPARSEDVFS_GPU_UUID'])
    return dict(uuid=uuid, pid=os.getpid(), runner=type(worker.model_runner).__module__)


def background(uuid):
    try:
        p = subprocess.run(['nvidia-smi', '-i', uuid, '--query-compute-apps=pid,process_name,used_gpu_memory',
                            '--format=csv,noheader'], capture_output=True, text=True, timeout=10)
        return dict(returncode=p.returncode, output=p.stdout, error=p.stderr,
                    caveat='WSL/Windows PID namespaces may differ; inspect for unrelated workloads')
    except (OSError, subprocess.TimeoutExpired) as exc:
        return dict(unavailable=str(exc))


def baseline(a):
    """Native-worker correctness baseline; no frequency writes, telemetry optional."""
    import torch
    from vllm import LLM
    if not a.ack_default_auto:
        raise ValueError('Baseline cannot read an existing lock policy: start at automatic clocks and pass --ack-default-auto')
    verify_model(a.model, hashes=True)
    dataset = read(a.dataset); work = dataset_contract(dataset)
    if dataset['model_manifest_sha256'] != verify_model(a.model)['manifest_sha256']:
        raise ValueError('Dataset belongs to another model')
    llm = LLM(**engine_args(a.model, custom=False)); monitor = None; collector = None
    try:
        audit = llm.collective_rpc(audit_worker)
        rows, clocks, telemetry_errors = [], [], []
        try: monitor = backend(gpu_config(a))
        except Exception as exc: telemetry_errors.append(str(exc))
        for ids in dataset['splits']['train'][:3]: generate(llm, dict(input_token_ids=ids, output_tokens=128))
        if monitor:
            collector = Meter(monitor,a.gpu_uuid)
            try: collector.start()
            except Exception as exc:
                telemetry_errors.append(str(exc)); collector = None
        for split in ('train','test'):
            for ids in dataset['splits'][split]:
                row = generate(llm, dict(input_token_ids=ids, output_tokens=128))
                row.update(split=split, prompt_sha256=digest_json(ids)); rows.append(row)
        if collector:
            try: collector.stop()
            except Exception as exc: telemetry_errors.append(str(exc))
            clocks = collector.rows; collector = None
        write(Path(a.out)/'baseline.json', dict(workload=work, model=verify_model(a.model),
              runs=rows, outputs={r['prompt_sha256']:r['output_sha256'] for r in rows},
              environment=environment_record(), audit=audit, telemetry=clocks,
              telemetry_errors=telemetry_errors, real_weights=True, real_cuda=True,
              frequency_applied=False, energy_measured=False, success_claim=False))
    finally:
        try:
            if collector:
                try: collector.stop()
                except Exception: pass
            if monitor: monitor.close()
        finally: llm.llm_engine.engine_core.shutdown()


def smoke(a):
    """Small native dummy interface test, explicitly outside the real-weight contract."""
    if a.apply: raise ValueError('Dummy cannot authorize --apply')
    from vllm import LLM, SamplingParams
    root = Path(a.out)/'dummy_model'; root.mkdir(parents=True, exist_ok=True)
    cfg = dict(architectures=['LlamaForCausalLM'], model_type='llama', hidden_size=128,
               intermediate_size=256, num_hidden_layers=2, num_attention_heads=4,
               num_key_value_heads=4, vocab_size=256, max_position_embeddings=512, torch_dtype='bfloat16')
    write(root/'config.json', cfg)
    llm = LLM(model=str(root), tokenizer=str(root), skip_tokenizer_init=True, load_format='dummy',
              dtype='bfloat16', enforce_eager=True, max_model_len=256, max_num_seqs=1,
              gpu_memory_utilization=.2, enable_prefix_caching=False, enable_chunked_prefill=False,
              async_scheduling=False)
    try:
        out = llm.generate([dict(prompt_token_ids=[1,2,3])], SamplingParams(temperature=0,max_tokens=4,ignore_eos=True), use_tqdm=False)
        write(Path(a.out)/'dummy.json', dict(synthetic=True, real_weights=False, frequency_applied=False,
              real_cuda=True, vllm_native_interface_only=True, sparse_hook_coverage=False,
              output_token_ids=list(out[0].outputs[0].token_ids), deployable=False))
    finally: llm.llm_engine.engine_core.shutdown()


def probe(a):
    cfg = gpu_config(a); device = backend(cfg)
    try:
        info = device.probe(); telemetry = device.telemetry()
    finally: device.close()
    write(Path(a.out)/'probe.json', dict(capabilities=info, telemetry=telemetry,
          clock_writes_performed=False, lock_reset_capability='UNVERIFIED until explicit calibration --apply'))
    if a.baseline:
        from .planning import select_rates
        base = read(a.baseline)
        if base['audit'][0]['uuid'].lower() != a.gpu_uuid.lower(): raise ValueError('Baseline GPU differs')
        clocks = [r['clock_mhz'] for r in base['telemetry'] if r.get('clock_mhz') and r.get('utilization_pct') != 0]
        cfg['rates_mhz'] = select_rates(info['rates_mhz'], percentile(clocks,.95))
    elif not cfg['rates_mhz']:
        print('Read-only probe saved. Pass --baseline after real inference to select the frequency grid.')
        return
    write(Path(a.out)/'gpu.json', cfg)


def window(a, job):
    import torch
    import vllm
    from vllm import LLM
    cfg = job['gpu']; mode = job['mode']; custom = mode != 'auto'
    real = custom and cfg['backend'] != 'mock' and mode != 'instrumented'
    if not job.get('chain'): require_apply(a, cfg)
    if real: os.environ['SPARSEDVFS_APPLY'] = '1'
    data = read(a.dataset); work = dataset_contract(data)
    if data['model_manifest_sha256'] != verify_model(a.model)['manifest_sha256']:
        raise ValueError('Dataset/model mismatch')
    base = read(a.baseline)
    if base['workload'] != work or not base.get('real_weights'): raise ValueError('Invalid baseline')
    if base.get('environment') != environment_record():
        raise ValueError('Baseline environment/libraries differ; regenerate native baseline')
    split = job.get('split','train'); prompts = data['splits'][split]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    wcfg = dict(gpu=cfg, model=a.model, mode=mode, workload=work, enabled=True,
                apply=real, granularity='module', out_dir=str(out/'worker'),
                fixed_triplet=[0,job.get('mhz',cfg['rates_mhz'][0])*1000000,0],
                switch_calibrated=bool(a.calibration))
    if a.calibration: wcfg['calibration'] = str(Path(a.calibration).resolve())
    if job.get('plan'): wcfg['plan'] = job['plan']
    result = dict(restoration_ok=False); llm = None; monitor = None; meter = None
    configured = False; auto_controller = None
    try:
        # Fresh process + fresh model for each window: same allocator/startup policy for both arms.
        llm = LLM(**engine_args(a.model, custom=custom))
        audit = llm.collective_rpc(audit_worker)
        run_identity = identity(wcfg, vllm.__version__, torch.__version__, torch.version.cuda, torch.cuda.get_device_name())
        if job.get('chain'):
            from .run import build_plan
            profile = dict(identity=run_identity, synthetic=True, switch_us=1000,
                           samples=[dict(mhz=f, latencies={p:{u:1.0 for u in UNITS} for p in ('prefill','decode')}) for f in cfg['rates_mhz']])
            write(out/'mock_profile.json', profile)
            write(out/'mock_plan.json', build_plan(profile,out/'mock_profile.json'))
            wcfg['plan'] = str(out/'mock_plan.json')
        if custom:
            worker_identity = rpc(llm, 'sdvfs_configure', config=wcfg); configured = True
            if worker_identity != run_identity: raise ValueError('Parent/worker runtime identity differs')
            rpc(llm, 'sdvfs_prepare')
        else:
            # Reset is a physical write and is gated by require_apply above. Own
            # the selected GPU until this window ends, including automatic mode.
            from .controller import Controller
            auto_controller = Controller(cfg,out/'auto_controller')
            auto_controller.backend.reset()
        def request(ids):
            row = generate(llm, dict(input_token_ids=ids, output_tokens=128))
            row['prompt_sha256'] = digest_json(ids)
            if row['output_sha256'] != base['outputs'][row['prompt_sha256']]:
                raise RuntimeError('Output differs from native real-weight baseline')
            return row
        for i in range(3): request(prompts[i % len(prompts)])
        if custom: rpc(llm, 'sdvfs_export', reset=True)
        before = rpc(llm,'sdvfs_gpu_readback') if custom else None
        processes_before = background(a.gpu_uuid)
        if not job.get('chain'):
            monitor = backend(cfg)
            meter = Meter(monitor, a.gpu_uuid, read(a.power_evidence) if a.power_evidence else None)
            meter.start()
        rows = []; start = time.monotonic(); count = job.get('requests',0)
        while True:
            rows.append(request(prompts[len(rows) % len(prompts)]))
            if count:
                if len(rows) >= count: break
            elif len(rows) % len(prompts) == 0 and time.monotonic()-start >= max(30, a.min_seconds): break
        energy = meter.stop() if meter else None
        if meter: write(out/'telemetry.json', meter.rows)
        meter = None
        after = rpc(llm,'sdvfs_gpu_readback') if custom else None
        if real and mode == 'fixed' and any(abs(x['observed_mhz']-job['mhz'])>15 for x in (before,after)):
            raise RuntimeError('Fixed-frequency readback differs from requested MHz')
        groups = {p:{u:[] for u in UNITS} for p in ('prefill','decode')}
        if custom:
            trace = rpc(llm,'sdvfs_export',reset=True,close=True)
            # The final idempotent close may export an empty trace after reset;
            # keep the exact measured interval separately for audit.
            write(out/'measured_trace.json',trace)
            for i in range(len(rows)):
                lo,hi = i*128,(i+1)*128
                validate_trace(dict(steps=trace['steps'][lo:hi], rows=trace['rows'][lo*len(UNITS):hi*len(UNITS)]), dict(output_tokens=128))
            for row in trace['rows']: groups[row['phase']][row['name']].append(row['latency_ms'])
        outputs = [{k:r[k] for k in ('prompt_sha256','output_sha256')} for r in rows]
        result.update(identity=run_identity, audit=audit, synthetic=cfg['backend']=='mock', real_weights=True,
                      real_cuda=True, split=split, mode=mode, frequency_applied=real,
                      coverage_complete=custom, output_checked=True, outputs=outputs, requests=rows,
                      tokens=len(rows)*128, mean_latency_s=statistics.mean(r['elapsed_s'] for r in rows),
                      energy=energy, readback_before=before, readback_after=after,
                      background_before=processes_before, background_after=background(a.gpu_uuid),
                      pair_id=job.get('pair_id'), selection_sha256=job.get('selection_sha256'),
                      latencies={p:{u:statistics.mean(v) for u,v in units.items()} for p,units in groups.items()} if custom else None)
    finally:
        cleanup_errors = []
        if meter:
            try: meter.stop()
            except Exception as exc: cleanup_errors.append('meter: '+repr(exc))
            finally: write(out/'telemetry.json',meter.rows)
        if llm and configured:
            try: rpc(llm,'sdvfs_export',close=True)
            except Exception as exc: cleanup_errors.append('controller restore: '+repr(exc))
        if monitor:
            try: monitor.close()
            except Exception as exc: cleanup_errors.append('telemetry close: '+repr(exc))
        if auto_controller:
            try: auto_controller.close()
            except Exception as exc: cleanup_errors.append('auto reset: '+repr(exc))
        if llm:
            try: llm.llm_engine.engine_core.shutdown()
            except Exception as exc: cleanup_errors.append('worker shutdown: '+repr(exc))
        result['restoration_ok'] = not cleanup_errors
        result['cleanup_errors'] = cleanup_errors
        write(out/'result.json', result)
        if cleanup_errors: raise RuntimeError('; '.join(cleanup_errors))
    return result


def launch(a, folder, **job):
    folder = Path(folder).resolve(); folder.mkdir(parents=True, exist_ok=True)
    path = folder/'job.json'; write(path,job)
    cmd = [sys.executable, '-B','-m','rtx5090.conda','_window','--model',str(Path(a.model).resolve()),
           '--gpu-uuid',a.gpu_uuid,'--out',str(folder),'--library-dir',str(Path(a.library_dir).resolve()),
           '--dataset',str(Path(a.dataset).resolve()),'--baseline',str(Path(a.baseline).resolve()),'--job',str(path),
           '--min-seconds',str(a.min_seconds)]
    for flag,value in (('--calibration',a.calibration),('--power-evidence',a.power_evidence)):
        if value: cmd += [flag,str(Path(value).resolve())]
    if a.apply: cmd.append('--apply')
    if a.ack_default_auto: cmd.append('--ack-default-auto')
    print('Running',folder.name,job['mode'],flush=True)
    with (folder/'process.log').open('w',encoding='utf-8') as log:
        subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    return read(folder/'result.json')


def dispatch(a):
    out = Path(a.out); command = a.command
    if command == 'dataset': make_dataset(a.model,out/'dataset.json'); return
    if command == 'smoke': smoke(a); return
    if command == 'baseline': baseline(a); return
    if command == '_window': window(a,read(a.job)); return
    if command == 'probe': probe(a); return
    if command == 'recover':
        cfg=gpu_config(a); require_apply(a,cfg)
        from .controller import Controller
        with Controller(cfg,out/'recovery') as controller: controller.backend.reset()
        write(out/'recovery.json',dict(reset_returned=True,physical_restore_verified=False)); return
    if command == 'calibrate':
        cfg=gpu_config(a); require_apply(a,cfg)
        import torch
        assert_cuda_uuid(torch,a.gpu_uuid)
        calibrate(cfg,out/'calibration.json'); return
    if command == 'chain':
        cfg=dict(backend='mock',uuid=a.gpu_uuid,rates_mhz=[900,1500])
        if a.apply: raise ValueError('Mock chain cannot --apply')
        launch(a,out,mode='serial',gpu=cfg,chain=True,requests=2,split='train'); return
    if command == 'profile':
        cfg=gpu_config(a); require_apply(a,cfg); verify_model(a.model,hashes=True)
        cal=read(a.calibration)
        if cal['gpu'] != cfg or cal.get('synthetic') or not cal.get('measured') or not cal.get('restore_returned'):
            raise ValueError('Matching real calibration required')
        start=time.monotonic(); samples=[]; active_identity=None
        for rate in cfg['rates_mhz']:
            row=launch(a,out/str(rate),mode='fixed',gpu=cfg,mhz=rate,split='train')
            if active_identity is not None and active_identity!=row['identity']: raise ValueError('Profile identity changed')
            active_identity=row['identity']
            samples.append(dict(mhz=rate,latencies=row['latencies'],energy=row['energy'],result_file=str(out/str(rate)/'result.json')))
        write(out/'profile.json',dict(identity=active_identity,synthetic=False,samples=samples,
              switch_us=cal['switch_us'],switch_source='measured',calibration_sha256=digest_json(cal),
              energy_measured=True,offline_profile_wall_s=time.monotonic()-start)); return
    if command == 'plan':
        from .planning import candidate
        profile=read(a.profile)
        for i,slowdown in enumerate((0,.05,.15)):
            write(out/f'candidate_{i}.json',candidate(profile,a.profile,slowdown))
        return
    if command == 'tune': tune(a); return
    if command == 'run':
        cfg=gpu_config(a); require_apply(a,cfg); verify_model(a.model,hashes=True)
        if a.mode in ('serial','lookahead') and not a.plan: raise ValueError('--plan required')
        if a.mode=='fixed' and a.mhz not in cfg['rates_mhz']: raise ValueError('--mhz must be in calibrated grid')
        launch(a,out,mode=a.mode,gpu=cfg,plan=str(Path(a.plan).resolve()) if a.plan else None,
               mhz=a.mhz or cfg['rates_mhz'][0],split=a.split,requests=a.requests)
        return
    if command == 'compare': compare(a); return
    if command == 'report':
        report=acceptance(read(out/'pairs.json'))
        write(out/'report.json',report); print(json.dumps(report,indent=2)); return
    raise ValueError('Unsupported command')


def tune(a):
    from .planning import candidate
    cfg=gpu_config(a); require_apply(a,cfg); verify_model(a.model,hashes=True)
    out=Path(a.out); profile=read(a.profile); start=time.monotonic()
    auto=launch(a,out/'auto_pilot',mode='auto',gpu=cfg,split='train')
    count=math.ceil(len(auto['requests'])*1.5/8)*8
    auto=launch(a,out/'auto_train',mode='auto',gpu=cfg,requests=count,split='train')
    trials=[]
    for i,slow in enumerate((0,.05,.15)):
        plan=out/f'candidate_{i}.json'; write(plan,candidate(profile,a.profile,slow))
        row=launch(a,out/f'serial_{i}',mode='serial',gpu=cfg,plan=str(plan.resolve()),requests=count,split='train')
        trials.append(dict(plan=str(plan.resolve()),mode='serial',result=row))
    def eligible(item):
        row=item['result']
        if row['identity']!=auto['identity'] or row['outputs']!=auto['outputs']:
            raise ValueError('Tuning identity/output mismatch')
        return row['mean_latency_s']<=auto['mean_latency_s']*1.05
    viable=[t for t in trials if eligible(t)]
    if viable:
        serial=min(viable,key=lambda t:t['result']['energy']['energy_j']/t['result']['tokens'])
        row=launch(a,out/'lookahead',mode='lookahead',gpu=cfg,plan=serial['plan'],requests=count,split='train')
        trials.append(dict(plan=serial['plan'],mode='lookahead',result=row))
        margin=row['energy']['uncertainty_fraction']+serial['result']['energy']['uncertainty_fraction']
        if eligible(trials[-1]) and row['energy']['energy_j'] < serial['result']['energy']['energy_j']*(1-margin):
            viable.append(trials[-1])
    write(out/'tuning.json',dict(auto=auto,trials=trials,offline_tuning_wall_s=time.monotonic()-start))
    if not viable: raise RuntimeError('No candidate met training latency limit; no selection was produced')
    best=min(viable,key=lambda t:t['result']['energy']['energy_j']/t['result']['tokens'])
    value=dict(schema=1,plan=best['plan'],plan_sha256=digest_json(read(best['plan'])),mode=best['mode'],
               identity=auto['identity'],dataset_sha256=digest_json(read(a.dataset)),
               gpu=cfg,calibration_sha256=digest_json(read(a.calibration)),
               selected_on='train_only',training_energy_change=best['result']['energy']['energy_j']/auto['energy']['energy_j']-1,
               success_claim=False)
    write(out/'selection.json',value)


def compare(a):
    cfg=gpu_config(a); require_apply(a,cfg); verify_model(a.model,hashes=True)
    selection=read(a.selection); out=Path(a.out)
    if a.pairs<10: raise ValueError('At least 10 pairs required')
    if (selection.get('selected_on')!='train_only' or selection['gpu']!=cfg or
            selection['dataset_sha256']!=digest_json(read(a.dataset)) or
            selection['plan_sha256']!=digest_json(read(selection['plan'])) or
            selection['calibration_sha256']!=digest_json(read(a.calibration))):
        raise ValueError('Frozen selection/configuration changed')
    # Determine duration on train, never optimize from held-out results.
    pilot=launch(a,out/'duration_pilot',mode='auto',gpu=cfg,split='train')
    count=math.ceil(len(pilot['requests'])*2/8)*8
    pairs=[]; rng=random.Random(5090)
    for i in range(a.pairs):
        order=['auto','sparse']; rng.shuffle(order); pair=dict(pair_id=i,order=order)
        for arm in order:
            row=launch(a,out/f'pair_{i:02d}_{arm}',mode='auto' if arm=='auto' else selection['mode'],
                       gpu=cfg,plan=selection['plan'] if arm=='sparse' else None,requests=count,split='test',
                       pair_id=i,selection_sha256=digest_json(selection))
            if row['identity']!=selection['identity']: raise ValueError('Runtime differs from selected plan')
            pair[arm]=row
        pairs.append(pair); write(out/'pairs.json',pairs)
    report=acceptance(pairs); write(out/'report.json',report)
    print(json.dumps(report,indent=2))
