"""WSL/container entry point. Real GPU paths are intentionally fail-closed."""
import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path
from .contracts import MODEL_ID, REVISION, UNITS, engine_args, digest_json, verify_model


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path, value):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def download(path):
    from huggingface_hub import snapshot_download
    root=Path(snapshot_download(MODEL_ID, revision=REVISION, local_dir=path))
    files={}
    for p in root.rglob('*'):
        if p.is_file() and '.cache' not in p.parts and p.name!='sparsedvfs_model_lock.json':
            with p.open('rb') as f: sha=hashlib.file_digest(f,'sha256').hexdigest()
            files[str(p.relative_to(root))]=dict(size=p.stat().st_size,sha256=sha)
    write(root/'sparsedvfs_model_lock.json',dict(model_id=MODEL_ID,revision=REVISION,files=files))
    verify_model(root,hashes=True)


def workload(model='/model', output_tokens=128):
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
    seed=tokenizer.encode('Explain why frequency scheduling needs measured execution boundaries.',add_special_tokens=False)
    ids=(seed*512)[:512]
    return dict(input_tokens=512,output_tokens=output_tokens,input_token_ids=ids,prompt_sha256=digest_json(ids),
                temperature=0.0,seed=0,ignore_eos=True,add_special_tokens=False,warmups=2)


def public(data): return {k:v for k,v in data.items() if k!='input_token_ids'}
def rpc(llm,name,**kwargs):
    values=llm.collective_rpc(name,kwargs=kwargs)
    if len(values)!=1: raise RuntimeError('Expected one GPU worker')
    return values[0]


def generate(llm,data):
    from vllm import SamplingParams
    start=time.perf_counter()
    output=llm.generate([{'prompt_token_ids':data['input_token_ids']}],SamplingParams(
        temperature=0.0,seed=0,max_tokens=data['output_tokens'],min_tokens=data['output_tokens'],ignore_eos=True),use_tqdm=False)[0]
    ids=list(output.outputs[0].token_ids)
    if len(ids)!=data['output_tokens']: raise RuntimeError('Incomplete output')
    return dict(output_token_ids=ids,output_sha256=digest_json(ids),elapsed_s=time.perf_counter()-start)


def validate_trace(trace,data):
    steps=trace['steps']
    if len(steps)!=data['output_tokens'] or [s['phase'] for s in steps]!=['prefill']+['decode']*(len(steps)-1):
        raise ValueError('Unexpected prefill/decode sequence')
    if any(s['order']!=UNITS for s in steps) or len(trace['rows'])!=len(UNITS)*len(steps):
        raise ValueError('Incomplete module coverage')
    if len({row['stream'] for row in trace['rows']})!=1: raise ValueError('Multiple CUDA streams')


def build_plan(profile,profile_file,slowdown=0.05):
    if not 0<=slowdown<=1: raise ValueError('slowdown must be in [0,1]')
    switch=profile['switch_us']
    if switch<=0: raise ValueError('Missing switch latency')
    samples=profile['samples']; blocks=[]
    for phase in ('prefill','decode'):
        choices=[]
        for unit in UNITS:
            rows=[(int(s['mhz']),s['latencies'][phase][unit]) for s in samples]
            fastest=min(t for _,t in rows)
            mhz,ms=min((f,t) for f,t in rows if t<=fastest*(1+slowdown))
            choices.append((unit,mhz,ms))
        for unit,mhz,ms in choices:
            if blocks and blocks[-1]['phase']==phase and blocks[-1]['triplet'][1]==mhz*1000000:
                blocks[-1]['operators'].append(unit); blocks[-1]['estimated_ms']+=ms
            else: blocks.append(dict(block_id=len(blocks),phase=phase,operators=[unit],triplet=[0,mhz*1000000,0],estimated_ms=ms))
    for b in blocks: b['amortized_strict_gt']=b['estimated_ms']*1000>switch
    return dict(schema=5,identity=profile['identity'],synthetic=profile['synthetic'],
                granularity='module',selector='measured_latency_budget',units=UNITS,
                profile_file=str(Path(profile_file).resolve()),profile_sha256=digest_json(profile),
                switch_us=switch,blocks=blocks,slowdown_budget=slowdown,
                caveat='Lowest eligible GPU clock is a heuristic, not measured energy optimization')


def calibrate(config,out):
    from .controller import Controller
    from .native import environment_record
    if Path(out).exists(): raise FileExistsError('Use a fresh calibration output; do not overwrite prior evidence')
    keepalive=lambda:None
    if config['backend']!='mock':
        import torch
        tensor=torch.ones((1024,1024),device='cuda',dtype=torch.bfloat16)
        def keepalive():
            _=tensor@tensor
            torch.cuda.current_stream().synchronize()
    rows=[]
    with Controller(config,Path(out).parent/'calibration_controller') as c:
        for source in c.rates:
            for target in c.rates:
                if source==target: continue
                for repeat in range(5):
                    for rate in (source,target):
                        start=time.monotonic_ns(); c.set([0,rate*1000000,0]); returned=time.monotonic_ns()
                        deadline=time.monotonic()+2
                        while True:
                            keepalive()
                            observed=c.backend.read()
                            if abs(observed-rate)<=15: break
                            if time.monotonic()>deadline: raise RuntimeError('GPU clock did not match requested rate')
                            time.sleep(.001)
                        matched=time.monotonic_ns()
                        if rate==target: rows.append(dict(source_mhz=source,target_mhz=target,repeat=repeat,
                            call_us=(returned-start)/1000,matching_readback_us=(matched-start)/1000))
        if not rows: raise ValueError('At least two rates required for calibration')
        ordered=sorted(r['matching_readback_us'] for r in rows)
        result=dict(gpu=config,synthetic=c.backend.synthetic,records=rows,
            environment=environment_record(),
            switch_us=ordered[min(len(ordered)-1,int(len(ordered)*.95))],
            measured=True,physical_pll_settle_verified=False,clock_domain='caller_monotonic')
    # Publish usable calibration only AFTER Controller.__exit__ has reset successfully.
    result['restore_returned'] = True
    write(out,result)


def infer(args):
    from vllm import LLM
    verify_model(args.model,hashes=True)
    data=workload(args.model); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    native=args.command=='baseline'
    gpu=read(args.gpu) if not native else None
    real=gpu is not None and gpu['backend']!='mock'
    if real and not args.apply: raise ValueError('Real NVML requires --apply')
    if real: os.environ['SPARSEDVFS_APPLY']='1'
    llm=LLM(**engine_args(args.model,custom=not native))
    if native:
        for _ in range(2): generate(llm,data)
        rows=[generate(llm,data) for _ in range(args.repeat)]
        write(out/'baseline.json',dict(workload=public(data),runs=rows,model_id=MODEL_ID,revision=REVISION))
        return
    baseline=read(args.baseline)
    if baseline['workload']!=public(data) or baseline['model_id']!=MODEL_ID or baseline['revision']!=REVISION:
        raise ValueError('Baseline differs from locked workload/model')
    hashes={r['output_sha256'] for r in baseline['runs']}
    if len(hashes)!=1: raise ValueError('Baseline outputs are not deterministic')
    calibration=read(args.calibration) if args.calibration else None
    if calibration and (calibration['gpu']!=gpu or not calibration.get('measured')):
        raise ValueError('Calibration backend/grid differs')
    if real and (not calibration or calibration['synthetic']): raise ValueError('Real control requires measured real calibration')
    samples=[]; results=[]; identity=None
    rates=gpu['rates_mhz'] if args.command=='profile' else [None]
    for rate in rates:
        mode='fixed' if rate is not None else args.mode
        cfg=dict(mode=mode,gpu=gpu,model=args.model,enabled=True,apply=real,workload=public(data),granularity='module',
                 out_dir=str((out/str(rate or mode)).resolve()),fixed_triplet=[0,(rate or gpu['rates_mhz'][0])*1000000,0],
                 switch_calibrated=bool(calibration and calibration.get('measured')))
        if args.calibration: cfg['calibration']=str(Path(args.calibration).resolve())
        if mode in ('serial','lookahead'): cfg['plan']=str(Path(args.plan).resolve())
        identity=rpc(llm,'sdvfs_configure',config=cfg)
        try:
            rpc(llm,'sdvfs_prepare')
            for _ in range(2): generate(llm,data)
            rpc(llm,'sdvfs_export',reset=True)
            before=rpc(llm,'sdvfs_gpu_readback')
            rows=[generate(llm,data) for _ in range(args.repeat)]
            after=rpc(llm,'sdvfs_gpu_readback')
            if real and rate and any(abs(x['observed_mhz']-rate)>15 for x in (before,after)):
                raise RuntimeError('Fixed profile GPU clock readback does not match target')
            trace=rpc(llm,'sdvfs_export',reset=True,close=True)
            # Each repetition has one prefill and output_tokens-1 decode steps.
            for i in range(args.repeat):
                start=i*data['output_tokens']; end=start+data['output_tokens']
                validate_trace(dict(steps=trace['steps'][start:end],rows=trace['rows'][start*len(UNITS):end*len(UNITS)]),data)
            if {r['output_sha256'] for r in rows}!=hashes: raise RuntimeError('Output differs from native quantized baseline')
            groups={p:{u:[] for u in UNITS} for p in ('prefill','decode')}
            for row in trace['rows']: groups[row['phase']][row['name']].append(row['latency_ms'])
            if rate: samples.append(dict(mhz=rate,readback_before=before,readback_after=after,
                latencies={p:{u:statistics.mean(v) for u,v in units.items()} for p,units in groups.items()}))
            results.extend(rows)
        finally: rpc(llm,'sdvfs_export',close=True)
    if args.command=='profile':
        write(out/'profile.json',dict(identity=identity,synthetic=not real,samples=samples,
            switch_us=calibration['switch_us'] if calibration else 1000,
            calibration_sha256=digest_json(calibration) if calibration else None,
            switch_source='measured' if calibration else 'assumed_mock_only',energy_measured=False))
    write(out/'results.json',dict(runs=results,real_cuda_executed=True,real_frequency_control=real,
        output_matches_baseline=True,energy_measured=False))


def main():
    p=argparse.ArgumentParser(); p.add_argument('command',choices=['download','probe','calibrate','baseline','profile','plan','run'])
    p.add_argument('--out',default='/results'); p.add_argument('--gpu'); p.add_argument('--apply',action='store_true')
    p.add_argument('--model',default=os.environ.get('SPARSEDVFS_MODEL','/model'))
    p.add_argument('--baseline'); p.add_argument('--calibration'); p.add_argument('--profile'); p.add_argument('--plan')
    p.add_argument('--mode',choices=['instrumented','fixed','serial','lookahead'],default='serial')
    p.add_argument('--repeat',type=int,default=3); p.add_argument('--slowdown',type=float,default=.05)
    a=p.parse_args()
    if a.repeat<1: p.error('repeat must be positive')
    if a.command=='download': download(a.out)
    elif a.command=='probe':
        from .backends import backend
        b=backend(read(a.gpu))
        try: write(a.out,b.probe())
        finally: b.close()
    elif a.command=='calibrate':
        config=read(a.gpu)
        if config['backend']!='mock' and not a.apply: p.error('calibrate real clocks requires --apply')
        calibrate(config,a.out)
    elif a.command=='plan': write(a.out,build_plan(read(a.profile),a.profile,a.slowdown))
    else: infer(a)


if __name__=='__main__': main()
