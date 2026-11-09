"""Independent synthetic primitives in the resident CUDA context; no clock writes.
Collected labels are not capacity parameters or same-path model certification.
"""
import time
from .observe_core import digest,write_json
from .stateless_calibration_adapter import prepare,invoke
from .performance import safe_error


def cases():
    return [dict(id=f'{role}-{op}-{n}',role=role,operator=op,n=n)
            for role,sizes in [('train',(256,512)),('holdout',(384,640))]
            for op in ('linear','add','mul') for n in sizes]


def collect(torch,rank,identity,out):
    from pathlib import Path
    from sparsedvfs_vllm.operators import tensor_metadata
    path=Path(out)/f'rank-{rank}-independent-primitives.json'
    rows=[];result=dict(status='failed',rank=rank,uuid=identity['gpu_uuid'],
        identity_sha256=digest(identity),rows=rows,clock_writes=False,
        parameters_validated=False,same_path_verified=False,synthetic_inputs=True,
        scope='resident_context_independent_small_primitives_not_MLP_capacity_or_operator_power')
    split={role:[c['id'] for c in cases() if c['role']==role] for role in ('train','holdout')}
    result['split']=split
    try:
        if torch.cuda.is_current_stream_capturing():raise RuntimeError('CUDA capture active')
        # At most two 640x640 BF16 operands and transient output; no model weights copied.
        free,_=torch.cuda.mem_get_info()
        if free<32*2**20:raise RuntimeError('Less than 32 MiB free; no allocation attempted')
        with torch.inference_mode():
            for index,case in enumerate(cases()):
                generator=torch.Generator(device=f'cuda:{rank}').manual_seed(119000+rank*100+index)
                n=case['n'];a=torch.randn((n,n),device=f'cuda:{rank}',dtype=torch.bfloat16,generator=generator)
                b=torch.randn((n,n),device=f'cuda:{rank}',dtype=torch.bfloat16,generator=generator)
                func={'linear':torch.ops.aten.linear.default,'add':torch.ops.aten.add.Tensor,
                      'mul':torch.ops.aten.mul.Tensor}[case['operator']]
                args=(a,b);answer=func(*args)
                def metadata():
                    return dict(overload=str(func),inputs=tensor_metadata(dict(args=args,kwargs={}),torch),
                                outputs=tensor_metadata(answer,torch))
                meta=metadata();spec=prepare(meta,case['role'],case['id'],split)
                row=dict(case,seed=119000+rank*100+index,metadata=meta,work=spec['work'],
                         batches=[],input_origin='fresh_generator_not_inference_tensors',
                         kernel_verified=False,cache_state_matched=False)
                rows.append(row)
                # Guarded actual operand binding before the timed region; do not time Python metadata hashing.
                checked=invoke(spec,func,args,{},metadata());del checked
                for _ in range(3):answer=func(*args)
                torch.cuda.synchronize()
                for _ in range(3):
                    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    begin=time.monotonic_ns();start.record()
                    for _ in range(32):answer=func(*args)
                    end.record();end.synchronize()
                    row['batches'].append(dict(repeats=32,elapsed_s=start.elapsed_time(end)/1000,
                                              host_elapsed_s=(time.monotonic_ns()-begin)/1e9))
                del answer,a,b,args
        result['status']='independent_primitive_labels_collected_not_parameters_validated'
    except BaseException as exc:
        result['error']=safe_error(exc)
        raise
    finally:write_json(path,result)
    return result
