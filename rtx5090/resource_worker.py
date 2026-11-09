"""Bounded stateless resource experiment in the resident model thread.

No weight mutation, attention replay, collective replay, profiler or clock writes.
"""
import time
from pathlib import Path
from .eager_ipc import rpc
from .observe_core import digest, write_json


FAMILIES=('linear','add','mul','copy','cast','rsqrt','reduction')


def specification(family, size, split, seed):
    if family not in FAMILIES or split not in ('train','holdout') or size not in (256,512,384):
        raise ValueError('Unknown bounded primitive')
    if (split=='holdout')!=(size==384):raise ValueError('Shape split mismatch')
    return dict(family=family,size=size,split=split,seed=int(seed))


def collect(torch, rank, identity, command):
    spec=specification(**command['resource_case']);n=spec['size'];family=spec['family']
    path=Path(command['out'])/f'rank-{rank}-resource.json'
    result=dict(status='failed',rank=rank,uuid=identity['gpu_uuid'],identity_sha256=digest(identity),
        specification=spec,clock_writes=False,same_path_verified=False,synthetic_inputs=True,
        cache_state='repeated_working_set_not_matched_to_inference',data_semantics='logical_tensor_bytes')
    try:
        if torch.cuda.is_current_stream_capturing():raise RuntimeError('Graph capture active')
        free,_=torch.cuda.mem_get_info()
        if free<32*2**20:raise RuntimeError('Need 32 MiB free; no allocation attempted')
        device=f'cuda:{rank}';gen=torch.Generator(device=device).manual_seed(spec['seed']+rank)
        # Include both memory-dominated and compute-dominated GEMMs without
        # duplicating resident model weights or exhausting the second card.
        m=1 if n==256 else n
        shape=(m,4096) if family=='linear' else (n*2048,)
        a=torch.rand(shape,device=device,dtype=torch.bfloat16,generator=gen)+.5
        b=torch.rand((1024,4096) if family=='linear' else shape,device=device,dtype=torch.bfloat16,generator=gen)
        if family=='linear': call=lambda:torch.ops.aten.linear.default(a,b)
        elif family=='add':call=lambda:torch.ops.aten.add.Tensor(a,b)
        elif family=='mul':call=lambda:torch.ops.aten.mul.Tensor(a,b)
        elif family=='copy':call=lambda:b.copy_(a)
        elif family=='cast':call=lambda:a.float()
        elif family=='rsqrt':call=lambda:torch.rsqrt(a)
        else:call=lambda:torch.sum(a)
        count=a.numel()
        result.update(shape=[m,1024,4096] if family=='linear' else list(shape),work_flop=(2*m*1024*4096 if family=='linear' else count if family in ('add','mul') else 0 if family=='copy' else None),
            data_byte=(2*(m*4096+1024*4096+m*1024) if family=='linear' else 6*count if family in ('add','mul','cast') else 4*count if family in ('copy','rsqrt') else 2*count+2),
            workload_compute_path_unresolved=family in ('cast','rsqrt','reduction'))
        for _ in range(5):answer=call()
        torch.cuda.synchronize()
        result['batches']=[]
        for repeats in (1,8,32,128):
            for _ in range(3):
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                begin=time.monotonic_ns();start.record()
                for _ in range(repeats):answer=call()
                end.record();end.synchronize()
                result['batches'].append(dict(repeats=repeats,elapsed_s=start.elapsed_time(end)/1000,
                    host_elapsed_s=(time.monotonic_ns()-begin)/1e9))
        def gate(seq):
            return rpc(command['resource_coordinator'],dict(rank=rank,uuid=identity['gpu_uuid'],
                fingerprint=command['fingerprint'],seq=seq,gpu_complete=True,
                key=[command['request_id'],'resource',seq]),timeout=45)
        torch.cuda.synchronize();gate(0)
        begin=time.monotonic_ns();iterations=0
        # Repeated homogeneous resource window; synchronization gaps retained.
        while (time.monotonic_ns()-begin)/1e9<10:
            for _ in range(128):answer=call()
            torch.cuda.synchronize();iterations+=128
        result.update(power_work_begin_ns=begin,power_work_end_ns=time.monotonic_ns(),power_iterations=iterations)
        gate(1)
        result['status']='resource_labels_collected'
        del answer,a,b
    finally:write_json(path,result)
    return result
