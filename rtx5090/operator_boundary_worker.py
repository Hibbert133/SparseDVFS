"""No-write operator rendezvous on the model thread; no NCCL calls added."""
import json
import threading
import time
from pathlib import Path
import torch
from .observe_worker import Observer
from .observe_core import digest
from .eager_core import context_key
from .eager_ipc import rpc
from sparsedvfs_vllm.operators import tensor_metadata


class BoundaryObserver(Observer):
    def __init__(self,model,rank,context,command,identity):
        super().__init__(model,rank,context,60000)
        self.command=command;self.identity=identity;self.ordinals={};self.seen=[]
        self.candidates={c['context']:c for c in command['boundaries']}
        self.path=Path(command['out'])/f'rank-{rank}-boundary.jsonl'
        self.journal_lock=threading.Lock()

    def log(self,**row):
        with self.journal_lock:
            with self.path.open('a',encoding='utf-8') as f:
                f.write(json.dumps(dict(rank=self.rank,time_ns=time.monotonic_ns(),**row))+'\n')

    def dispatch(self,func,args,kwargs):
        if self.busy:return func(*args,**kwargs)
        context=self.context()
        if context is None:return func(*args,**kwargs)
        key=context_key(context);candidate=self.candidates.get(key)
        if candidate is None:return func(*args,**kwargs)
        index=self.ordinals.get(key,0);self.ordinals[key]=index+1
        if index!=candidate['source_ordinals'][self.rank]:return func(*args,**kwargs)
        self.busy=True
        try:
            actual=dict(overload=str(func),inputs=tensor_metadata(dict(args=args,kwargs=kwargs),torch))
            expected=candidate['contracts'][self.rank]
            if (self.parents[-1]!=candidate['parent'] or
                actual!=dict(overload=expected['overload'],inputs=expected['inputs'])):
                raise ValueError('Boundary actual input/parent contract mismatch')
            sequence=len(self.seen)
            if key!=self.command['boundaries'][sequence]['context']:raise ValueError('Boundary context order mismatch')
            self.log(event='gpu_wait_begin',context=key,ordinal=index)
            expired=threading.Event()
            def timeout():
                expired.set();self.log(event='gpu_wait_timeout',context=key)
                try:rpc(self.command['coordinator'],dict(op='abort',fingerprint=self.command['fingerprint'],reason='GPU completion timeout'),2)
                except Exception:pass
            timer=threading.Timer(10,timeout);timer.daemon=True;timer.start()
            begin=time.monotonic_ns()
            try:torch.cuda.synchronize()
            finally:timer.cancel()
            if expired.is_set():raise TimeoutError('GPU completion exceeded 10s; boundary not released')
            synced=time.monotonic_ns();self.log(event='gpu_complete',context=key,gpu_wait_ns=synced-begin)
            response=rpc(self.command['coordinator'],dict(rank=self.rank,uuid=self.identity['gpu_uuid'],
                fingerprint=self.command['fingerprint'],seq=sequence,gpu_complete=True,
                key=[context['request_id'],key,sequence]),12)
            if response!=dict(released=True,seq=sequence,clock_writes=False):raise ValueError('Invalid rendezvous release')
            self.log(event='released',context=key,seq=sequence,rendezvous_ns=time.monotonic_ns()-synced)
            result=func(*args,**kwargs)
            if tensor_metadata(result,torch)!=expected['outputs']:raise ValueError('Boundary output contract mismatch')
            self.seen.append(key)
            return result
        except BaseException as exc:
            self.log(event='failed',context=key,error=type(exc).__name__+': '+str(exc))
            try:rpc(self.command['coordinator'],dict(op='abort',fingerprint=self.command['fingerprint'],reason='Rank boundary failed'),2)
            except Exception:pass
            raise
        finally:self.busy=False

    def finish(self):
        if self.seen!=[c['context'] for c in self.command['boundaries']]:
            raise ValueError('Missing operator rendezvous boundary')
