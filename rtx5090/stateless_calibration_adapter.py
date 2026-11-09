"""Restricted worker-side call adapter. No collector, parameter fit or clock writes.

Independent calibration inputs must be supplied by the caller. Original inference
operands must never be passed here and then described as independent calibration.
"""
from .operator_parameter_gaps import linear_work,elementwise_work
from .observe_core import digest

SUPPORTED={'aten.linear.default','aten.add.Tensor','aten.sub.Tensor','aten.mul.Tensor'}


def prepare(meta, role, sample_id, split_ids):
    if role not in ('train','holdout'): raise ValueError('Explicit sample role required')
    train=split_ids.get('train',[]);held=split_ids.get('holdout',[])
    if not train or not held or len(set(train))!=len(train) or len(set(held))!=len(held) or set(train)&set(held):
        raise ValueError('Distinct nonempty training and holdout sample IDs required')
    if sample_id not in split_ids[role]: raise ValueError('Sample not assigned to this split')
    if meta.get('overload') not in SUPPORTED: raise ValueError('No stateless replay adapter for this operator')
    work=linear_work(meta) or elementwise_work(meta)
    if work is None: raise ValueError('Unsupported tensor/workload contract')
    return dict(schema='stateless-calibration-call-v1',signature_sha256=digest(meta),
                signature=meta,role=role,sample_id=sample_id,split_sha256=digest(split_ids),
                work=work,parameters_validated=False,independent_inputs_verified=False)


def invoke(spec,func,args,kwargs,actual_meta):
    # Exact rank/device/layout contract: normalization is deliberately not used.
    if spec.get('schema')!='stateless-calibration-call-v1' or digest(actual_meta)!=spec['signature_sha256']:
        raise ValueError('Actual call metadata differs from prepared calibration signature')
    if str(func)!=actual_meta['overload'] or str(func) not in SUPPORTED:
        raise ValueError('Callable is not the prepared stateless overload')
    schema=getattr(func,'_schema',None)
    if schema is None or getattr(schema,'is_mutable',True):
        raise ValueError('Read-only dispatcher schema required')
    # Execute once. CUDA errors propagate; there is no retry, state recovery or
    # silent fallback. A collector must provide evidence that metadata is real.
    return func(*args,**kwargs)
