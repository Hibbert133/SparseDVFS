"""Conservative logical traffic definitions; no DRAM or conversion capacity claim."""
import math

WIDTHS={'torch.bool':1,'torch.uint8':1,'torch.int8':1,'torch.int16':2,
        'torch.int32':4,'torch.int64':8,'torch.float16':2,'torch.bfloat16':2,
        'torch.float32':4,'torch.float64':8}


def dense_tensor(t):
    if not isinstance(t,dict) or t.get('tensor') is not True: return False
    if t.get('dtype') not in WIDTHS or t.get('layout')!='torch.strided': return False
    shape=t.get('shape');stride=t.get('stride')
    if not isinstance(shape,list) or not isinstance(stride,list) or len(shape)!=len(stride): return False
    if any(type(d) is not int or d<=0 for d in shape): return False
    expected=1
    for size,step in zip(reversed(shape),reversed(stride)):
        if type(step) is not int or step<0 or (size!=1 and step!=expected): return False
        expected*=size
    return True


def memory_work(meta,kernels):
    # No GPU activity could mean alias/no-op/memcpy association missing: do not infer traffic.
    if not kernels: return None
    op=meta.get('overload');args=meta.get('inputs',{}).get('args',[])
    output=meta.get('outputs')
    if op=='aten.copy_.default':
        if len(args) not in (2,3): return None
        destination,source=args[:2]
    elif op in ('aten.to.dtype','aten.to.dtype_layout'):
        if not args: return None
        source=args[0];destination=output
    else: return None
    if not all(dense_tensor(t) for t in (source,destination,output)): return None
    if source['shape']!=destination['shape'] or output!=destination: return None
    devices=[t.get('device') for t in (source,destination)]
    if devices[0] not in ('cuda:0','cuda:1') or devices[0]!=devices[1]: return None
    converted=source['dtype']!=destination['dtype']
    if op!='aten.copy_.default' and not converted: return None  # possible identity/alias
    count=math.prod(source['shape'])
    return dict(work_flop=None if converted else 0,
                data_byte=count*(WIDTHS[source['dtype']]+WIDTHS[destination['dtype']]),
                derivation='dtype_conversion_logical_bytes_only_v1' if converted else 'dense_same_dtype_copy_v1',
                unresolved=['conversion_compute_path'] if converted else [],
                note='logical one-read one-write tensor footprint; alias, physical traffic and execution capacity unverified')
