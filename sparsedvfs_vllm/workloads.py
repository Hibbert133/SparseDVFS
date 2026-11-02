"""Conservative static descriptors; tensor bytes are NOT measured DRAM traffic."""
import math


def describe(entry):
    result = dict(work_flop=None, data_byte=None, data_semantics=None,
                  logical_tensor_bytes=None, s_comp=0.0, s_mem=0.0,
                  workload_provenance='Unknown fused/module workload: requires target evidence')
    args = entry.get('metadata', {}).get('args', [])
    if entry.get('overload') != 'aten.mm.default' or len(args) < 2:
        return result
    a, b = args[:2]
    if not isinstance(a,dict) or not isinstance(b,dict):
        return result
    ashape, bshape = a.get('shape',[]), b.get('shape',[])
    if len(ashape) != 2 or len(bshape) != 2 or ashape[1] != bshape[0]:
        return result
    if any(not isinstance(v,int) or v <= 0 for v in ashape+bshape):
        return result
    m,k = ashape; _,n = bshape
    result['work_flop'] = 2*m*k*n
    result['workload_provenance'] = 'Static dense ATen mm arithmetic 2*M*N*K from certified input shapes; no padding/kernel work inferred'
    sizes = {'torch.bfloat16':2,'torch.float16':2,'torch.float32':4,'torch.float64':8}
    if a.get('dtype') == b.get('dtype') and a.get('dtype') in sizes:
        result['logical_tensor_bytes'] = (m*k+k*n+m*n)*sizes[a['dtype']]
    return result


def template(profile):
    certificate = profile.get('coverage_certificate',{})
    cases = certificate.get('cases',{})
    operators = {}
    for row in profile['candidates'][0]['operators']:
        entries = {e['name']:e for e in cases.get(row['phase'],{}).get('layout',[])}
        operators[row['phase']+'/'+row['name']] = describe(entries.get(row['name'],{}))
    return dict(schema=5,synthetic=bool(profile.get('synthetic')),operators=operators,
                note='Fill data_byte with measured/effective traffic plus provenance; logical_tensor_bytes is a separate lower-level descriptor, never auto-used in Eq1. Opaque fused work stays unknown.')
