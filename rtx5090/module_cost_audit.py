"""Offline duration-weighted module event diagnostic; never operator energy."""
import argparse
import math
from pathlib import Path
from statistics import median
from .eager_core import UNITS
from .observe_core import GPU_UUIDS, digest
from .service_core import read, write_new
from .request_budget_partition import plan


def audit(data):
    if data.get('status') != 'two_setting_modules_timing_complete' or not data['restoration']['verified']:
        raise ValueError('Require completed and restored module collection')
    contexts = ['prefill'] + ['decode.%d' % i for i in range(63)]
    nodes = [c+'/'+u for c in contexts for u in UNITS]
    expected = {'%d:%s' % (r,n) for r in (0,1) for n in nodes}
    windows = data['windows']
    if len(windows) != 6 or {(w['repeat'],w['requested_mhz']) for w in windows} != {(i,f) for i in range(3) for f in (2250,2527)}:
        raise ValueError('Require all three requests per setting')
    if len({w['prompt_hash'] for w in windows}) != 6:
        raise ValueError('Repeated measurement prompts')
    for w in windows:
        if set(w['operators']) != expected:
            raise ValueError('Missing/extra module contexts')
        for key, op in w['operators'].items():
            if op['uuid'] != GPU_UUIDS[int(key[0])]:
                raise ValueError('Rank/UUID mismatch')
            if not math.isfinite(op['median_s']) or op['median_s'] < 0:
                raise ValueError('Invalid event time')

    def total(window, members, rank):
        if rank == 'pair_projection':
            return sum(max(window['operators']['%d:%s'%(r,n)]['median_s'] for r in (0,1)) for n in members)
        return sum(window['operators']['%d:%s'%(rank,n)]['median_s'] for n in members)

    def cell(members, rank):
        values = {str(f): [total(w,members,rank) for w in windows if w['requested_mhz']==f] for f in (2250,2527)}
        lo, hi = median(values['2250']),median(values['2527'])
        return dict(request_event_sums_s=values,median_2250_s=lo,median_2527_s=hi,
                    ratio=lo/hi if hi>0 else None)

    rows=[]; heavy=[]
    for phase in ('prefill','decode'):
        members=[n for n in nodes if n.startswith(phase)]
        for rank in (0,1,'pair_projection'):
            rows.append(dict(phase=phase,rank=rank,**cell(members,rank)))
        unit_rows=[dict(unit=u,**cell([n for n in members if n.split('/')[1]==u],'pair_projection')) for u in UNITS]
        denominator=sum(v['median_2527_s'] for v in unit_rows)
        for v in unit_rows:
            v['high_event_share']=v['median_2527_s']/denominator if denominator>0 else None
            v['phase']=phase
        heavy.extend(sorted(unit_rows,key=lambda x:x['median_2527_s'],reverse=True))
    return dict(schema='module-cost-audit-v1',source_sha256=digest(data),
                identity_fingerprint=data['identity_fingerprint'],clock_writes=False,deployable=False,
                phase_sums=rows,module_ranking=heavy,
                note='Median of three per-request sums, not median of individual ratios. '
                     'Pair projection is sum of per-unit rank maxima, not measured critical path. '
                     'Phase totals and shares are current-stream event diagnostics, not HTTP latency or energy. '
                     'Distinct prompts are not paired equivalence or independent validation.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection',required=True);p.add_argument('--out',required=True)
    a=p.parse_args(); source=Path(a.collection)
    data=read(source/'collection.json'); result=audit(data)
    budget_input=read(source/'module-budget-input.json')
    if budget_input['identity_fingerprint'] != data['identity_fingerprint']:
        raise ValueError('Budget input identity mismatch')
    budget=plan(budget_input)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    write_new(out/'weighted-events.json',result)
    write_new(out/'budget-input-audit.json',budget)
    lines=['Offline weighted event sums; no sampling/writes.',
           'rank | phase | 2250 event-sum ms | 2527 event-sum ms | change %']
    def change(row): return 'unresolved' if row['ratio'] is None else '%.2f'%((row['ratio']-1)*100)
    for row in result['phase_sums']:
        lines.append('%s | %s | %.3f | %.3f | %s'%(row['rank'],row['phase'],row['median_2250_s']*1000,row['median_2527_s']*1000,change(row)))
    lines.append('Top module event shares (pair projection; not board energy allocation):')
    for phase in ('prefill','decode'):
        for row in [v for v in result['module_ranking'] if v['phase']==phase][:8]:
            lines.append('%s | %s | high share %.2f%% | change %s%%' %
                         (phase,row['unit'],100*(row['high_event_share'] or 0),change(row)))
    lines.append('blockers: '+str(budget['blockers']))
    text='\n'.join(lines);(out/'REPORT.txt').write_text(text+'\n',encoding='utf-8')
    print(text);print('Saved:',out)


if __name__=='__main__': main()
