"""Read-only decomposition of stored observed-band transition traces."""
import argparse
import csv
from pathlib import Path
from statistics import median
from .observe_core import GPU_UUIDS,digest
from .service_core import read,write_new


def analyze(row,bands):
    frames=row['loaded_samples']; begin=row['begin_ns']; api=row['api_return_ns']
    if not frames: raise ValueError('No stored polling frames')
    end=row.get('confirmed_ns',max(s['end_ns'] for s in frames[-1]))
    if not frames or not begin<=api<=end: raise ValueError('Invalid transition timing')
    group_bounds=[]; paired=[]; clock_paired=[]; flat=[]
    for index,frame in enumerate(frames):
        if len(frame)!=2 or {s['uuid'] for s in frame}!=set(GPU_UUIDS): raise ValueError('Missing dual-card sample')
        low=min(s['begin_ns'] for s in frame); high=max(s['end_ns'] for s in frame)
        if low<api or high<low or (group_bounds and low<group_bounds[-1][1]):
            raise ValueError('Invalid/overlapping polling groups')
        group_bounds.append((low,high))
        good=[]; clock_good=[]
        for s in frame:
            if not low<=s['begin_ns']<=s['end_ns']<=high: raise ValueError('Invalid query bracket')
            raw=s['raw']; mhz=raw.get('clock_mhz'); util=raw.get('utilization_pct')
            inside=mhz is not None and bands[s['uuid']][0]<=mhz<=bands[s['uuid']][1]
            loaded=(util or 0)>0
            accepted=inside and loaded and not raw.get('errors')
            good.append(accepted); clock_good.append(inside)
            field=raw.get('field_query_times_ns',{}).get('clock_mhz',{})
            times=raw.get('field_query_times_ns',{})
            remote_end=max((t['end'] for t in times.values()),default=raw['timestamp_ns'])
            remote_duration=remote_end-raw['timestamp_ns']
            query_duration=s['end_ns']-s['begin_ns']
            mapped=s.get('mapped',{}).get('clock_mhz',{})
            flat.append(dict(index=index,uuid=s['uuid'],clock_mhz=mhz,utilization_pct=util,
                clock_inside=inside,loaded=loaded,accepted=accepted,
                begin_ms=(s['begin_ns']-begin)/1e6,end_ms=(s['end_ns']-begin)/1e6,
                local_query_ms=query_duration/1e6,remote_telemetry_ms=remote_duration/1e6,
                outside_remote_span_ms=(query_duration-remote_duration)/1e6,
                remote_clock_query_ms=(field['end']-field['start'])/1e6 if field else None,
                mapped_clock_midpoint_ms=(mapped['t_ns']-begin)/1e6 if mapped else None,
                mapped_clock_uncertainty_ms=mapped['uncertainty_ns']/1e6 if mapped else None))
        paired.append(all(good)); clock_paired.append(all(clock_good))
    if group_bounds[-1][1]!=end: raise ValueError('Confirmed end does not match final query bracket')
    def first(flags): return next((i for i,v in enumerate(flags) if v),None)
    first_pair=first(paired); first_clock=first(clock_paired)
    run_end=next((i for i in range(2,len(paired)) if all(paired[i-2:i+1])),None)
    if row.get('confirmed') and run_end!=len(paired)-1: raise ValueError('Stored confirmation does not match three-frame predicate')
    poll_gaps=[b[0]-a[1] for a,b in zip(group_bounds,group_bounds[1:])]
    groups=sum(b-a for a,b in group_bounds)
    initial=group_bounds[0][0]-api
    decomposition=dict(api_ms=(api-begin)/1e6,initial_dispatch_ms=initial/1e6,
                       polling_groups_ms=groups/1e6,between_group_gaps_ms=sum(poll_gaps)/1e6,
                       total_ms=(end-begin)/1e6)
    if abs(sum(decomposition[k] for k in ('api_ms','initial_dispatch_ms','polling_groups_ms','between_group_gaps_ms'))-decomposition['total_ms'])>1e-5:
        raise ValueError('Decomposition failed to account for entire interval')
    by_uuid=[]
    for uuid in GPU_UUIDS:
        rows=[r for r in flat if r['uuid']==uuid]
        hit=next((r for r in rows if r['accepted']),None)
        clock_hit=next((r for r in rows if r['clock_inside']),None)
        runs=[]
        for r in rows:
            if runs and runs[-1]['mhz']==r['clock_mhz']:
                runs[-1].update(last_query_end_ms=r['end_ms'],samples=runs[-1]['samples']+1)
            else:
                runs.append(dict(mhz=r['clock_mhz'],first_query_end_ms=r['end_ms'],last_query_end_ms=r['end_ms'],samples=1))
        by_uuid.append(dict(uuid=uuid,first_clock_in_band_ms=clock_hit['end_ms'] if clock_hit else None,
            first_loaded_in_band_ms=hit['end_ms'] if hit else None,
            max_query_ms=max(r['local_query_ms'] for r in rows),median_query_ms=median(r['local_query_ms'] for r in rows),
            max_remote_telemetry_ms=max(r['remote_telemetry_ms'] for r in rows),
            clock_in_band_but_idle_samples=sum(r['clock_inside'] and not r['loaded'] for r in rows),
            longest_queries=sorted(rows,key=lambda r:r['local_query_ms'],reverse=True)[:3],clock_value_runs=runs))
    first_time=(group_bounds[first_pair][1]-begin)/1e6 if first_pair is not None else None
    result=dict(from_requested_mhz=row['from_requested_mhz'],to_requested_mhz=row['to_requested_mhz'],
        samples_per_gpu=len(frames),decomposition=decomposition,
        first_pair_clock_in_band_ms=(group_bounds[first_clock][1]-begin)/1e6 if first_clock is not None else None,
        first_pair_loaded_in_band_ms=first_time,
        after_first_pair_until_confirmation_ms=decomposition['total_ms']-first_time if first_time is not None else None,
        final_three_frame_span_ms=(group_bounds[-1][1]-group_bounds[-3][1])/1e6 if run_end is not None else None,
        between_group_gap_ms=dict(min=min(poll_gaps)/1e6 if poll_gaps else None,
             median=median(poll_gaps)/1e6 if poll_gaps else None,max=max(poll_gaps)/1e6 if poll_gaps else None),
        by_uuid=by_uuid,
        interpretation='Local query spans include bridge lock/transport/server work and scheduling; gaps include intended sleep and scheduling. Clock-value repeats do not prove sensor refresh period.',
        physical_switch_latency_identified=False)
    return result,flat


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True); p.add_argument('--out',required=True); a=p.parse_args()
    source=Path(a.source); probe=read(source/'probe.json')
    if probe['status']!='observed_band_probe_complete' or not probe['restoration']['verified']:
        raise ValueError('Require completed probe and verified restoration')
    out=Path(a.out); out.mkdir(parents=True,exist_ok=False)
    reports=[]; timeline=[]; hashes={}
    for i,expected in enumerate(probe['transitions']):
        row=read(source/('transition-%d.json'%i))
        if digest(row)!=digest(expected): raise ValueError('Transition artifact differs from probe summary')
        hashes[str(i)]=digest(row)
        summary,flat=analyze(row,probe['bands'][str(row['to_requested_mhz'])])
        reports.append(dict(index=i,**summary))
        timeline.extend(dict(transition=i,**r) for r in flat)
    write_new(out/'audit.json',dict(status='offline_transition_audit_complete',probe_sha256=digest(probe),
        transition_sha256=hashes,identity_fingerprint=probe['identity_fingerprint'],transitions=reports,
        clock_writes=False,physical_switch_latency_identified=False))
    with (out/'timeline.csv').open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(timeline[0]));writer.writeheader();writer.writerows(timeline)
    lines=['Offline transition timing audit; all values ms, no new sampling.',
           'transition | total | API | initial dispatch | query groups | gaps | first paired loaded | remaining confirmation']
    for r in reports:
        d=r['decomposition']
        lines.append('%s->%s | %.2f | %.2f | %.2f | %.2f | %.2f | %s | %s' %
            (r['from_requested_mhz'],r['to_requested_mhz'],d['total_ms'],d['api_ms'],d['initial_dispatch_ms'],
             d['polling_groups_ms'],d['between_group_gaps_ms'],r['first_pair_loaded_in_band_ms'],r['after_first_pair_until_confirmation_ms']))
        for rank,g in enumerate(r['by_uuid']):
            lines.append('  GPU%d: first clock=%s, first loaded=%s, max query=%.2f, max remote=%.2f, in-band idle=%s' %
                (rank,g['first_clock_in_band_ms'],g['first_loaded_in_band_ms'],g['max_query_ms'],g['max_remote_telemetry_ms'],g['clock_in_band_but_idle_samples']))
        lines.append('  gap min/median/max: '+str(r['between_group_gap_ms'])+'; samples/GPU: '+str(r['samples_per_gpu']))
    lines.append('Hardware settling vs cached telemetry cannot be separated from these readbacks alone. See timeline.csv and clock_value_runs in audit.json.')
    (out/'REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines));print('Saved:',out)


if __name__=='__main__':main()
