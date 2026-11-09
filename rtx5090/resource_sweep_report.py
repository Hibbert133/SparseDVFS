"""Summarize an exploratory sweep, retaining failures and unmeasured settings."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from .observe_core import digest
from .service_core import read,write_new


def report(data):
    if data.get('collection_role')!='sweep':raise ValueError('Sweep collection required')
    groups=defaultdict(list)
    for row in data['rows']:
        groups[(row['rank'],row['family'],str(row['shape']),row['requested_mhz'])].append(row)
    rows=[]
    for (rank,family,shape,rate),items in sorted(groups.items()):
        rows.append(dict(rank=rank,family=family,shape=shape,requested_mhz=rate,windows=len(items),
            actual_mean_mhz=mean(r['frequency_hz']/1e6 for r in items),
            voltage_mean_v=mean(r['voltage_v'] for r in items),
            temperature_min_c=min(r['feature_ranges']['temperature_c'][0] for r in items),
            temperature_max_c=max(r['feature_ranges']['temperature_c'][1] for r in items),
            latency_mean_us=mean(r['latency_s']*1e6 for r in items),
            board_power_mean_w=mean(r['energy_j']/r['host_window_s'] for r in items),
            max_energy_budget=max(r['energy_relative_budget'] for r in items),
            memory_mhz=sorted({r['memory_mhz'] for r in items})))
    expected={(rank,j['case']['family'],j['case']['size'],j['requested_mhz']) for rank in (0,1) for j in data['protocol']}
    # Shape 256 denotes M=1 GEMV; shape 512 is compute GEMM or copy.
    observed={(r['rank'],r['family'],256 if r['family']=='linear' and r['shape'][0]==1 else 512,r['requested_mhz'])
              for r in data['rows']}
    memory={r['memory_mhz'] for r in data['rows']}
    return dict(status='exploratory_sweep_report',source_sha256=digest(data),collection_status=data['status'],
        error=data.get('error'),restoration=data.get('restoration'),rows=rows,
        missing_rank_resource_settings=sorted(expected-observed),
        memory_frequency_constant=len(memory)==1,deployable=False,parameters_fitted=False,
        scope='repeated_synthetic_resource_timing_and_board_window_power_not_request_latency',
        limitations=['working_set_may_be_cache_resident_not_DRAM_bandwidth',
                     'timing_and_power_frequency_not_simultaneously_observed',
                     'no_per_operator_measured_energy_or_5_percent_request_decision'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();result=report(read(a.input));out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    write_new(out/'report.json',result)
    if result['rows']:
        with (out/'curves.csv').open('x',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(result['rows'][0]));writer.writeheader();writer.writerows(result['rows'])
    print('collection:',result['collection_status'],'summary rows:',len(result['rows']),
        'missing:',len(result['missing_rank_resource_settings']))
    print('restoration:',result['restoration']);print('Saved:',out)

if __name__=='__main__':main()
