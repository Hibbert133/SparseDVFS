"""Offline dense-linear workload accounting; not calibrated Eq1 parameters."""
import argparse
from pathlib import Path
from .matrix_probe import candidates
from .service_core import read, write_new
from .service_client import sha
from .observe_core import GPU_UUIDS


def account(job, rank):
    m, n, k = job['m'], job['n'], job['k']
    return dict(rank=rank, uuid=GPU_UUIDS[rank], phase=job['phase'], module=job['module'],
        signature_id=job['row']['signature_id'], M=m, N=n, K=k,
        work_flop=2*m*n*k, work_convention='dense_FMA_counts_as_two_FLOPs',
        input_bytes=2*m*k, weight_bytes=2*n*k, output_bytes=2*m*n,
        logical_data_bytes=2*(m*k+n*k+m*n),
        arithmetic_intensity_flop_per_logical_byte=(m*n*k)/(m*k+n*k+m*n),
        data_semantics='one_logical_read_of_X_W_and_write_of_Y_not_measured_DRAM_traffic',
        kernel_names=job['row']['signature']['kernel_names'],
        internal_workspace_and_reduction_bytes=None, cache_state_verified=False,
        effective_s_comp=0, effective_s_mem=0,
        sparsity_basis='no_verified_skipping_or_traffic_reduction_evidence',
        peak_flops=None, bandwidth_Bps=None, overhead_s=None,
        latency_prediction_s=None, calibration_ready=False)


def run(args):
    trace = Path(args.trace_dir).resolve()
    obs = read(trace/'observation.json')
    if obs['status'] != 'complete' or obs['coverage']['passed'] is not True:
        raise ValueError('Require completed dual-rank observation')
    result = dict(status='workload_accounting_only', clock_writes=False,
        identity_fingerprint=obs['identity_fingerprint'], observation_sha256=sha(trace/'observation.json'),
        scope='four_MLP_shapes_per_rank_not_complete_model_or_kernel_DAG',
        equation_1_parameters_valid=False, equation_2_parameters_valid=False,
        sources=[], workloads=[])
    for rank in (0, 1):
        files = list(trace.glob(f'rank-{rank}-matrix-signatures-*.json'))
        if not files: raise ValueError(f'Missing rank {rank} matrix signatures')
        source = max(files, key=lambda p:p.stat().st_mtime_ns)
        result['sources'].append(dict(rank=rank, path=str(source), sha256=sha(source)))
        result['workloads'].extend(account(j, rank) for j in candidates(read(source), rank))
    for phase in ('prefill', 'decode'):
        for module in ('gate_up_proj', 'down_proj'):
            pair = [r for r in result['workloads'] if r['phase']==phase and r['module']==module]
            for key in ('M', 'N', 'K', 'work_flop', 'logical_data_bytes'):
                if pair[0][key] != pair[1][key]: raise ValueError('Rank workload shape mismatch')
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    write_new(out/'workloads.json', result)
    lines = ['Offline workload accounting; logical bytes are NOT measured DRAM traffic.',
             'rank | phase | module | M,N,K | GFLOP | logical MiB | FLOP/logical byte']
    for r in result['workloads']:
        lines.append(f"{r['rank']} | {r['phase']} | {r['module']} | {r['M']},{r['N']},{r['K']} | "
                     f"{r['work_flop']/1e9:.6f} | {r['logical_data_bytes']/2**20:.6f} | "
                     f"{r['arithmetic_intensity_flop_per_logical_byte']:.6f}")
    lines.extend(['Not calibrated: compute capacity, bandwidth, overhead, cache state and internal traffic.',
                  'Do not fit profiler durations or divide board energy among these operators.'])
    (out/'REPORT.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines))
    print('Saved:', out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace-dir', required=True)
    p.add_argument('--out', required=True)
    run(p.parse_args())
