"""Independent primitive microbenchmarks; conditional Eq1, NOT same-path certification.

Keeps the eager service resident. Each selected GPU uses a short-lived CUDA
process only after a conservative free-memory check. No clocks are written.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from .observe_core import GPU_UUIDS, digest
from .service_core import read, write_new, positive, errors
from .performance import safe_error
from sparsedvfs.paper_core import timeline_seconds


def estimate(rows):
    """Uses ONLY named calibration primitives; never target replay labels."""
    by_name = {r['name']: r for r in rows}
    h = statistics.median(by_name['launch_proxy']['seconds'])
    positive(h, 'launch proxy')
    result = dict(overhead_s=h)
    for name, field, amount in [('gemm_train', 'peak_flops', 'work_flop'),
                                ('copy_train', 'bandwidth_Bps', 'logical_data_bytes')]:
        row = by_name[name]
        t = statistics.median(row['seconds']) - h
        positive(t, name+' duration minus proxy overhead')
        result[field] = row[amount]/t
    return dict(result, estimator='median_independent_primitives_minus_tiny_copy_proxy',
        data_semantics='logical_tensor_bytes_not_measured_DRAM',
        same_path_verified=False, parameters_validated=False,
        cpu_emc_controlled=False, synthetic_tensor_values=True)


def compare(parameters, rows):
    result = []
    for row in rows:
        actual = statistics.mean(row['seconds'])
        positive(actual, 'held-out observed latency')
        predicted = timeline_seconds(row['work_flop'], row['logical_data_bytes'], 0, 0,
            parameters['peak_flops'], parameters['bandwidth_Bps'], parameters['overhead_s'])
        result.append(dict(name=row['name'], observed_s=actual, predicted_s=predicted,
                           relative_error=(predicted-actual)/actual))
    return dict(rows=result, scores=errors([r['observed_s'] for r in result],
                                         [r['predicted_s'] for r in result]),
                parameter_tuning_from_these_labels=False)


def snapshot(uuid):
    begin = time.monotonic_ns()
    raw = subprocess.check_output(['nvidia-smi', '-i', uuid,
        '--query-gpu=uuid,clocks.sm,clocks.mem,temperature.gpu,memory.free',
        '--format=csv,noheader,nounits'], text=True, timeout=15).strip()
    cells = [v.strip() for v in raw.split(',')]
    if len(cells) != 5 or cells[0] != uuid: raise ValueError('GPU snapshot identity mismatch')
    return dict(uuid=uuid, sm_mhz=int(cells[1]), memory_mhz=int(cells[2]),
        temperature_c=int(cells[3]), free_mib=int(cells[4]),
        query_start_ns=begin, query_end_ns=time.monotonic_ns(),
        during_batch_frequency_verified=False)


def worker(args):
    out = Path(args.out)
    uuid = GPU_UUIDS[args.rank]
    result = dict(status='failed', rank=args.rank, uuid=uuid, rows=[], clock_writes=False)
    try:
        before = snapshot(uuid)
        result['before_context'] = before
        if before['free_mib'] < 1536:
            result['status'] = 'blocked_insufficient_free_memory'
            return result
        os.environ['CUDA_VISIBLE_DEVICES'] = uuid
        import torch
        from .native import assert_cuda_uuid
        assert_cuda_uuid(torch, uuid)
        identity = read(args.identity)
        reference = identity['ranks'][args.rank]
        if str(torch.__version__) != reference['torch_version'] or torch.version.cuda != reference['torch_cuda']:
            raise ValueError('Worker and benchmark torch/CUDA versions differ')
        result['torch'], result['cuda'] = str(torch.__version__), torch.version.cuda
        result['after_context'] = snapshot(uuid)
        result['cuda_free_mib'] = torch.cuda.mem_get_info()[0]//2**20
        if min(result['after_context']['free_mib'], result['cuda_free_mib']) < 1024:
            result['status'] = 'blocked_insufficient_free_memory_after_context'
            return result
        torch.manual_seed(11062001)
        def measure(name, op, work, data):
            row = dict(name=name, work_flop=work, logical_data_bytes=data, seconds=[],
                host_seconds=[], repeats=32, before=snapshot(uuid))
            result['rows'].append(row)
            for _ in range(5): op()
            torch.cuda.synchronize()
            for _ in range(3):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin = time.monotonic_ns()
                start.record()
                for _ in range(32): op()
                end.record(); end.synchronize()
                row['seconds'].append(start.elapsed_time(end)/1000/32)
                row['host_seconds'].append((time.monotonic_ns()-begin)/1e9/32)
            row['after'] = snapshot(uuid)
            return row
        with torch.inference_mode():
            a = torch.ones(1, device='cuda', dtype=torch.bfloat16)
            b = torch.empty_like(a)
            measure('launch_proxy', lambda:b.copy_(a), 0, 4)
            del a,b
            def gemm(name, n):
                a = torch.randn((n,n), device='cuda', dtype=torch.bfloat16)
                b = torch.randn_like(a); c = torch.empty_like(a)
                measure(name, lambda:torch.mm(a,b,out=c), 2*n**3, 6*n*n)
            def copy(name, mib):
                count = mib*2**20//2
                a = torch.ones(count, device='cuda', dtype=torch.bfloat16)
                b = torch.empty_like(a)
                measure(name, lambda:b.copy_(a), 0, 4*count)
            gemm('gemm_train', 3072)
            copy('copy_train', 64)
            result['parameters'] = estimate(result['rows'])
            # Freeze before evaluating held-out primitive shapes or reading MLP timing labels.
            write_new(out/'parameters.json', dict(result['parameters'],
                identity_fingerprint=identity['fingerprint'], rank=args.rank, uuid=uuid,
                training_sha256=digest(result['rows'])))
            gemm('gemm_holdout', 2048)
            copy('copy_holdout', 96)
        result['primitive_holdout'] = compare(result['parameters'],
            [r for r in result['rows'] if r['name'].endswith('holdout')])
        report = read(Path(args.replay)/('rank-%d-matrix-replay.json'%args.rank))
        if report['uuid'] != uuid or report['identity_sha256'] != digest(reference):
            raise ValueError('Replay worker identity mismatch')
        result['MLP_transfer_diagnostic'] = compare(result['parameters'], [dict(
            name=r['phase']+'/'+r['module'], work_flop=r['work_flop'],
            logical_data_bytes=r['logical_data_bytes'],
            seconds=[b['mean_call_s'] for b in r['batches']]) for r in report['rows']])
        result.update(status='conditional_primitive_model_collected',
            MLP_same_path_verified=False, MLP_frequency_cache_matched=False,
            equation_1_parameters_validated=False, equation_2_parameters_validated=False,
            block_selection_authorized=False)
        return result
    except BaseException as exc:
        result['error'] = safe_error(exc)
        raise
    finally:
        write_new(out/'measurement.json', result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ('identity','replay','out'): p.add_argument('--'+flag, required=True)
    p.add_argument('--rank', type=int, choices=[0,1])
    p.add_argument('--worker', action='store_true')
    args = p.parse_args()
    if args.worker:
        if args.rank is None: p.error('--worker requires --rank')
        worker(args); return
    if os.name != 'posix': p.error('Run controller in target WSL')
    import fcntl
    from . import eager
    identity = read(args.identity)
    with (Path(identity['runtime'])/'client.lock').open('a') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = eager.checked(args.identity)
        replay = read(Path(args.replay)/'observation.json')
        if replay['status'] != 'complete' or replay['identity_fingerprint'] != identity['fingerprint']:
            raise ValueError('Replay is not from current service')
        out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=False)
        result = dict(status='collecting', clock_writes=False, ranks=[],
            identity_fingerprint=identity['fingerprint'], commit=eager.git('rev-parse','HEAD'),
            scope='independent_primitive_microbenchmarks_not_same_kernel_MLP_calibration')
        try:
            for rank in ([args.rank] if args.rank is not None else [0,1]):
                folder = out/('rank-%d'%rank); folder.mkdir()
                with (folder/'worker.log').open('w') as log:
                    process = subprocess.run([sys.executable, '-B', '-m', __spec__.name,
                        '--worker', '--rank', str(rank), '--identity', str(Path(args.identity).resolve()),
                        '--replay', str(Path(args.replay).resolve()), '--out', str(folder)],
                        stdout=log, stderr=subprocess.STDOUT, timeout=180)
                row = read(folder/'measurement.json'); result['ranks'].append(row)
                print(json.dumps(dict(rank=rank,status=row['status'],
                    parameters=row.get('parameters'), primitive_holdout=row.get('primitive_holdout'),
                    MLP_transfer_diagnostic=row.get('MLP_transfer_diagnostic')), ensure_ascii=False, indent=2), flush=True)
                if process.returncode: raise RuntimeError('Benchmark failed; inspect '+str(folder/'worker.log'))
            eager.checked(args.identity)
            result['status'] = ('conditional_collection_complete' if all(
                r['status']=='conditional_primitive_model_collected' for r in result['ranks']) else 'partial_collection')
        except BaseException as exc:
            result.update(status='failed', error=safe_error(exc)); raise
        finally:
            write_new(out/'collection.json', result)
        print('Saved:', out)


if __name__ == '__main__': main()
