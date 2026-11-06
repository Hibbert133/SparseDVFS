"""Worker-local real-tensor linear replay timing, NOT calibrated Eq1/Eq2.

One first-occurring MLP projection per phase/kind, three 32-call batches.
No weight copy, synthetic inputs, collectives, clock setter or profiler.
Repeated cache state differs from the original inference; report it explicitly.
"""
from contextlib import contextmanager
import math
import time
from pathlib import Path
from collections import Counter
import json
from statistics import mean


def kernel_sequence(trace):
    """Isolated call capture; reject ambiguous multiple device/stream timelines."""
    events = [e for e in trace['traceEvents'] if e.get('cat') == 'kernel' and e.get('ph') == 'X']
    if not events: raise ValueError('No CUDA kernels in isolated linear capture')
    lanes = {(e.get('args', {}).get('device'), e.get('args', {}).get('stream')) for e in events}
    if len(lanes) != 1 or any(None in lane for lane in lanes):
        raise ValueError('Multiple/unknown kernel streams; cannot attribute isolated capture')
    events.sort(key=lambda e:e['ts'])
    return [dict(name=e['name'], grid=e.get('args', {}).get('grid'),
                 block=e.get('args', {}).get('block')) for e in events]


def compare_sequences(source, replay):
    a, b = kernel_sequence(source), kernel_sequence(replay)
    return dict(source_sequence=a, replay_sequence=b,
        names_counts_order_match=[v['name'] for v in a] == [v['name'] for v in b],
        launch_geometry_match=all(v['grid'] is not None and v['block'] is not None for v in a+b) and a == b,
        source_kernel_counts=dict(Counter(v['name'] for v in a)),
        scope='isolated_instrumented_original_vs_same_tensor_replay_not_cache_equivalence')


def batch_scaling(batches):
    """Independent count holdout; diagnostic only, NOT physical Eq1 overhead."""
    for b in batches:
        if not math.isfinite(b['elapsed_s']) or b['elapsed_s'] <= 0:
            raise ValueError('Invalid batch scaling time')
    means = {n:mean(b['elapsed_s'] for b in batches if b['repeats']==n) for n in (1,8,32,128)}
    xs = (1,32,128); xbar = mean(xs); ybar = mean(means[n] for n in xs)
    slope = sum((n-xbar)*(means[n]-ybar) for n in xs)/sum((n-xbar)**2 for n in xs)
    intercept = ybar-slope*xbar
    prediction = intercept+8*slope
    return dict(training_counts=list(xs), holdout_count=8, slope_s_per_call=slope,
        batch_intercept_s=intercept, holdout_observed_s=means[8], holdout_predicted_s=prediction,
        holdout_relative_error=(prediction-means[8])/means[8],
        physical_overhead_identified=False, equation_1_capacity_identified=False,
        note='Repeated-input batching diagnostic, not independent request validation')


def case_spec(parent, phase, overload, inputs, output, rank):
    if overload != 'aten.linear.default' or phase not in ('prefill', 'decode'):
        return None
    module = parent.rsplit('.', 1)[-1]
    if not parent.endswith(('.mlp.gate_up_proj', '.mlp.down_proj')):
        return None
    m = 512 if phase == 'prefill' else 1
    n, k = (17408, 5120) if module == 'gate_up_proj' else (5120, 8704)
    if len(inputs) != 2:
        raise ValueError('Only bias-free linear replay is audited')
    for meta, shape in zip([*inputs, output], ([m, k], [n, k], [m, n])):
        if (meta.get('tensor') is not True or meta['shape'] != shape or
                meta['stride'] != [shape[1], 1] or meta['dtype'] != 'torch.bfloat16' or
                meta['device'] != 'cuda:'+str(rank) or meta['layout'] != 'torch.strided'):
            raise ValueError('Worker replay tensor contract changed')
    return dict(phase=phase, module=module, parent=parent, M=m, N=n, K=k,
                work_flop=2*m*n*k, logical_data_bytes=2*(m*k+n*k+m*n))


def validate_rows(rows):
    expected = {(p, m) for p in ('prefill', 'decode') for m in ('gate_up_proj', 'down_proj')}
    if len(rows) != 4 or {(r['phase'], r['module']) for r in rows} != expected:
        raise ValueError('Missing or duplicate real-tensor replay cases')
    for row in rows:
        counts = (1,8,32,128) if row.get('collection_protocol') == 'kernel_verified_batch_counts_v1' else (32,)
        if len(row['batches']) != 3*len(counts) or any(
                sum(b['repeats']==n for b in row['batches']) != 3 for n in counts):
            raise ValueError('Incomplete replay batches')
        for batch in row['batches']:
            value = batch['elapsed_s']
            if not math.isfinite(value) or value <= 0 or batch['repeats'] not in counts:
                raise ValueError('Invalid replay batch timing')


class MatrixReplay:
    def __init__(self, model, rank, context, kernel_out=None):
        self.rank, self.context = rank, context
        self.parents, self.handles, self.rows, self.seen = [], [], [], set()
        self.busy = False
        self.kernel_out = Path(kernel_out) if kernel_out else None
        try:
            for name, module in model.named_modules():
                if not name.endswith(('.mlp.gate_up_proj', '.mlp.down_proj')):
                    continue
                def enter(mod, args, name=name): self.parents.append(name)
                def leave(mod, args, out, name=name):
                    if not self.parents or self.parents.pop() != name:
                        raise RuntimeError('Replay module nesting changed')
                self.handles.append(module.register_forward_pre_hook(enter))
                self.handles.append(module.register_forward_hook(leave, always_call=True))
        except BaseException:
            self.close()
            raise

    def close(self):
        for handle in self.handles: handle.remove()
        self.handles.clear()

    @contextmanager
    def scope(self):
        from torch.utils._python_dispatch import TorchDispatchMode
        owner = self
        class Replay(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                return owner.dispatch(func, args, kwargs or {})
        with Replay(): yield

    def dispatch(self, func, args, kwargs):
        import torch
        from sparsedvfs_vllm.operators import tensor_metadata
        # Return exactly the original result; replay outputs are discarded.
        if self.busy or not self.parents or str(func) != 'aten.linear.default': return func(*args, **kwargs)
        context = self.context()
        key = (context['phase'], self.parents[-1].rsplit('.', 1)[-1])
        if key in self.seen: return func(*args, **kwargs)
        self.busy = True
        try:
            if kwargs: raise ValueError('Unexpected linear keyword arguments')
            def capture(label):
                # No earlier queued kernels may enter the isolated capture.
                torch.cuda.synchronize()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.CUDA],
                                            record_shapes=False, with_stack=False,
                                            profile_memory=False) as profiler:
                    answer = func(*args, **kwargs)
                    torch.cuda.synchronize()
                path = self.kernel_out/('rank-%d-%s-%s-%s.json'%(self.rank,key[0],key[1],label))
                if path.exists(): raise FileExistsError(str(path))
                profiler.export_chrome_trace(str(path))
                return answer, json.loads(path.read_text(encoding='utf-8'))
            if self.kernel_out:
                result, original_trace = capture('original')
                extra, replay_trace = capture('replay')
                del extra
                match = compare_sequences(original_trace, replay_trace)
                del original_trace, replay_trace
            else:
                result = func(*args, **kwargs)
            spec = case_spec(self.parents[-1], context['phase'], str(func),
                [tensor_metadata(x, torch) for x in args], tensor_metadata(result, torch), self.rank)
            if spec is None: return result
            if torch.cuda.is_current_stream_capturing(): raise RuntimeError('Replay during CUDA capture')
            stream = torch.cuda.current_stream()
            row = dict(spec, scheduler=dict(context), rank=self.rank,
                input_metadata=[tensor_metadata(x, torch) for x in args],
                output_metadata=tensor_metadata(result, torch), batches=[],
                weight_storage_reused=True, activation_storage_reused=True,
                cache_state_matched=False, kernel_names_verified=False,
                timing_scope='repeated_real_tensor_current_stream_batches_including_launch_gaps',
                calibration_ready=False, independent_validation=False)
            self.rows.append(row)
            if self.kernel_out:
                row.update(collection_protocol='kernel_verified_batch_counts_v1',
                    pytorch_operator=str(func), kernel_comparison=match,
                    kernel_names_verified=match['names_counts_order_match'],
                    original_call_was_profiled=True, timing_profiler_enabled=False)
            counts = (1,8,32,128) if self.kernel_out else (32,)
            # Interleave batch sizes; preserve every batch, no retuning/retries.
            for repeats in [n for _ in range(3) for n in counts]:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                stream.synchronize()
                begin = time.monotonic_ns()
                start.record(stream)
                for _ in range(repeats):
                    replay_output = func(*args, **kwargs)
                    del replay_output
                end.record(stream)
                end.synchronize()
                elapsed = start.elapsed_time(end)/1000
                row['batches'].append(dict(repeats=repeats, elapsed_s=elapsed,
                    mean_call_s=elapsed/repeats, host_begin_ns=begin, host_end_ns=time.monotonic_ns()))
                if torch.cuda.current_stream() != stream:
                    raise RuntimeError('Replay current stream changed')
            if self.kernel_out: row['batch_scaling'] = batch_scaling(row['batches'])
            self.seen.add(key)
        finally:
            self.busy = False
        return result
