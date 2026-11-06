"""Worker-local real-tensor linear replay timing, NOT calibrated Eq1/Eq2.

One first-occurring MLP projection per phase/kind, three 32-call batches.
No weight copy, synthetic inputs, collectives, clock setter or profiler.
Repeated cache state differs from the original inference; report it explicitly.
"""
from contextlib import contextmanager
import math
import time


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
        if len(row['batches']) != 3:
            raise ValueError('Incomplete replay batches')
        for batch in row['batches']:
            value = batch['elapsed_s']
            if not math.isfinite(value) or value <= 0 or batch['repeats'] != 32:
                raise ValueError('Invalid replay batch timing')


class MatrixReplay:
    def __init__(self, model, rank, context):
        self.rank, self.context = rank, context
        self.parents, self.handles, self.rows, self.seen = [], [], [], set()
        self.busy = False
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
        result = func(*args, **kwargs)
        if self.busy or not self.parents or str(func) != 'aten.linear.default': return result
        context = self.context()
        key = (context['phase'], self.parents[-1].rsplit('.', 1)[-1])
        if key in self.seen: return result
        self.busy = True
        try:
            if kwargs: raise ValueError('Unexpected linear keyword arguments')
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
            for _ in range(3):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                stream.synchronize()
                begin = time.monotonic_ns()
                start.record(stream)
                for _ in range(32):
                    replay_output = func(*args, **kwargs)
                    del replay_output
                end.record(stream)
                end.synchronize()
                elapsed = start.elapsed_time(end)/1000
                row['batches'].append(dict(repeats=32, elapsed_s=elapsed,
                    mean_call_s=elapsed/32, host_begin_ns=begin, host_end_ns=time.monotonic_ns()))
                if torch.cuda.current_stream() != stream:
                    raise RuntimeError('Replay current stream changed')
            self.seen.add(key)
        finally:
            self.busy = False
        return result
