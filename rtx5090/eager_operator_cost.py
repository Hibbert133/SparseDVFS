"""One-pass resident operator event envelopes. No profiler or stateful replay."""
import time
import torch
from .observe_worker import Observer
from .eager_core import context_key
from sparsedvfs_vllm.operators import tensor_metadata


class OperatorCost(Observer):
    def __init__(self, model, rank, context):
        super().__init__(model, rank, context, 60000)
        self.ordinals = {}
        self.contexts = ('prefill', 'decode.0', 'decode.31', 'decode.62')

    def dispatch(self, func, args, kwargs):
        if self.busy:
            return func(*args, **kwargs)
        context = self.context()
        if context is None:
            return func(*args, **kwargs)
        key = context_key(context)
        # Representative early/middle/last positions; not full decode coverage.
        if key not in self.contexts:
            return func(*args, **kwargs)
        self.busy = True
        try:
            if len(self.rows) >= self.limit:
                raise RuntimeError('Operator cost event limit exceeded')
            ordinal = self.ordinals.get(key, 0)
            self.ordinals[key] = ordinal + 1
            stream = torch.cuda.current_stream()
            row = dict(rank=self.rank, context=key, ordinal=ordinal,
                       parent=self.parents[-1] if self.parents else '__worker__',
                       overload=str(func), scheduler=dict(context),
                       inputs=tensor_metadata({'args': args, 'kwargs': kwargs}, torch),
                       stream=int(stream.cuda_stream),
                       kernel_mapping_verified=False, energy_measured=False)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            self.rows.append(row)
            start.record(stream)
            begin = time.perf_counter_ns()
            output = func(*args, **kwargs)  # Exactly once, including attention and collective.
            returned = time.perf_counter_ns()
            end.record(stream)
            row.update(host_dispatch_s=(returned-begin)/1e9,
                       outputs=tensor_metadata(output, torch),
                       stream_changed=int(torch.cuda.current_stream().cuda_stream) != int(stream.cuda_stream))
            self.events.append((row, start, end))
            return output
        finally:
            self.busy = False
