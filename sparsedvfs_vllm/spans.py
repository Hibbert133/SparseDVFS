"""Hooks observe vLLM's execution; they never replay layers or touch cache state."""
import time
from .contracts import UNITS


def discover(model):
    causal = getattr(model, 'language_model', model)
    text = getattr(causal, 'model', None)
    if text is None or type(text).__name__ != 'Qwen3_5Model':
        raise RuntimeError(f'Expected the v0.19 Qwen3_5Model; got {type(text).__name__}')
    layers = list(text.layers)
    if len(layers) != 64 or text.start_layer != 0 or text.end_layer != 64:
        raise RuntimeError('All 64 layers must reside on one worker')
    types = [layer.layer_type for layer in layers]
    if types != ['linear_attention', 'linear_attention', 'linear_attention', 'full_attention'] * 16:
        raise RuntimeError('Unexpected Qwen3.5-27B hybrid layer order')
    if not callable(getattr(model, 'compute_logits', None)):
        raise RuntimeError('Top-level compute_logits is absent')
    return [('embed', text.embed_tokens)] + [(f'layer.{i}', layer) for i, layer in enumerate(layers)] + [('norm', text.norm)], types


class Spans:
    def __init__(self, model, torch, schedule_getter):
        self.torch, self.schedule_getter = torch, schedule_getter
        self.active, self.order, self.pending = None, [], {}
        self.events, self.rows, self.steps, self.handles = [], [], [], []
        self.main_stream = None
        modules, self.layer_types = discover(model)
        for name, module in modules:
            def pre(mod, args, name=name):
                self.begin(name)
            def post(mod, args, result, name=name):
                self.end(name)
            self.handles.append(module.register_forward_pre_hook(pre))
            self.handles.append(module.register_forward_hook(post))
        self.model, self.original_logits = model, model.compute_logits
        def compute_logits(*args, **kwargs):
            self.begin('logits')
            try:
                return self.original_logits(*args, **kwargs)
            finally:
                self.end('logits')
        model.compute_logits = compute_logits

    def start(self, context, run_id):
        if self.active is not None:
            raise RuntimeError('execute_model/sample_tokens scope still open')
        self.active = dict(context, run_id=run_id)
        self.order, self.pending = [], {}
        schedule = self.schedule_getter()
        if schedule:
            schedule.start_forward(context['phase'], context['step'])
        self.reap()

    def begin(self, name):
        if self.active is None:
            return
        index = len(self.order)
        if index >= len(UNITS) or name != UNITS[index] or self.pending:
            raise RuntimeError(f'Unexpected execution order: index={index}, unit={name}, open={list(self.pending)}')
        if self.torch.cuda.is_current_stream_capturing():
            raise RuntimeError('CUDA Graph capture is unsupported')
        stream = self.torch.cuda.current_stream().cuda_stream
        if self.main_stream is not None and self.main_stream != stream:
            raise RuntimeError('All logical units must use one ordered current CUDA main stream')
        self.main_stream = stream
        schedule = self.schedule_getter()
        if schedule:
            schedule.before(self.active['phase'], name)
        event = self.torch.cuda.Event(enable_timing=True)
        event.record(self.torch.cuda.current_stream())
        self.pending[name] = (event, time.monotonic_ns(), self.torch.cuda.current_stream().cuda_stream)
        self.order.append(name)

    def end(self, name):
        if self.active is None:
            return
        start, host_start, stream = self.pending.pop(name)
        if stream != self.torch.cuda.current_stream().cuda_stream:
            raise RuntimeError('A logical unit changed its current CUDA stream')
        end = self.torch.cuda.Event(enable_timing=True)
        end.record(self.torch.cuda.current_stream())
        self.events.append((dict(self.active, name=name, cpu_submit_ms=(time.monotonic_ns()-host_start)/1e6,
                                 host_start_ns=host_start, stream=int(stream)), start, end))
        schedule = self.schedule_getter()
        if schedule:
            schedule.after(self.active['phase'], name)

    def finish(self):
        if self.active is None or self.pending or self.order != UNITS:
            raise RuntimeError(f'Incomplete step coverage: {self.order}')
        self.steps.append(dict(self.active, order=list(self.order), coverage_complete=True))
        self.active = None
        self.reap()

    def reap(self, wait=False):
        if wait:
            self.torch.cuda.synchronize()
        pending = []
        for row, start, end in self.events:
            if end.query():
                self.rows.append(dict(row, latency_ms=start.elapsed_time(end), s_comp=0.0, s_mem=0.0,
                                      sparsity_exact_zero=None))
            else:
                pending.append((row, start, end))
        self.events = pending

    def export(self, reset=False):
        if self.active:
            raise RuntimeError('Cannot export between execute_model and sample_tokens')
        self.reap(wait=True)
        data = dict(rows=list(self.rows), steps=list(self.steps), units=UNITS,
                    layer_types=self.layer_types, timing_scope='events_on_current_vllm_main_stream')
        if reset:
            self.rows.clear()
            self.steps.clear()
        return data

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.model.compute_logits = self.original_logits
