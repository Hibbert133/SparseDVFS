"""Scoped dispatcher observation. No tensor values, copies or layer replay."""
import time
from contextlib import contextmanager
from .spans import Spans
from .operator_contracts import context_key, layout_entry, marker_name, validate_live


def tensor_metadata(value, torch):
    if isinstance(value, torch.Tensor):
        return dict(tensor=True, shape=list(value.shape), dtype=str(value.dtype),
                    device=str(value.device), stride=list(value.stride()), layout=str(value.layout))
    if isinstance(value, (list, tuple)):
        return [tensor_metadata(v, torch) for v in value]
    if isinstance(value, dict):
        return {str(k): tensor_metadata(v, torch) for k, v in sorted(value.items(), key=lambda p: str(p[0]))}
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return {'type': type(value).__name__}


class OperatorSpans(Spans):
    """Module hooks label dispatcher calls; only dispatcher GPU units schedule.

    Full profiling is enabled only for diagnostic certification. Ordinary
    instrumented/profile/bench modes use events, not torch.profiler.
    """
    def __init__(self, model, torch, schedule_getter):
        self.operator_schedule_getter = schedule_getter
        self.operator_rows, self.operator_steps, self.operator_events = [], [], []
        self.parent, self.calls, self.certificate = '__worker__', [], None
        self.diagnostic, self.expected, self.unit_set = False, None, set()
        # Module hooks remain coverage evidence and must never double-schedule.
        super().__init__(model, torch, lambda: None)

    def configure(self, certificate=None, diagnostic=False):
        self.certificate, self.diagnostic = certificate, diagnostic

    def begin(self, name):
        super().begin(name)
        if self.active is not None:
            self.parent = name

    def end(self, name):
        super().end(name)
        self.parent = '__worker__'

    def start(self, context, run_id):
        super().start(context, run_id)
        current = self.torch.cuda.current_stream().cuda_stream
        if self.main_stream is not None and self.main_stream != current:
            raise RuntimeError('Worker changed the certified main CUDA stream')
        self.main_stream = current
        self.calls, self.parent = [], '__worker__'
        self.case = context_key(context)
        if self.certificate is not None:
            if self.case not in self.certificate['cases']:
                raise RuntimeError('Uncertified scheduler context; retrace before applying DVFS')
            case = self.certificate['cases'][self.case]
            self.expected, self.unit_set = case['layout'], set(case['units'])
        else:
            self.expected, self.unit_set = None, set()
        schedule = self.operator_schedule_getter()
        if schedule:
            schedule.start_forward(self.case, context['step'])

    @contextmanager
    def dispatch_scope(self):
        if self.active is None:
            yield
            return
        # Internal API availability/version is checked on the actual pinned image.
        from torch.utils._python_dispatch import TorchDispatchMode
        observer = self
        class Observe(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                return observer.dispatch(func, args, kwargs or {})
        with Observe():
            yield

    def dispatch(self, func, args, kwargs):
        torch = self.torch
        ordinal = len(self.calls)
        entry = layout_entry(self.parent, ordinal, str(func),
                             tensor_metadata({'args': args, 'kwargs': kwargs}, torch))
        if self.expected is not None:
            validate_live(self.expected, ordinal, entry)  # Before calling/scheduling.
        marker = marker_name(self.active['request_id'], self.active['step'], ordinal)
        stream = torch.cuda.current_stream()
        if self.main_stream != stream.cuda_stream or torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Dispatcher crossed/captured the certified CUDA main stream')
        schedule = self.operator_schedule_getter()
        scheduled = schedule is not None and entry['name'] in self.unit_set
        if scheduled:
            schedule.before(self.case, entry['name'])
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(stream)
        host_start = time.monotonic_ns()
        record = torch.profiler.record_function(marker) if self.diagnostic else None
        if record:
            record.__enter__()
        try:
            result = func(*args, **kwargs)  # Exactly one original dispatch call.
        finally:
            if record:
                record.__exit__(*__import__('sys').exc_info())
        submitted = time.monotonic_ns()
        end.record(stream)
        if torch.cuda.current_stream().cuda_stream != stream.cuda_stream:
            raise RuntimeError('Dispatcher changed its CUDA stream')
        if scheduled:
            schedule.after(self.case, entry['name'])
        self.calls.append(dict(marker=marker, layout=entry))
        row = dict(self.active, name=entry['name'], phase=self.case,
                   model_phase=self.active['phase'], parent=self.parent, ordinal=ordinal,
                   overload=entry['overload'], opaque_fused=entry['opaque_fused'],
                   metadata=entry['metadata'], marker=marker, stream=int(stream.cuda_stream),
                   host_start_ns=host_start, host_return_ns=submitted,
                   cpu_submit_ms=(submitted-host_start)/1e6)
        self.operator_events.append((row, start, end))
        return result

    def finish(self):
        if self.expected is not None and len(self.calls) != len(self.expected):
            raise RuntimeError('Incomplete dispatcher order; stop and retrace')
        context = dict(self.active, calls=list(self.calls), context_key=self.case)
        super().finish()  # Also verifies all 68 module/logits/sample scopes.
        self.operator_steps.append(context)

    def reap(self, wait=False):
        super().reap(wait)
        pending = []
        for row, start, end in self.operator_events:
            if end.query():
                self.operator_rows.append(dict(row, latency_ms=start.elapsed_time(end),
                                               s_comp=0.0, s_mem=0.0))
            else:
                pending.append((row, start, end))
        self.operator_events = pending

    def export(self, reset=False):
        data = super().export(reset=False)
        data.update(module_rows=data['rows'], rows=list(self.operator_rows),
                    operator_steps=list(self.operator_steps), granularity='operator',
                    timing_scope='dispatcher scopes on the current verified main stream')
        if self.certificate is not None:
            data['units'] = {key: case['units'] for key, case in self.certificate['cases'].items()}
            data['coverage_sha256'] = self.certificate['certificate_sha256']
        if reset:
            self.reset()
        return data

    def reset(self):
        self.rows.clear()
        self.steps.clear()
        self.operator_rows.clear()
        self.operator_steps.clear()
