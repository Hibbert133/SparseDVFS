"""Read-only TP worker: no FUSE, clock backend, or legacy model contracts."""
from contextlib import contextmanager
import hashlib
import inspect
import os
from pathlib import Path
import time

import torch
import vllm
from vllm.v1.worker.gpu_worker import Worker

from sparsedvfs_vllm.operators import tensor_metadata
from .native import check_vllm_sources
from .observe_core import (SCHEMA, MARKER, DEVELOPMENT, PhaseTracker, canonical_uuid,
                           correlate, read_json, write_json)
from .performance import safe_error


def file_identity(cls):
    source = inspect.getfile(cls)
    return dict(type=f'{cls.__module__}.{cls.__qualname__}', file=source,
                sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest())


class Observer:
    def __init__(self, model, rank, context, limit):
        self.rank, self.context, self.limit = rank, context, limit
        self.rows, self.events, self.hooks, self.parents = [], [], [], []
        self.busy = False
        for name, module in model.named_modules():
            label = name or '__model__'
            def enter(mod, args, label=label): self.parents.append(label)
            def leave(mod, args, output, label=label):
                if not self.parents or self.parents.pop() != label:
                    raise RuntimeError('Module hook nesting changed')
            self.hooks.append(module.register_forward_pre_hook(enter))
            self.hooks.append(module.register_forward_hook(leave, always_call=True))

    @contextmanager
    def scope(self):
        from torch.utils._python_dispatch import TorchDispatchMode
        observer = self
        class Observe(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                return observer.dispatch(func, args, kwargs or {})
        with Observe():
            yield

    def dispatch(self, func, args, kwargs):
        # The metadata and timing calls below must not recursively observe themselves.
        if self.busy:
            return func(*args, **kwargs)
        self.busy = True
        row = None
        try:
            if len(self.rows) >= self.limit:
                raise RuntimeError('Observation operator limit reached; partial evidence retained')
            context = dict(self.context())
            ordinal = len(self.rows)
            marker = f'{MARKER}{self.rank}|{ordinal}'
            stream = torch.cuda.current_stream()
            before = tensor_metadata({'args': args, 'kwargs': kwargs}, torch)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            row = dict(marker=marker, rank=self.rank, phase=context['phase'],
                context=context, parent=self.parents[-1] if self.parents else '__worker__',
                overload=str(func), opaque=not str(func).startswith('aten.'),
                inputs=before, outputs=None, stream=int(stream.cuda_stream),
                host_start_ns=time.monotonic_ns(), gpu_activities=[])
            self.rows.append(row)
            start.record(stream)
            with torch.profiler.record_function(marker):
                output = func(*args, **kwargs)
            end.record(stream)
            row.update(host_return_ns=time.monotonic_ns(),
                       outputs=tensor_metadata(output, torch),
                       stream_changed=(torch.cuda.current_stream().cuda_stream != stream.cuda_stream))
            self.events.append((row, start, end))
            return output
        except BaseException as exc:
            if row is not None:
                row['error'] = safe_error(exc)
            raise
        finally:
            self.busy = False

    def finish(self):
        # One boundary synchronization after generate; never one per operator.
        torch.cuda.synchronize()
        for row, start, end in self.events:
            row['stream_event_ms'] = start.elapsed_time(end)
        self.events.clear()

    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class ObservationWorker(Worker):
    def load_model(self, *, load_dummy_weights=False):
        super().load_model(load_dummy_weights=load_dummy_weights)
        self._ob_active = False
        self._ob_observer = self._ob_profiler = None
        self._ob_context = None
        self._ob_sources = check_vllm_sources()
        if self.use_v2_model_runner:
            raise RuntimeError('Observation is audited for vLLM 0.32 V1 only')

    def observation_identity(self):
        cfg = self.vllm_config
        prop = torch.cuda.get_device_properties(self.device)
        if getattr(prop, 'uuid', None) is None:
            raise RuntimeError('Worker CUDA UUID unavailable')
        import triton
        expected = os.environ.get('SPARSEDVFS_TRITON_PATH')
        if not expected or Path(triton.__file__).resolve() != Path(expected).resolve()/'triton/__init__.py':
            raise RuntimeError('Worker did not import the selected persistent Triton')
        if triton.__version__ != '3.6.0':
            raise RuntimeError('Expected the existing independent Triton 3.6.0')
        selected = {}
        for name, fields in {
            'model_config': ('model', 'dtype', 'quantization', 'max_model_len', 'enforce_eager'),
            'parallel_config': ('tensor_parallel_size', 'pipeline_parallel_size', 'data_parallel_size', 'disable_custom_all_reduce'),
            'scheduler_config': ('max_num_seqs', 'max_num_batched_tokens', 'enable_chunked_prefill', 'async_scheduling'),
            'cache_config': ('enable_prefix_caching', 'cache_dtype', 'gpu_memory_utilization'),
            'load_config': ('load_format',),
        }.items():
            item = getattr(cfg, name)
            selected[name] = {key: str(getattr(item, key)) if key == 'dtype' else getattr(item, key, None)
                              for key in fields}
        return dict(rank=int(self.rank), local_rank=int(self.local_rank), pid=os.getpid(),
            gpu_uuid=canonical_uuid(prop.uuid), gpu_name=prop.name,
            tp_size=cfg.parallel_config.tensor_parallel_size,
            dtype=str(cfg.model_config.dtype), quantization=cfg.model_config.quantization,
            enforce_eager=cfg.model_config.enforce_eager, load_format=cfg.load_config.load_format,
            v2_runner=self.use_v2_model_runner, worker=file_identity(type(self)),
            runner=file_identity(type(self.model_runner)),
            model=file_identity(type(self.model_runner.get_model())),
            vllm_version=vllm.__version__, torch_version=torch.__version__, torch_cuda=torch.version.cuda,
            triton=dict(version=triton.__version__, file=triton.__file__,
                        sha256=hashlib.sha256(Path(triton.__file__).read_bytes()).hexdigest()),
            vllm_sources_sha256=self._ob_sources, actual_config=selected,
            communication_env={key: os.environ.get(key) for key in
                ('NCCL_P2P_DISABLE', 'NCCL_SHM_DISABLE', 'NCCL_IB_DISABLE')},
            clock_writes_performed=False)

    def observation_start(self, config):
        if self._ob_active:
            raise RuntimeError('An observation is already active')
        if config['prompt_tokens'] not in DEVELOPMENT or config['split'] != 'development':
            raise ValueError('First delivery only collects development workloads; holdout stays sealed')
        self._ob_root = Path(config['directory']) / f'rank-{self.rank}'
        self._ob_root.mkdir(parents=True, exist_ok=False)
        self._ob_config = config
        self._ob_tracker, self._ob_steps = PhaseTracker(), []
        self._ob_context = dict(phase='outside_execution')
        self._ob_observer = Observer(self.model_runner.get_model(), int(self.rank),
                                    lambda: self._ob_context, config['max_operators'])
        self._ob_profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False, profile_memory=False, with_stack=False)
        # Shapes already come from dispatcher metadata; profiler record_shapes
        # retains tensor references and can change allocation behavior.
        try:
            self._ob_profiler.start()
            self._ob_active = True
        except BaseException:
            self._ob_observer.close()
            raise
        write_json(self._ob_root/'identity.json', self.observation_identity())
        return dict(rank=int(self.rank), started=True)

    def execute_model(self, scheduler_output):
        if not self._ob_active:
            return super().execute_model(scheduler_output)
        try:
            context = self._ob_tracker.step(scheduler_output)
            self._ob_context = context or dict(phase='scheduler_idle')
            if context:
                self._ob_steps.append(dict(context))
            with self._ob_observer.scope():
                return super().execute_model(scheduler_output)
        except BaseException as exc:
            self._ob_failure(exc)
            raise

    def sample_tokens(self, grammar_output):
        if not self._ob_active:
            return super().sample_tokens(grammar_output)
        try:
            self._ob_context = dict(self._ob_context, boundary='sample_tokens')
            with self._ob_observer.scope():
                return super().sample_tokens(grammar_output)
        except BaseException as exc:
            self._ob_failure(exc)
            raise

    def _ob_failure(self, exc):
        write_json(self._ob_root/'failure.json', dict(status='failed', error=safe_error(exc),
                   context=self._ob_context, steps=self._ob_steps,
                   operators=self._ob_observer.rows, clock_writes_performed=False))

    def observation_stop(self):
        if not self._ob_active:
            return dict(rank=int(self.rank), status='inactive')
        self._ob_active = False
        report = dict(schema=SCHEMA, status='failed', rank=int(self.rank),
            request_key=self._ob_config['request_key'], split='development',
            prompt_tokens=self._ob_config['prompt_tokens'],
            steps=self._ob_steps,
            operators=self._ob_observer.rows, timing='diagnostic_profiler_and_stream_events',
            clock_writes_performed=False, replay_validated=False)
        stopped = False
        try:
            report['identity'] = self.observation_identity()
            self._ob_observer.finish()
            self._ob_profiler.stop()
            stopped = True
            self._ob_profiler.export_chrome_trace(str(self._ob_root/'trace.json'))
            events = read_json(self._ob_root/'trace.json')['traceEvents']
            report['attribution'] = correlate(events, report['operators'])
            if not report['operators'] or not report['attribution']['gpu_activities']:
                raise RuntimeError('No dispatcher/GPU evidence; observation is not complete')
            report['status'] = 'complete'
        except BaseException as exc:
            report['error'] = safe_error(exc)
            raise
        finally:
            if not stopped:
                try:
                    self._ob_profiler.stop()
                except BaseException as exc:
                    report['profiler_stop_error'] = safe_error(exc)
            self._ob_observer.close()
            write_json(self._ob_root/'request.json', report)
            self._ob_profiler = self._ob_observer = None
        return dict(rank=int(self.rank), status=report['status'],
                    request_file=str(self._ob_root/'request.json'),
                    gpu_activities=report['attribution']['gpu_activities'])
