"""Actual vLLM 0.32 V1 BF16/TP2 worker. Commands never execute CUDA on RPC threads."""
import atexit
import os
from pathlib import Path
import threading
import time

import torch

from .observe_worker import ObservationWorker
from .observe_core import PhaseTracker, write_json, digest
from . import eager_core as core
from .eager_ipc import LocalServer, rpc
from .performance import safe_error
from .service_client import sha, process_record
from sparsedvfs_vllm.spans import discover


class EagerWorker(ObservationWorker):
    def load_model(self, *, load_dummy_weights=False):
        if load_dummy_weights: raise ValueError('Dummy weights forbidden in eager experiment')
        super().load_model(load_dummy_weights=False)
        self._eg_root = Path(os.environ['SPARSEDVFS_EAGER_RUN'])
        self._eg_runtime = Path(os.environ['SPARSEDVFS_EAGER_RUNTIME'])
        self._eg_session = os.environ['SPARSEDVFS_EAGER_SESSION']
        self._eg_lock = threading.RLock()
        self._eg_mode, self._eg_pending, self._eg_context = 'off', None, None
        self._eg_running, self._eg_poisoned = False, None
        self._eg_tracker = PhaseTracker()
        self._eg_rows, self._eg_events, self._eg_steps, self._eg_handles = [], [], [], []
        self._eg_seq = 0
        self._eg_replay = None
        self._eg_operator_cost = None
        self._eg_boundary = None
        model = self.model_runner.get_model()
        modules, self._eg_layer_types = discover(model)
        for name, module in modules:
            self._eg_handles.append(module.register_forward_pre_hook(
                lambda mod, args, name=name: self._eg_begin(name)))
            self._eg_handles.append(module.register_forward_hook(
                lambda mod, args, out, name=name: self._eg_end(name)))
        original = model.compute_logits
        def logits(*args, **kwargs):
            self._eg_begin('logits')
            try: return original(*args, **kwargs)
            finally: self._eg_end('logits')
        model.compute_logits = logits
        self._eg_identity = self.eager_identity()
        write_json(self._eg_root/f'rank-{self.rank}-identity.json', self._eg_identity)
        self._eg_server = LocalServer(self._eg_runtime/f'rank-{self.rank}.sock', self._eg_command)
        atexit.register(self._eg_server.close)

    def eager_identity(self):
        value = self.observation_identity()
        cfg = self.vllm_config
        compilation = cfg.compilation_config
        def disabled(field):
            option = getattr(compilation, field, None)
            return option is not None and (getattr(option, 'value', option) == 0 or getattr(option, 'name', '') == 'NONE')
        p, s, m = cfg.parallel_config, cfg.scheduler_config, cfg.model_config
        if p.pipeline_parallel_size != 1 or p.data_parallel_size != 1 or getattr(p, 'enable_dbo', False):
            raise ValueError('Only TP2/PP1/DP1 without DBO is audited')
        value.update(schema=core.SCHEMA, session=self._eg_session,
            num_hidden_layers=m.hf_text_config.num_hidden_layers,
            speculative_config=None if cfg.speculative_config is None else str(cfg.speculative_config),
            compilation_disabled=disabled('mode'), graphs_disabled=disabled('cudagraph_mode'),
            compilation_mode=str(compilation.mode), cudagraph_mode=str(compilation.cudagraph_mode),
            nccl_cumem_host_enable=os.environ.get('NCCL_CUMEM_HOST_ENABLE'),
            units=core.UNITS, layer_types=self._eg_layer_types,
            weights=core.weights(m.model), process=process_record(os.getpid()),
            tool_hashes={p.name: sha(p) for p in Path(__file__).parent.glob('eager_*.py')},
            runtime=str(self._eg_runtime), capabilities=core.CAPABILITIES,
            boundary_completion='device_synchronize_all_streams_serial_diagnostic',
            resolved_weight_loader='auto_actual_loader_class_not_observed')
        value['matrix_replay_version'] = 2
        value['operator_cost_version'] = 2
        value['operator_boundary_version'] = 2
        value['operator_boundary_sha256'] = sha(Path(__file__).parent/'operator_boundary_worker.py')
        value['independent_primitives_version'] = 1
        value['resource_calibration_version'] = 1
        value['resource_calibration_sha256'] = sha(Path(__file__).parent/'resource_worker.py')
        value['independent_primitives_hashes'] = {name: sha(Path(__file__).parent/name) for name in
            ('worker_primitives.py', 'stateless_calibration_adapter.py')}
        # The two-rank check is performed by the client after BOTH files exist.
        if not value['compilation_disabled'] or not value['graphs_disabled']:
            raise RuntimeError('Eager did not disable compile/graphs in actual worker config')
        return value

    def _eg_command(self, command):
        with self._eg_lock:
            if command.get('session') != self._eg_session: raise ValueError('Wrong session')
            if command['op'] == 'status':
                return dict(identity=self._eg_identity, mode=self._eg_mode,
                    pending=self._eg_pending is not None, running=self._eg_running,
                    poisoned=self._eg_poisoned)
            if self._eg_poisoned: raise RuntimeError(self._eg_poisoned)
            if self._eg_running or self._eg_context is not None:
                raise RuntimeError('Configure only between complete requests')
            if command['op'] != 'arm': raise ValueError('Unknown eager RPC operation')
            mode = command['mode']
            if mode not in ('off', 'observe', 'units', 'serial', 'matrix_replay', 'matrix_calibrate', 'operator_cost', 'independent_primitives', 'operator_boundary', 'resource_calibration'): raise ValueError('Unsupported mode')
            if mode == 'resource_calibration':
                from .resource_worker import specification
                specification(**command['resource_case'])
                if command.get('apply') is not True: raise ValueError('Calibration requires explicit apply')
            if self._eg_pending and mode != 'off': raise ValueError('A request is already armed')
            if mode == 'serial':
                core.validate_plan(command['plan'], command['fingerprint'])
                core.require_gate(command['gate'], command['fingerprint'], command['telemetry_hash'])
                if command.get('apply') is not True: raise ValueError('Serial requires explicit --apply')
            if mode == 'off':
                self._eg_mode = 'off'
                self._eg_pending = None
            else:
                directory = Path(command['out']).resolve()
                if not directory.is_absolute() or not directory.is_dir(): raise ValueError('Missing run output directory')
                self._eg_pending = command
            return dict(rank=int(self.rank), armed=mode)

    def _eg_activate(self):
        with self._eg_lock:
            if self._eg_poisoned: raise RuntimeError(self._eg_poisoned)
            command = self._eg_pending
            if command is None: return
            self._eg_pending = None
            self._eg_mode, self._eg_config = command['mode'], command
            self._eg_tracker = PhaseTracker()
            self._eg_rows, self._eg_events, self._eg_steps = [], [], []
            self._eg_seq = 0
            if self._eg_mode == 'operator_boundary':
                from .operator_boundary_worker import BoundaryObserver
                self._eg_boundary = BoundaryObserver(self.model_runner.get_model(), int(self.rank),
                    lambda: self._eg_context, command, self._eg_identity)
            if self._eg_mode == 'operator_cost':
                from .eager_operator_cost import OperatorCost
                self._eg_operator_cost = OperatorCost(self.model_runner.get_model(), int(self.rank),
                                                      lambda: self._eg_context)
            if self._eg_mode in ('matrix_replay', 'matrix_calibrate'):
                from .eager_matrix_replay import MatrixReplay
                self._eg_replay = MatrixReplay(self.model_runner.get_model(), int(self.rank),
                    lambda: self._eg_context,
                    kernel_out=command['out'] if self._eg_mode == 'matrix_calibrate' else None)
            if self._eg_mode == 'observe':
                self.observation_start(dict(prompt_tokens=512, split='development',
                    directory=command['out'], request_key=command['request_id'], max_operators=250000))

    def _eg_begin(self, name):
        if self._eg_poisoned: raise RuntimeError(self._eg_poisoned)
        context = self._eg_context
        if context is None: return
        index = len(self._eg_order)
        if index >= len(core.UNITS) or core.UNITS[index] != name or self._eg_open is not None:
            raise RuntimeError('Nested/repeated/reordered execution boundary')
        if torch.cuda.is_current_stream_capturing(): raise RuntimeError('Unexpected graph capture')
        if self._eg_mode == 'serial':
            blocks = self._eg_config['plan']['contexts'][self._eg_key]
            for i, block in enumerate(blocks):
                if block['units'][0] == name:
                    # Full-device completion deliberately includes NCCL/custom streams.
                    # No later model unit has been submitted by this rank yet.
                    torch.cuda.synchronize(self.device)
                    response = rpc(self._eg_config['coordinator'], dict(rank=int(self.rank),
                        uuid=self._eg_identity['gpu_uuid'], fingerprint=self._eg_config['fingerprint'],
                        seq=self._eg_seq, gpu_complete=True,
                        key=[context['request_id'], self._eg_key, i]), timeout=30)
                    self._eg_seq += 1
                    if response.get('readback_verified') is not True or response['target_mhz'] != block['mhz']:
                        raise RuntimeError('Coordinator did not authorize this block')
                    break
        self._eg_open = name
        self._eg_order.append(name)
        if self._eg_mode == 'units':
            self._eg_start = torch.cuda.Event(enable_timing=True)
            self._eg_start.record()
            self._eg_stream = int(torch.cuda.current_stream().cuda_stream)

    def _eg_end(self, name):
        if self._eg_context is None: return
        if self._eg_open != name: raise RuntimeError('Unbalanced unit boundary')
        if self._eg_mode == 'units':
            if int(torch.cuda.current_stream().cuda_stream) != self._eg_stream:
                raise RuntimeError('Current stream changed across module')
            end = torch.cuda.Event(enable_timing=True); end.record()
            self._eg_events.append((dict(context=self._eg_key, unit=name,
                scheduler=dict(self._eg_context), stream=self._eg_stream), self._eg_start, end))
        self._eg_open = None

    def execute_model(self, scheduler_output):
        self._eg_activate()
        if self._eg_mode == 'off': return super().execute_model(scheduler_output)
        try:
            context = self._eg_tracker.step(scheduler_output)
            if context is None: return super().execute_model(scheduler_output)
            with self._eg_lock:
                self._eg_running = True
                self._eg_context = context
            if context['step'] == 0:
                new = next(r for r in scheduler_output.scheduled_new_reqs if r.req_id == context['request_id'])
                if digest(list(new.prompt_token_ids)) != self._eg_config['prompt_hash']:
                    raise RuntimeError('An unrelated request arrived while experiment was armed')
            if self._eg_mode == 'independent_primitives' and context['step'] == 0:
                from .worker_primitives import collect
                collect(torch, int(self.rank), self._eg_identity, self._eg_config['out'])
            if self._eg_mode == 'resource_calibration' and context['step'] == 0:
                from .resource_worker import collect
                collect(torch, int(self.rank), self._eg_identity, self._eg_config)
            self._eg_key = core.context_key(context)
            self._eg_order, self._eg_open = [], None
            if self._eg_mode == 'operator_boundary':
                with self._eg_boundary.scope():
                    result = super().execute_model(scheduler_output)
            elif self._eg_mode == 'operator_cost':
                with self._eg_operator_cost.scope():
                    result = super().execute_model(scheduler_output)
            elif self._eg_mode in ('matrix_replay', 'matrix_calibrate'):
                with self._eg_replay.scope():
                    result = super().execute_model(scheduler_output)
            else:
                result = super().execute_model(scheduler_output)
            if result is not None: raise RuntimeError('V1 execute/sample boundary changed')
            return result
        except BaseException as exc:
            self._eg_failure(exc); raise

    def sample_tokens(self, grammar_output):
        if self._eg_poisoned: raise RuntimeError(self._eg_poisoned)
        if self._eg_context is None: return super().sample_tokens(grammar_output)
        try:
            if grammar_output is not None: raise RuntimeError('Structured generation is outside protocol')
            self._eg_begin('sample')
            result = super().sample_tokens(grammar_output)
            self._eg_end('sample')
            if self._eg_order != core.UNITS: raise RuntimeError('Incomplete model coverage')
            self._eg_steps.append(dict(context=self._eg_key, scheduler=self._eg_context,
                                       units=list(self._eg_order)))
            last = self._eg_key == 'decode.62'
            self._eg_context = None
            if last:
                torch.cuda.synchronize(self.device)
                for row, start, end in self._eg_events:
                    self._eg_rows.append(dict(row, stream_event_ms=start.elapsed_time(end)))
                self._eg_events.clear()
                if self._eg_mode == 'operator_boundary':
                    self._eg_boundary.finish()
                    write_json(Path(self._eg_config['out'])/f'rank-{self.rank}-boundary.json',
                        dict(status='complete',rank=int(self.rank),identity_sha256=digest(self._eg_identity),
                             contexts=self._eg_boundary.seen,clock_writes=False))
                    self._eg_boundary.close(); self._eg_boundary = None
                if self._eg_mode == 'observe': self.observation_stop()
                if self._eg_mode == 'operator_cost':
                    self._eg_operator_cost.finish()
                    write_json(Path(self._eg_config['out'])/f'rank-{self.rank}-operator-cost.json',
                        dict(status='operator_event_envelopes_collected', rank=int(self.rank),
                             uuid=self._eg_identity['gpu_uuid'], identity_sha256=digest(self._eg_identity),
                             rows=self._eg_operator_cost.rows, contexts=list(self._eg_operator_cost.contexts),
                             scope='execute_model_only_excludes_sample_tokens',
                             timing='instrumented_current_stream_event_envelope_includes_launch_gaps',
                             all_stream_completion_per_operator_verified=False,
                             equation_1_parameters_validated=False, energy_measured=False))
                    self._eg_operator_cost.close()
                    self._eg_operator_cost = None
                if self._eg_mode in ('matrix_replay', 'matrix_calibrate'):
                    from .eager_matrix_replay import validate_rows
                    validate_rows(self._eg_replay.rows)
                    write_json(Path(self._eg_config['out'])/f'rank-{self.rank}-matrix-replay.json',
                        dict(status='real_tensor_replay_timing_only', rank=int(self.rank),
                             uuid=self._eg_identity['gpu_uuid'], rows=self._eg_replay.rows,
                             identity_sha256=digest(self._eg_identity), clock_writes=False,
                             equation_1_parameters_validated=False, equation_2_parameters_validated=False))
                    self._eg_replay.close()
                    self._eg_replay = None
                write_json(Path(self._eg_config['out'])/f'rank-{self.rank}-execution.json',
                    dict(schema=core.SCHEMA, status='complete', rank=int(self.rank),
                        identity_sha256=digest(self._eg_identity), mode=self._eg_mode,
                        steps=self._eg_steps if self._eg_mode != 'serial' else None,
                        contexts=[s['context'] for s in self._eg_steps],
                        unit_order_verified=True, steps_sha256=digest(self._eg_steps), rows=self._eg_rows,
                        timing_scope='diagnostic_current_stream_not_isolated_kernel_or_board_energy',
                        clock_writes_by_worker=False))
                with self._eg_lock:
                    self._eg_running = False
                    self._eg_mode = 'off'  # One explicitly armed request, never bleed into warmup.
            return result
        except BaseException as exc:
            self._eg_failure(exc); raise

    def _eg_failure(self, exc):
        if getattr(self, '_eg_poisoned', None): return  # Preserve the first failure.
        if getattr(self, '_eg_boundary', None) is not None:
            self._eg_boundary.close(); self._eg_boundary = None
        if getattr(self, '_eg_operator_cost', None) is not None:
            self._eg_operator_cost.close()
            write_json(Path(self._eg_config['out'])/f'rank-{self.rank}-operator-cost-failed.json',
                       dict(status='failed', rows=self._eg_operator_cost.rows, error=safe_error(exc)))
            self._eg_operator_cost = None
        if getattr(self, '_eg_replay', None) is not None:
            self._eg_replay.close()
            write_json(Path(self._eg_config['out'])/f'rank-{self.rank}-matrix-replay-failed.json',
                       dict(status='failed', rows=self._eg_replay.rows, error=safe_error(exc)))
            self._eg_replay = None
        self._eg_poisoned = safe_error(exc)
        write_json(self._eg_root/f'rank-{self.rank}-failure.json', dict(error=self._eg_poisoned,
            context=self._eg_context, steps=self._eg_steps, mode=self._eg_mode,
            note='coordinator must restore both GPUs; no worker-side reset race'))
