"""Loaded by --worker-cls inside the actual vLLM GPU worker process."""
import atexit
import json
import os
import signal
from contextlib import nullcontext
from pathlib import Path
import torch
import vllm
from vllm.v1.worker.gpu_worker import Worker
from sparsedvfs.fuse import FUSE
from sparsedvfs.hardware import load_json, write_json, frequencies, snapshot, wait_frequencies, observation
from sparsedvfs.qwen_runner import Schedule
from sparsedvfs.runtime_policy import RuntimePlanner
from .contracts import PROTOCOL, REVISION, UNITS, MODES, PhaseTracker, validate_engine, validate_plan, verify_model, digest_json
from .spans import Spans
from .state import restored


class SparseDVFSWorker(Worker):
    def load_model(self, *, load_dummy_weights=False):
        try:
            super().load_model(load_dummy_weights=load_dummy_weights)
        except BaseException as exc:
            self._sd_failure(exc, 'load_model')
            raise
        validate_engine(self.vllm_config, vllm.__version__, self.use_v2_model_runner)
        self._sd_schedule, self._sd_fuse, self._sd_cfg = None, None, {}
        self._sd_granularity = 'module'
        self._sd_tracker, self._sd_request, self._sd_enabled = PhaseTracker(), None, False
        self._sd_trace = Spans(self.model_runner.get_model(), torch, lambda: self._sd_schedule)
        self._sd_journal_rows = self._sd_journal_steps = 0
        self._sd_journal_ops = 0
        self._sd_close_requested = False
        self._sd_path = os.environ.get('SPARSEDVFS_WORKER_CONFIG')
        if self._sd_path:
            self.sdvfs_configure(load_json(self._sd_path))
        atexit.register(self._sd_shutdown)
        # Preserve vLLM's handler. A SIGKILL or host power loss needs explicit restore.
        old = signal.getsignal(signal.SIGTERM)
        def terminate(signum, frame):
            self._sd_shutdown()
            if callable(old):
                old(signum, frame)
            else:
                raise SystemExit(128 + signum)
        try:
            signal.signal(signal.SIGTERM, terminate)
            signal.signal(signal.SIGUSR2, self._sd_request_close)
        except ValueError:
            pass

    def _sd_identity(self, config):
        hw = load_json(config['hardware']) if config.get('hardware') else {}
        model_lock = verify_model('/model')['manifest_sha256']
        return dict(backend=PROTOCOL, model='Qwen/Qwen3.5-27B', revision=REVISION,
                    model_manifest_sha256=model_lock,
                    board=dict(model=hw.get('device_model'), bsp=hw.get('jetson_release')),
                    power_definition={k: v for k, v in (hw.get('power') or {}).items()
                                      if k not in ('path', 'voltage_path', 'current_path')},
                    frequency_constraints=dict(
                        cpu=[{k: d.get(k) for k in ('original_min', 'original_max', 'governor')}
                             for d in hw.get('cpu', [])],
                        cpu_legal_khz=hw.get('cpu_common_khz'),
                        gpu={k: hw.get('gpu', {}).get(k) for k in ('original_min', 'original_max', 'governor', 'frequencies')},
                        emc={k: hw.get('emc', {}).get(k) for k in ('backend', 'min_rate', 'max_rate')}),
                    image_digest=os.environ.get('SPARSEDVFS_IMAGE_DIGEST'),
                    vllm=vllm.__version__, torch=torch.__version__, cuda=torch.version.cuda,
                    device=torch.cuda.get_device_name(), dtype='bfloat16',
                    units=UNITS if config.get('granularity', 'module') == 'module' else 'certified_dispatcher_layout',
                    granularity=config.get('granularity', 'module'), workload=config['workload'],
                    cache=dict(prefix=False, chunked_prefill=False, async_scheduling=False,
                               kv_and_mamba_state='vllm_managed'),
                    engine=dict(eager=True, tp=1, pp=1, dp=1, max_num_seqs=1,
                                max_model_len=4096, gpu_memory_utilization=0.70))

    def sdvfs_configure(self, config):
        """A new RPC method; existing Worker methods are overridden by subclassing."""
        if self._sd_trace.active:
            raise RuntimeError('Configure only when the worker is idle')
        mode = config.get('mode', 'instrumented')
        if mode not in MODES:
            raise ValueError('Unknown experiment mode')
        if not isinstance(config.get('apply', False), bool):
            raise ValueError('apply must be a JSON boolean')
        if config.get('apply') and os.environ.get('SPARSEDVFS_APPLY') != '1':
            raise ValueError('Actual DVFS requires the launcher explicit --apply flag')
        if config.get('apply') and mode == 'instrumented':
            raise ValueError('Instrumented control must not apply DVFS')
        granularity = config.get('granularity', 'module')
        if granularity not in ('module', 'operator'):
            raise ValueError('Unknown granularity')
        identity = self._sd_identity(config)
        plan = load_json(config['plan']) if mode in ('serial', 'lookahead') else None
        certificate = None
        if granularity == 'operator':
            from .operator_contracts import validate_certificate
            if config.get('coverage'):
                certificate = load_json(config['coverage'])
            elif plan:
                certificate = plan.get('coverage_certificate')
            if certificate is not None:
                validate_certificate(certificate, identity)
            if config.get('apply') and certificate is None:
                raise ValueError('Operator frequency control requires a matching real coverage certificate')
        if plan:
            validate_plan(plan, identity)
            if digest_json(load_json(plan['profile_file'])) != plan['profile_sha256']:
                raise ValueError('Profile changed after plan generation; regenerate the v3 plan')
        self._sd_close_controller()
        if granularity != getattr(self, '_sd_granularity', 'module'):
            self._sd_trace.remove()
            if granularity == 'operator':
                from .operators import OperatorSpans
                self._sd_trace = OperatorSpans(self.model_runner.get_model(), torch, lambda: self._sd_schedule)
            else:
                self._sd_trace = Spans(self.model_runner.get_model(), torch, lambda: self._sd_schedule)
            self._sd_granularity = granularity
        if granularity == 'operator':
            self._sd_trace.configure(certificate, bool(config.get('diagnostic_profiler')))
        self._sd_cfg = dict(config, mode=mode)
        self._sd_identity_value = identity
        self._sd_enabled = False
        self._sd_request = None
        self._sd_plan = plan
        Path(config['out_dir']).mkdir(parents=True, exist_ok=True)
        write_json(Path(config['out_dir']) / 'worker_identity.json',
                   dict(identity=identity, pid=os.getpid(), worker_class=type(self).__name__,
                        layer_types=self._sd_trace.layer_types, apply=bool(config.get('apply'))))
        return identity

    def _sd_close_controller(self):
        controller = self._sd_fuse
        self._sd_schedule, self._sd_fuse = None, None
        if controller is not None:
            before = load_json(controller.directory / 'clock_snapshot.json')
            try:
                controller.close()  # CUDA drain/close precedes clock restoration.
            except Exception as exc:
                write_json(controller.directory / 'restore_status.json', dict(restored=False, error=repr(exc)))
                raise
            after = snapshot(controller.hardware)
            matched = restored(before, after)
            write_json(controller.directory / 'restore_status.json',
                       dict(restored=matched, snapshot=str(controller.directory / 'clock_snapshot.json'), after=after))
            if not matched:
                raise RuntimeError('Clock restoration readback differs from the snapshot')

    def _sd_arm(self, context):
        if context['request_id'] == self._sd_request:
            return
        config = self._sd_cfg
        enabled = bool(config.get('enabled', False))
        if config.get('control_file'):
            control = load_json(config['control_file'])
            enabled = bool(control['enabled'])
            config['run_id'] = control.get('run_id', 'service')
        if not enabled and self._sd_fuse and not config.get('keep_fixed_during_warmup'):
            self._sd_close_controller()
        self._sd_enabled, self._sd_request = enabled, context['request_id']
        if config['mode'] in ('serial', 'lookahead') and context['prompt_tokens'] != config['workload']['input_tokens']:
            raise RuntimeError('Prompt length differs from the profiled workload; use production-native or reprofile')
        if enabled and config['mode'] in ('serial', 'lookahead'):
            w, sampling = config['workload'], context['sampling']
            if (context['prompt_sha256'] != w['prompt_sha256'] or sampling != dict(
                    temperature=0.0, seed=0, max_tokens=w['output_tokens'],
                    min_tokens=w['output_tokens'], ignore_eos=True)):
                raise RuntimeError('Repro service requires the profiled token IDs and greedy/fixed-length sampling settings')
        if not enabled:
            return
        self.sdvfs_prepare()

    def sdvfs_prepare(self):
        """Initialize the controller at idle, outside the timed request window."""
        if self._sd_trace.active:
            raise RuntimeError('Prepare only when idle')
        config = self._sd_cfg
        if config['mode'] == 'instrumented' or self._sd_fuse:
            return {'prepared': True}
        hw = load_json(config['hardware'])
        policy = None
        if self._sd_plan and self._sd_plan['selector'] == 'paper':
            policy = RuntimePlanner(load_json(self._sd_plan['profile_file']), self._sd_plan, hw,
                                    calibration=self._sd_plan['calibration'], memory_policy='paper')
        self._sd_fuse = FUSE(hw, Path(config['out_dir']) / 'controller',
                             apply=bool(config.get('apply')), native_dry=not config.get('apply'))
        try:
            if Path('/results/mounts.json').exists():
                write_json(self._sd_fuse.directory / 'mounts.json', load_json('/results/mounts.json'))
            if config['mode'] == 'fixed':
                self._sd_fuse.set(config['fixed_triplet'])
            else:
                self._sd_fuse.enable_cuda(torch.cuda.current_device(),
                                         poll_us=config.get('poll_us', 100), expected_cuda=torch.version.cuda)
                self._sd_schedule = Schedule(self._sd_plan, self._sd_fuse, config['mode'], transport='cuda',
                                             runtime_policy=policy)
        except BaseException:
            self._sd_close_controller()
            raise
        return {'prepared': True}

    def sdvfs_profile_switches(self, grid):
        if self._sd_trace.active or self._sd_cfg['mode'] != 'fixed' or not self._sd_cfg.get('apply'):
            raise RuntimeError('Switch calibration requires an idle, applied fixed controller')
        from .inside import measure_switches
        try:
            return measure_switches(self._sd_fuse, self._sd_fuse.hardware, grid, self._sd_fuse.directory,
                                    repeats=self._sd_cfg.get('switch_repeat', 10))
        except BaseException:
            self._sd_close_controller()
            raise

    def sdvfs_profile_opp(self, triplet):
        if self._sd_trace.active or self._sd_cfg['mode'] != 'fixed' or not self._sd_fuse:
            raise RuntimeError('OPP update requires an idle fixed controller')
        self._sd_cfg['enabled'] = False
        self._sd_enabled, self._sd_request = False, None
        try:
            self._sd_fuse.set(triplet)
            tensor = torch.ones((512, 512), device='cuda', dtype=torch.bfloat16)
            def keepalive():
                _ = tensor @ tensor
                torch.cuda.current_stream().synchronize()
            matched = wait_frequencies(self._sd_fuse.hardware, triplet, keepalive=keepalive)
            return dict(matched, observation=observation(self._sd_fuse.hardware))
        except BaseException:
            self._sd_close_controller()
            raise

    def sdvfs_microbench(self):
        """Capacity estimates at the already pinned OPP; never inside request timing."""
        if self._sd_trace.active or self._sd_cfg['mode'] != 'fixed' or not self._sd_fuse:
            raise RuntimeError('Microbench requires an idle fixed controller')
        import statistics
        try:
            n, byte_count = 2048, 256*1024*1024
            a = torch.ones((n, n), device='cuda', dtype=torch.bfloat16)
            source = torch.ones(byte_count, device='cuda', dtype=torch.uint8)
            dest = torch.empty_like(source)
            def measure(function):
                for _ in range(2):
                    function()
                torch.cuda.synchronize()
                times = []
                for _ in range(10):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    function()
                    end.record()
                    end.synchronize()
                    times.append(start.elapsed_time(end)/1000)
                value = statistics.median(times)
                if value <= 0:
                    raise RuntimeError('Microbenchmark timer has insufficient resolution')
                return value
            gemm_s = measure(lambda: a @ a)
            copy_s = measure(lambda: dest.copy_(source))
            return dict(gemm_flops=2*n**3/gemm_s, streaming_logical_Bps=2*byte_count/copy_s,
                        gemm_shape=[n,n,n], copy_buffer_bytes=byte_count, repeats=10,
                        physical_dram_traffic_measured=False,
                        provenance='BF16 GEMM and uint8 streaming copy capacity estimates; not Qwen kernel performance')
        except BaseException:
            self._sd_close_controller()
            raise

    def sdvfs_observe(self, enabled, run_id):
        if self._sd_trace.active:
            raise RuntimeError('Observation gate can change only when idle')
        self.sdvfs_export(reset=True)
        self._sd_cfg.update(enabled=bool(enabled), run_id=run_id)
        self._sd_enabled, self._sd_request = False, None
        return {'enabled': bool(enabled)}

    def sdvfs_diagnostic(self, enabled):
        """Toggle only record_function markers at idle, not clock ownership."""
        if self._sd_trace.active or self._sd_granularity != 'operator' or not isinstance(enabled, bool):
            raise RuntimeError('Diagnostic markers require an idle operator worker and a boolean')
        self._sd_trace.diagnostic = enabled
        return {'diagnostic_profiler': enabled}

    def execute_model(self, scheduler_output):
        context = self._sd_tracker.step(scheduler_output)
        if context is None:
            return super().execute_model(scheduler_output)
        if not self._sd_cfg:
            raise RuntimeError('Configure SparseDVFS via worker RPC or SPARSEDVFS_WORKER_CONFIG')
        try:
            self._sd_arm(context)
            if not self._sd_enabled:
                return super().execute_model(scheduler_output)
            self._sd_trace.start(context, self._sd_cfg.get('run_id', 'run'))
            # Includes vLLM preprocessing, and is released by the first block.
            if self._sd_schedule:
                self._sd_fuse.cpu(self._sd_fuse.boost_khz)
            with self._sd_trace.dispatch_scope() if self._sd_granularity == 'operator' else nullcontext():
                result = super().execute_model(scheduler_output)
            if result is not None:
                raise RuntimeError('Expected V1 execute_model -> None -> sample_tokens for text generation')
            return result
        except BaseException as exc:
            self._sd_failure(exc, 'execute_model')
            self._sd_shutdown()
            raise

    def sample_tokens(self, grammar_output):
        if not getattr(self, '_sd_trace', None) or self._sd_trace.active is None:
            return super().sample_tokens(grammar_output)
        try:
            if grammar_output is not None:
                raise RuntimeError('Structured output introduces unprofiled sampling work')
            self._sd_trace.begin('sample')
            with self._sd_trace.dispatch_scope() if self._sd_granularity == 'operator' else nullcontext():
                result = super().sample_tokens(grammar_output)
            self._sd_trace.end('sample')
            self._sd_trace.finish()
            if self._sd_cfg.get('journal_during_request') and len(self._sd_trace.rows) >= 10000:
                self._sd_journal()
                if hasattr(self._sd_trace, 'reset'):
                    self._sd_trace.reset()
                else:
                    self._sd_trace.rows.clear()
                    self._sd_trace.steps.clear()
                self._sd_journal_rows = self._sd_journal_steps = self._sd_journal_ops = 0
            if self._sd_close_requested:
                self._sd_finish_close()
            return result
        except BaseException as exc:
            self._sd_failure(exc, 'sample_tokens')
            self._sd_shutdown()
            raise

    def _sd_journal(self):
        if not self._sd_cfg:
            return
        root = Path(self._sd_cfg['out_dir'])
        rows = getattr(self._sd_trace, 'operator_rows', self._sd_trace.rows)
        entries = [('_sd_journal_rows', 'worker_rows.jsonl', rows),
                   ('_sd_journal_steps', 'worker_steps.jsonl', self._sd_trace.steps)]
        if self._sd_granularity == 'operator':
            entries.append(('_sd_journal_ops', 'worker_operator_steps.jsonl', self._sd_trace.operator_steps))
        for attr, file, data in entries:
            start = getattr(self, attr)
            with (root / file).open('a', encoding='utf-8') as target:
                for row in data[start:]:
                    target.write(json.dumps(row) + '\n')
            setattr(self, attr, len(data))

    def sdvfs_export(self, reset=False, close=False):
        try:
            if self._sd_schedule:
                self._sd_schedule.finalize()
            data = self._sd_trace.export()
            self._sd_journal()
            data.update(identity=self._sd_identity_value, pid=os.getpid(),
                        runtime_repartitions=self._sd_schedule.repartitions if self._sd_schedule else [],
                        frequencies=frequencies(load_json(self._sd_cfg['hardware']))
                        if self._sd_cfg.get('hardware') else None)
            if self._sd_granularity == 'operator' and self._sd_trace.certificate is not None:
                data['coverage_certificate'] = self._sd_trace.certificate
            write_json(Path(self._sd_cfg['out_dir']) / 'worker_trace.json', data)
            return data
        finally:
            if close:
                self._sd_close_controller()
            if reset:
                if hasattr(self._sd_trace, 'reset'):
                    self._sd_trace.reset()
                else:
                    self._sd_trace.rows.clear()
                    self._sd_trace.steps.clear()
                self._sd_journal_rows = self._sd_journal_steps = self._sd_journal_ops = 0

    def _sd_request_close(self, signum, frame):
        self._sd_close_requested = True
        if self._sd_trace.active is None:
            self._sd_finish_close()

    def _sd_finish_close(self):
        root = Path(self._sd_cfg['out_dir'])
        request = load_json(root / 'close_request.json')
        nonce = request['nonce']
        try:
            if request.get('action', 'close') == 'prepare':
                self.sdvfs_prepare()
            else:
                self.sdvfs_export(close=True)
                self._sd_enabled = False
                self._sd_cfg['enabled'] = False
            ack = dict(nonce=nonce, closed=True)
        except Exception as exc:
            ack = dict(nonce=nonce, closed=False, error=repr(exc))
        self._sd_close_requested = False
        write_json(root / 'close_ack.json', ack)

    def _sd_shutdown(self):
        if not getattr(self, '_sd_cfg', None):
            return
        try:
            if self._sd_trace.active is None:
                self._sd_trace.reap(wait=True)
                self._sd_journal()
        finally:
            self._sd_close_controller()

    def _sd_failure(self, exc, stage):
        """OOM evidence comes from the process that actually owns the weights."""
        try:
            config = getattr(self, '_sd_cfg', None)
            if not config and os.environ.get('SPARSEDVFS_WORKER_CONFIG'):
                config = load_json(os.environ['SPARSEDVFS_WORKER_CONFIG'])
            if not config:
                return
            value = dict(stage=stage, error=repr(exc), oom='out of memory' in str(exc).lower(),
                         pid=os.getpid(), fallback_attempted=False)
            try:
                value['cuda_memory_summary'] = torch.cuda.memory_summary()
                value['cuda_free_total_bytes'] = list(torch.cuda.mem_get_info())
            except Exception as detail:
                value['memory_report_error'] = repr(detail)
            write_json(Path(config['out_dir']) / 'worker_failure.json', value)
        except Exception:
            pass  # Keep the original inference/cleanup error visible.
