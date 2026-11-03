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
from sparsedvfs_vllm.spans import Spans
from sparsedvfs_vllm.state import restored


class SparseDVFSWorker(Worker):
    def load_model(self, *, load_dummy_weights=False):
        try:
            super().load_model(load_dummy_weights=load_dummy_weights)
        except BaseException as exc:
            self._sd_failure(exc, 'load_model')
            raise
        validate_engine(self.vllm_config, vllm.__version__, self.use_v2_model_runner)
        if os.environ.get('SPARSEDVFS_GPU_UUID'):
            from .native import assert_cuda_uuid, check_vllm_sources
            assert_cuda_uuid(torch, os.environ['SPARSEDVFS_GPU_UUID'])
            check_vllm_sources()
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
        from .contracts import identity
        if os.environ.get('SPARSEDVFS_GPU_UUID') and config['gpu']['uuid'] != os.environ['SPARSEDVFS_GPU_UUID']:
            raise ValueError('Inference GPU and controller GPU UUID differ')
        return identity(config, vllm.__version__, torch.__version__, torch.version.cuda, torch.cuda.get_device_name())

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
        if config.get('apply') and config.get('gpu', {}).get('backend') == 'mock':
            raise ValueError('Mock backend cannot claim real apply')
        if config.get('gpu', {}).get('backend') != 'mock' and mode != 'instrumented' and not config.get('apply'):
            raise ValueError('Real GPU backend requires explicit apply')
        if config.get('mode')=='lookahead' and not config.get('switch_calibrated'):
            raise ValueError('Lookahead requires measured switch calibration')
        if config.get('apply') or mode == 'lookahead':
            calibration=load_json(config['calibration'])
            if calibration.get('gpu') != config['gpu'] or not calibration.get('measured'):
                raise ValueError('Missing matching measured GPU calibration')
            if config.get('apply') and calibration.get('synthetic'):
                raise ValueError('Mock calibration cannot authorize real GPU control')
            if os.environ.get('SPARSEDVFS_GPU_UUID'):
                from .native import environment_record
                if not calibration.get('restore_returned') or calibration.get('environment') != environment_record():
                    raise ValueError('Calibration environment/libraries differ; recalibrate in this Conda runtime')
        if config.get('apply') and mode == 'instrumented':
            raise ValueError('Instrumented control must not apply DVFS')
        granularity = config.get('granularity', 'module')
        if granularity not in ('module', 'operator'):
            raise ValueError('Unknown granularity')
        identity = self._sd_identity(config)
        plan = load_json(config['plan']) if mode in ('serial', 'lookahead') else None
        certificate = None
        if granularity == 'operator':
            from sparsedvfs_vllm.operator_contracts import validate_certificate
            if config.get('coverage'):
                certificate = load_json(config['coverage'])
            elif plan:
                certificate = plan.get('coverage_certificate')
            if certificate is not None:
                validate_certificate(certificate, identity)
            if config.get('apply') and certificate is None:
                raise ValueError('Operator frequency control requires a matching real coverage certificate')
        if plan:
            if config.get('apply') and plan.get('synthetic'): raise ValueError('Synthetic plans cannot authorize hardware control')
            validate_plan(plan, identity)
            if digest_json(load_json(plan['profile_file'])) != plan['profile_sha256']:
                raise ValueError('Profile changed after plan generation; regenerate the v3 plan')
            profile=load_json(plan['profile_file'])
            if config.get('calibration') and profile.get('calibration_sha256') != digest_json(load_json(config['calibration'])):
                raise ValueError('Calibration changed after profiling')
        self._sd_close_controller()
        if granularity != getattr(self, '_sd_granularity', 'module'):
            self._sd_trace.remove()
            if granularity == 'operator':
                from sparsedvfs_vllm.operators import OperatorSpans
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
            try:
                controller.close()
                write_json(controller.directory / 'restore_status.json', dict(reset_returned=True, original_policy='default_auto_ack', physical_restore_verified=False))
            except Exception as exc:
                write_json(controller.directory / 'restore_status.json', dict(reset_returned=False, error=repr(exc)))
                raise

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
            allowed = w.get('allowed_prompt_sha256', [w.get('prompt_sha256')])
            if (context['prompt_sha256'] not in allowed or sampling != dict(
                    temperature=0.0, seed=0, max_tokens=w['output_tokens'],
                    min_tokens=w['output_tokens'], ignore_eos=True)):
                raise RuntimeError('Repro service requires the profiled token IDs and greedy/fixed-length sampling settings')
        if not enabled:
            return
        self.sdvfs_prepare()

    def sdvfs_prepare(self):
        from .controller import Controller
        if self._sd_trace.active: raise RuntimeError('Prepare only when idle')
        config = self._sd_cfg
        if config['mode']=='instrumented' or self._sd_fuse: return {'prepared':True}
        self._sd_fuse = Controller(config['gpu'], Path(config['out_dir'])/'controller')
        try:
            if config['mode']=='fixed': self._sd_fuse.set(config['fixed_triplet'])
            else:
                self._sd_fuse.enable_cuda(torch.cuda.current_device(), expected_cuda=torch.version.cuda)
                self._sd_schedule = Schedule(self._sd_plan,self._sd_fuse,config['mode'],transport='cuda')
        except BaseException:
            self._sd_close_controller(); raise
        return {'prepared':True}

    def sdvfs_profile_switches(self, grid):
        raise RuntimeError('Use python -m rtx5090.run calibrate for GPU-only calibration')

    def sdvfs_profile_opp(self, triplet):
        if self._sd_trace.active: raise RuntimeError('Worker busy')
        self._sd_fuse.set(triplet)
        return {'requested':triplet}

    def sdvfs_gpu_readback(self):
        if self._sd_trace.active: raise RuntimeError('Readback only between requests')
        if not self._sd_fuse: return None
        return dict(observed_mhz=self._sd_fuse.backend.read(), synthetic=self._sd_fuse.backend.synthetic)

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
