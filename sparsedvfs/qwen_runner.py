"""Batch-one, text-only, eager Qwen3.5 runner. Imports GPU libraries on demand."""
import hashlib
import importlib.metadata
import json
import statistics
import time
from pathlib import Path
from .hardware import PowerMonitor, temperatures, write_json


def cuda_check():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Use a Thor/JetPack-compatible NVIDIA PyTorch build.')
    # Actual kernels test more than torch.cuda.is_available() or arch-list strings.
    x = torch.ones((32, 32), device='cuda', dtype=torch.bfloat16)
    _ = x @ x
    torch.cuda.synchronize()
    return {'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(), 'compute_capability': torch.cuda.get_device_capability(),
            'compiled_arches': torch.cuda.get_arch_list()}


def first_tensor(obj):
    import torch
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, (tuple, list)):
        for value in obj:
            found = first_tensor(value)
            if found is not None:
                return found
    return None


class Trace:
    def __init__(self, model, units, timing=False, sparsity=False, schedule=None):
        self.model, self.units = model, units
        self.timing, self.sparsity, self.schedule = timing, sparsity, schedule
        self.handles, self.rows, self.zeros, self.elements = [], [], {}, {}
        self.pending, self.call_order, self.forward_index, self.phase = {}, [], -1, None
        self.unit_names = [name for name, _ in units]

    def __enter__(self):
        import torch
        def root_pre(module, args, kwargs):
            if self.forward_index >= 0 and self.call_order != self.unit_names:
                raise RuntimeError('Qwen execution order differs from the profiled linear module chain')
            self.call_order = []
            self.forward_index += 1
            self.phase = 'prefill' if self.forward_index == 0 else 'decode'
            if self.schedule:
                self.schedule.start_forward(self.phase, self.forward_index)
        self.handles.append(self.model.register_forward_pre_hook(root_pre, with_kwargs=True))
        for name, module in self.units:
            def pre(mod, args, kwargs, name=name):
                self.call_order.append(name)
                if self.schedule:
                    self.schedule.before(self.phase, name)
                if self.timing:
                    event = torch.cuda.Event(enable_timing=True)
                    event.record()
                    self.pending[name] = (event, time.monotonic_ns(), self.phase)
            def post(mod, args, output, name=name):
                if self.timing:
                    end = torch.cuda.Event(enable_timing=True)
                    end.record()
                    start, cpu_start, phase = self.pending.pop(name)
                    self.rows.append((phase, name, start, end, (time.monotonic_ns() - cpu_start) / 1e6))
                if self.sparsity:
                    tensor = first_tensor(output)
                    if tensor is None or tensor.numel() == 0:
                        raise RuntimeError(f'No activation tensor for {name}')
                    key = (self.phase, name)
                    nonzero = int(torch.count_nonzero(tensor.detach()).item())
                    self.zeros[key] = self.zeros.get(key, 0) + tensor.numel() - nonzero
                    self.elements[key] = self.elements.get(key, 0) + tensor.numel()
                if self.schedule:
                    self.schedule.after(self.phase, name)
            self.handles.append(module.register_forward_pre_hook(pre, with_kwargs=True))
            self.handles.append(module.register_forward_hook(post))
        return self

    def result(self):
        if self.call_order != self.unit_names:
            raise RuntimeError('Incomplete module coverage in the final model forward')
        by_key = {}
        for phase, name, start, end, launch_ms in self.rows:
            by_key.setdefault((phase, name), []).append((start.elapsed_time(end), launch_ms))
        return [{'phase': phase, 'name': name, 'calls': len(values),
                 'latency_ms': statistics.mean(v[0] for v in values),
                 'cpu_launch_ms': statistics.mean(v[1] for v in values)}
                for (phase, name), values in by_key.items()]

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()


class Schedule:
    def __init__(self, plan, controller, mode, transport='cuda', runtime_policy=None):
        self.plan, self.controller, self.mode = plan, controller, mode
        self.transport, self.runtime_policy = transport, runtime_policy
        self.ticket, self.repartitions = None, []
        if transport == 'host' and mode == 'lookahead':
            raise ValueError('v2 lookahead requires the CUDA stream adapter; host transport supports serial debugging only')
        self.install(plan)

    def install(self, plan):
        self.plan = plan
        self.mapping, self.next = {}, {}
        for phase in dict.fromkeys(b['phase'] for b in plan['blocks']):
            blocks = [b for b in plan['blocks'] if b['phase'] == phase]
            for i, block in enumerate(blocks):
                for name in block['operators']:
                    self.mapping[(phase, name)] = block
                self.next[block['block_id']] = blocks[i + 1] if i + 1 < len(blocks) else None

    def start_forward(self, phase, step):
        if self.ticket is not None:
            raise RuntimeError('Previous submission scope was not closed')
        if self.runtime_policy:
            plan, observation = self.runtime_policy.replan(phase, step)
            self.install(plan)
            self.repartitions.append(observation)

    def before(self, phase, name):
        import torch
        block = self.mapping[(phase, name)]
        if name != block['operators'][0]:
            return
        upcoming = self.next[block['block_id']]
        if self.transport == 'cuda':
            # Stream-ordered host boundary + GPU start/end events. No device-wide
            # or per-block host CUDA synchronization in the v2 transport.
            self.ticket = self.controller.cuda_begin(torch.cuda.current_stream().cuda_stream, block, upcoming,
                                                       self.plan['switch_us'],
                                                       self.mode == 'lookahead' and block['amortized_strict_gt'])
        else:
            torch.cuda.current_stream().synchronize()
            self.controller.cancel()
            self.controller.set(block['triplet'], boost=True)

    def after(self, phase, name):
        block = self.mapping[(phase, name)]
        if name == block['operators'][-1]:
            if self.transport == 'cuda':
                self.controller.cuda_end(self.ticket)
                self.ticket = None
            else:
                self.controller.cpu(block['triplet'][0])

    def finalize(self):
        if self.ticket is not None:
            raise RuntimeError('Unclosed final submission scope')
        if self.transport == 'cuda':
            self.controller.cuda_drain()


class Runner:
    def __init__(self, model_path, prompt, tokens=32, dtype='bfloat16', attention='eager',
                 allow_download=False, revision=None, thinking=False):
        import torch
        import transformers
        from transformers import AutoConfig, AutoTokenizer
        self.torch, self.tokens = torch, tokens
        if tokens < 2:
            raise ValueError('at least two generated tokens are required for decode/TPOT')
        self.cuda_info = cuda_check()
        opts = {'local_files_only': not allow_download}
        if revision:
            opts['revision'] = revision
        cfg = AutoConfig.from_pretrained(model_path, **opts)
        if cfg.model_type == 'qwen3_5':
            cls = getattr(transformers, 'Qwen3_5ForConditionalGeneration', None)
        elif cfg.model_type == 'qwen3_5_text':
            cls = getattr(transformers, 'Qwen3_5ForCausalLM', None)
        else:
            raise ValueError(f'{cfg.model_type}: this demo supports dense Qwen3.5 text inference. MoE needs a separate routing-aware calibration.')
        if cls is None:
            raise RuntimeError('Installed transformers lacks Qwen3.5 classes; see requirements-thor.txt on the target.')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, **opts)
        self.model = cls.from_pretrained(model_path, dtype=getattr(torch, dtype),
                                         attn_implementation=attention, **opts).to('cuda').eval()
        ids = self.tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                                                 tokenize=True, add_generation_prompt=True,
                                                 enable_thinking=thinking)
        self.inputs = {'input_ids': torch.tensor([ids], dtype=torch.long, device='cuda'),
                       'attention_mask': torch.ones((1, len(ids)), dtype=torch.long, device='cuda')}
        modules = dict(self.model.named_modules())
        text_names = [n for n, m in modules.items() if m.__class__.__name__ == 'Qwen3_5TextModel']
        if len(text_names) != 1:
            raise RuntimeError('Cannot uniquely identify Qwen3_5TextModel; inspect the installed model implementation')
        base = text_names[0]
        text = modules[base]
        names = [base + '.embed_tokens'] + [base + f'.layers.{i}' for i in range(len(text.layers))]
        names += [base + '.norm', 'lm_head']
        self.units = [(n, modules[n]) for n in names]
        config_json = cfg.to_json_string(use_diff=False)
        backend_packages = {}
        for package in ('flash-attn', 'causal-conv1d', 'flash-linear-attention', 'triton'):
            try:
                backend_packages[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                backend_packages[package] = None
        local_root = Path(model_path)
        weight_files = []
        if local_root.is_dir():
            for pattern in ('*.safetensors', '*.safetensors.index.json', 'pytorch_model*.bin'):
                for p in sorted(local_root.glob(pattern)):
                    stat = p.stat()
                    weight_files.append({'name': p.name, 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns})
        self.identity = {'model_location': str(model_path), 'model_revision': revision,
                         'measurement_protocol': 'v2_current_main_stream_token_completion',
                         'resolved_hub_commit': getattr(cfg, '_commit_hash', None),
                         'local_weight_file_metadata': weight_files,
                         'config_sha256': hashlib.sha256(config_json.encode()).hexdigest(),
                         'model_class': self.model.__class__.__name__,
                         'input_token_ids': ids, 'new_tokens': tokens, 'dtype': dtype,
                         'attention': attention, 'thinking': thinking,
                         'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
                         'transformers': transformers.__version__, 'optional_backends': backend_packages,
                         'units': names, 'decoder_layer_types': [m.block_type for m in text.layers]}

    def warmup(self, repeats=2):
        for _ in range(repeats):
            self.generate({}, collect_power=False)

    def settle(self, hardware, triplet):
        from .hardware import wait_frequencies
        torch = self.torch
        x = torch.ones((256, 256), device='cuda', dtype=torch.bfloat16)
        def keepalive():
            _ = x @ x
            torch.cuda.synchronize()
        return wait_frequencies(hardware, triplet, keepalive=keepalive)

    def generate(self, hardware, timing=False, sparsity=False, schedule=None, collect_power=True):
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList
        timestamps = []
        class TokenClock(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                torch.cuda.current_stream().synchronize()
                timestamps.append(time.monotonic_ns())
                return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        trace = Trace(self.model, self.units, timing, sparsity, schedule) if timing or sparsity or schedule else None
        monitor = PowerMonitor(hardware.get('power') if collect_power else None)
        if trace:
            trace.__enter__()
        monitor.start()
        start = end = None
        try:
            torch.cuda.current_stream().synchronize()
            start = time.monotonic_ns()
            with torch.inference_mode():
                output = self.model.generate(**self.inputs, do_sample=False, use_cache=True,
                                             min_new_tokens=self.tokens, max_new_tokens=self.tokens,
                                             num_beams=1, logits_to_keep=1,
                                             pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                                             stopping_criteria=StoppingCriteriaList([TokenClock()]))
            torch.cuda.current_stream().synchronize()
            end = time.monotonic_ns()
            generated = output[0, self.inputs['input_ids'].shape[1]:].tolist()
            if len(generated) != self.tokens or len(timestamps) != self.tokens:
                raise RuntimeError('Generation length or callback count changed; do not reuse this profile')
            records = trace.result() if trace else []
        finally:
            monitor.stop()
            if trace:
                trace.__exit__(None, None, None)
        if schedule:
            schedule.finalize()  # timing-log retirement is outside the generation window
        result = {'elapsed_ms': (end - start) / 1e6, 'ttft_ms': (timestamps[0] - start) / 1e6,
                  'tpot_ms': (timestamps[-1] - timestamps[0]) / 1e6 / (len(timestamps) - 1),
                  'generated_tokens': len(generated), 'generated_token_ids': generated,
                  'text': self.tokenizer.decode(generated, skip_special_tokens=True),
                  'token_completion_monotonic_ns': timestamps,
                  'power': monitor.summarize(start, end), 'operators': records,
                  'instrumentation': {'token_completion_cuda_fence': True,
                                      'module_timing': timing, 'sparsity_reductions': sparsity,
                                      'block_boundary_cuda_fence': schedule is not None and schedule.transport == 'host',
                                      'transport': schedule.transport if schedule else None,
                                      'token_fence_scope': 'current_main_stream'},
                  'temperature_c': temperatures(hardware)}
        if schedule:
            result['runtime_repartitions'] = schedule.repartitions
        if trace and sparsity:
            result['sparsity'] = [{'phase': p, 'name': n, 'sparsity_exact_zero': trace.zeros[(p, n)] / size,
                                   's_comp': 0.0, 's_mem': 0.0}
                                  for (p, n), size in trace.elements.items()]
        result['power_samples'] = [{'monotonic_ns': t, 'watts': p} for t, p in monitor.samples]
        return result


def aggregate_profile(runner, hardware, grid, controller_factory, out_dir, repeats=3, warmups=2, stability_s=0.2):
    from .hardware import wait_frequencies, frequencies
    if repeats < 1:
        raise ValueError('repeats must be positive')
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    print('Collecting activation sparsity in a separate, untimed pass...', flush=True)
    sparse_run = runner.generate(hardware, sparsity=True, collect_power=False)
    sparse = {(r['phase'], r['name']): r['sparsity_exact_zero'] for r in sparse_run['sparsity']}
    candidates = []
    for index, triplet in enumerate(grid['triplets']):
        print(f'Profiling {index + 1}/{len(grid["triplets"])} OPP={triplet}', flush=True)
        candidate_dir = directory / f'opp_{index:03d}'
        with controller_factory(candidate_dir) as controller:
            controller.set(triplet)
            runner.settle(hardware, triplet)
            time.sleep(stability_s)
            runner.warmup(warmups)
            runs = [runner.generate(hardware, timing=True) for _ in range(repeats)]
            observed = frequencies(hardware)
        for run_index, run in enumerate(runs):
            write_json(candidate_dir / f'run_{run_index:03d}.json', run)
        if any(not r['power']['valid'] for r in runs):
            raise RuntimeError('Power measurement invalid/too short. Check hardware.power or increase --tokens. Raw runs saved.')
        power = statistics.mean(r['power']['average_power_w'] for r in runs)
        grouped = {}
        for run in runs:
            for op in run['operators']:
                grouped.setdefault((op['phase'], op['name']), []).append(op['latency_ms'])
        records = []
        for (phase, name), latencies in grouped.items():
            latency = statistics.mean(latencies)
            records.append({'phase': phase, 'name': name, 'latency_ms': latency,
                            'power_w': power, 'energy_proxy_j': power * latency / 1000,
                            'sparsity_exact_zero': sparse[(phase, name)], 's_comp': 0.0, 's_mem': 0.0})
        candidates.append({'triplet': triplet, 'operators': records,
                           'observed_at_end': observed,
                           'total_latency_ms': statistics.mean(r['elapsed_ms'] for r in runs),
                           'total_energy_j': statistics.mean(r['power']['energy_j'] for r in runs),
                           'token_ids': runs[0]['generated_token_ids']})
    return {'schema': 1, 'origin': 'measured_Thor_Qwen_reference_port', 'synthetic': False,
            'dvfs_applied': True, 'identity': runner.identity, 'cuda': runner.cuda_info,
            'units': [n for n, _ in runner.units], 'reference': grid['reference'],
            'candidates': candidates, 'degraded_no_emc': hardware['degraded_no_emc'],
            'power_attribution': 'whole_inference_average_proxy',
            'raw_sparsity_run_excluded_from_latency_and_energy': True,
            'all_opp_token_ids_equal': all(c['token_ids'] == candidates[0]['token_ids'] for c in candidates)}
