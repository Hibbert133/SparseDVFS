"""Runtime evidence from the pinned container; executed only on the target."""
import hashlib
import inspect
import json
from pathlib import Path
import torch
import vllm
from sparsedvfs.hardware import write_json


def main():
    from vllm.v1.worker import gpu_worker, gpu_model_runner, worker_base
    from vllm.model_executor.models import qwen3_5
    paths = [inspect.getsourcefile(module) for module in (gpu_worker, gpu_model_runner, worker_base, qwen3_5)]
    value = dict(vllm=vllm.__version__, torch=torch.__version__, torch_cuda=torch.version.cuda,
                 cuda_available=torch.cuda.is_available(),
                 sources=[dict(path=path, sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest()) for path in paths])
    try:
        from torch.utils._python_dispatch import TorchDispatchMode
        from vllm.config.profiler import ProfilerConfig
        source = inspect.getsourcefile(TorchDispatchMode)
        value['dispatcher_source'] = dict(path=source,sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest())
        activities = torch.profiler.supported_activities()
        value['profiler_activities'] = [str(a) for a in activities]
        value['operator_diagnostics_available'] = torch.profiler.ProfilerActivity.CUDA in activities
        value['profiler_config_source'] = inspect.getsourcefile(ProfilerConfig)
    except (ImportError,AttributeError) as exc:
        value.update(operator_diagnostics_available=False,operator_diagnostic_error=str(exc))
    if vllm.__version__.split('+')[0] != '0.19.0':
        raise RuntimeError(f'Unexpected image vLLM version: {vllm.__version__}')
    x = torch.ones((32, 32), dtype=torch.bfloat16, device='cuda')
    _ = x @ x
    torch.cuda.synchronize()
    value.update(device=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
                 compiled_arches=torch.cuda.get_arch_list(), bf16_matmul='PASS')
    write_json('/results/container_runtime.json', value)
    print(json.dumps(value, indent=2))


if __name__ == '__main__':
    main()
