# vLLM 0.32.0 / Conda compatibility evidence

The target machine is not attached to the implementation host. User-provided package/source evidence is distinguished from tests run here. The installed target six-file hashes match upstream tag `v0.32.0`; the entry point checks these exact hashes in `rtx5090.native.V032_SOURCES` before inference. A local patch or future version requires a fresh audit, not deleting a version check.

| Interface | Evidence and adaptation |
|---|---|
| Worker | [gpu_worker.py](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/worker/gpu_worker.py): keyword `load_model(load_dummy_weights=False)`, `execute_model(scheduler_output)`, `sample_tokens(grammar_output)` remain available. Subclass methods keep these signatures. |
| V1 execution | [gpu_model_runner.py](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/worker/gpu_model_runner.py): execute computes logits and saves execution state, then sampling completes the token. Hooks enforce embed → 64 layers → norm → logits → sample for every prefill/decode step. Non-None execute returns, missing/reordered/unclosed units fail. |
| V2 incompatibility | [gpu/model_runner.py](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/worker/gpu/model_runner.py): `sample_tokens` calls `sample`, which computes logits. Starting the sample span before that would violate the old order. V2 is explicitly disabled and rejected, not claimed compatible. |
| Runner default | [config/vllm.py](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/config/vllm.py): V2 can be selected by default; the launch process sets `VLLM_USE_V2_MODEL_RUNNER=0` before importing vLLM. |
| Worker RPC | [entrypoints/llm.py](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/entrypoints/llm.py): public `collective_rpc(method, timeout, args, kwargs)` supports the custom methods and a callable worker UUID/config audit. Exactly one result is required. |
| Scheduler fields | [sched/output.py](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/core/sched/output.py): new/cached request counters and scheduling fields match the phase tracker. Runtime checks still reject preemption, chunked prefill, multimodal, speculation and batching. |
| Shutdown | [core_client.py](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/engine/core_client.py): `llm.llm_engine.engine_core.shutdown()` exists; controller close precedes worker shutdown. Each benchmark window is a separate process. |

The old strict 0.19-only check is replaced with supported-version checks **plus source, runtime config, model structure, GPU UUID, output and boundary validation**. Docker 0.19 remains a historical entry; Conda requires the audited 0.32 V1 source. Plan protocol is bumped; old Thor/BF16/0.19 profiles do not authorize the new experiment.

User target: Python 3.12.16, vLLM 0.32.0+cu129, torch 2.13.0+cu129, installed Triton 3.7.1, transformers 5.19.0, g++ 13.3, driver 576.80, two 5090 cards. The user reports Qwen startup requires the independent Triton 3.6.0 override. `--triton-path` now explicitly selects that existing package without changing Conda. The actual imported version/path/package SHA256 and distribution metadata are recorded separately; child processes receive the same selection. This supports the chosen configuration but does not establish the root cause of the 3.7.1 failure or prove real INT4/CUDA compatibility. The target-machine baseline/chain gates still apply.

Model/tokenizer paths propagate into model verification, engine construction, worker configuration, dataset identity and subprocess argv. The model revision/INT4 checks remain enforced. Dummy uses a separate tiny Llama config and native worker, explicitly excludes SparseDVFS hook/energy acceptance, and cannot use `--apply`.

Source audit and dependency-free fixtures are **not installed-vLLM execution**. The real acceptance gates remain the target machine's baseline, CUDA+mock chain, clock calibration, profile and paired experiment logs.
