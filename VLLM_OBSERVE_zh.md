# 第一次交付：观察真实 vLLM 执行单元

本工具先取得实际执行清单，再决定如何标定 Eq. (1)。入口为 `python -B -m rtx5090.observe doctor/collect/summarize`。它使用独立 `ObservationWorker`，不加载旧 SparseDVFS worker、FUSE 或频率控制器，不需要 bridge 或 token，不写频率。

本次只交付观察与清单汇总。回放适配、独立参数验证和最终预测待目标机真实清单返回后进行；不能把 `inventory_only` 理解为 Eq. (1) 验证通过。

## 与已成功启动命令的关系

2026-11-03 用户提供的成功命令为 `vllm serve`，模型路径 `/mnt/d/models/Qwen3.5-27B`，dummy、TP=2、max-model-len=4096、max-num-seqs=1、gpu-memory-utilization=0.90、reasoning-parser=qwen3。其环境还包含三个 NCCL 禁用开关。

观察入口保留上述模型、并行规模和内存配置，并显式设置已确认计划要求的 BF16、eager、V1、language-model-only；这些在用户提供的服务命令中未显式设置，因此本工具不声称原服务默认执行路径已被原样验证。HTTP、聊天模板及 reasoning parser 不参与固定 token-ID 离线请求。

默认关闭 prefix caching，确保相同输入的预热不会让正式请求跳过完整 prefill。分块 prefill 和调度器行为按照实际配置记录，不套用旧工具禁止分块的断言。三个 `NCCL_*_DISABLE=1` 在本进程及子进程设置并校验，不修改系统配置。

默认输入长度 128/512/2048，每个场景先预热一次，再观察三次；每次生成恰好 64 token，并发为 1。输入来自本机 tokenizer 的合法非特殊 token ID，固定生成规则，`ignore_eos=True`，保留输入输出 ID。dummy 输出不用于模型质量或有效稀疏验证。

1024-token 场景只写入清单为 `sealed_not_collected`，第一版不提供采集它的选项，避免提前泄漏最终场景。

## 1．更新代码并释放双卡

在原服务终端正常停止自己启动的 vLLM 服务。若 bridge 仍运行，也在它的窗口正常退出，先处理任何恢复失败；本轮应处于默认自动频率状态。观察程序不负责恢复其他任务设置的锁频。

在 **WSL Bash** 执行：

```bash
cd /mnt/d/s00958243/SparseDVFS
git status --short
git fetch origin
git switch 5090
git merge --ff-only origin/5090
git log -1 --oneline
```

出现冲突或分叉时停止，保留本地修改，不强制覆盖。

## 2．先运行只读 doctor

继续在同一个 **WSL Bash**：

```bash
conda activate vllm-qwen
export PYTHONPATH="/mnt/d/s00958243/triton-360:${PYTHONPATH:-}"
export SPARSEDVFS_TRITON_PATH=/mnt/d/s00958243/triton-360
export TRITON_CACHE_DIR=/tmp/triton360-vllm-cache
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1

OBS_OUT="$PWD/runs/vllm-observe-$(date +%Y%m%d-%H%M%S)"
python -B -m rtx5090.observe doctor --out "$OBS_OUT"
```

缓存路径按用户最新成功启动脚本保留；独立 Triton 包仍位于持久目录。缓存被清理可以重建，不能把缓存路径和包安装路径混淆。

预期 `status: ready`。`target_worker_verified: false` 是正常的：doctor 只核对本机包、已审阅 vLLM 源码、两张 CUDA 卡的 UUID、模型配置文件及待用参数，尚未启动 TP workers。

`doctor.json` 保存检查结果，不保存 token。任何失败先停在此步。源码不匹配时反馈实际错误，不删除版本/源码检查，不安装或降级环境。

## 3．采集开发场景

doctor 成功后，继续在同一终端：

```bash
python -B -m rtx5090.observe collect --out "$OBS_OUT"
```

该命令自行启动双卡 dummy 模型，不连接已有 HTTP 服务。预期依次显示三个 warmup 和九条 `both ranks exported`。完成后 `status: collected`、requests=9，自动关闭本命令启动的 engine。

真实 worker 的 PID、UUID、dtype、TP 配置、V1/eager、Triton 路径及源码哈希均需通过检查。rank 0 对应 `GPU-22d1cd22-d198-45c4-2b54-974484026ab5`，rank 1 对应 `GPU-e2f02606-bac6-52cd-f8a1-15a027db7da0`。

采样会增加运行时间和主机内存占用，每个观察请求单独导出 profiler 文件。默认每个 rank、每个请求最多记录 250000 次 dispatcher 调用，超过即保存失败证据，不静默截断。RPC 默认等待上限 600 秒；超时不代表采样成功。

若 GPU 内存不足、运行模式不兼容或任一 rank 失败，保留整个目录；不自动改变 dtype、内存比例、并行规模或执行方式。修复后使用新的 `OBS_OUT` 重跑 doctor/collect。

如确需匹配成功配置中的其他参数，`--engine-config PATH.json` 支持有限白名单：内存比例、最大长度、批 token 数、prefix/chunked prefill、async scheduling、attention/compilation 配置、KV cache dtype、custom all-reduce 开关。doctor 和 collect 必须传同一文件。BF16、dummy、TP=2、V1、eager、单请求不允许通过该文件替换；实际配置会在 worker identity 中再次记录。

## 4．离线汇总

collect 成功后：

```bash
python -B -m rtx5090.observe summarize --out "$OBS_OUT"
```

此命令可在无 GPU、无 torch/vLLM 的机器上运行。它只读完整的九个请求、十八份 rank 导出结果，生成 `inventory.json`。重复执行不会覆盖已有报告。

| 产物 | 内容 |
| --- | --- |
| `doctor.json` | 软件/源码/UUID 检查与计划启动参数 |
| `manifest.json` | 工作负载、实际 rank 身份、配置差异、请求结果、engine 退出状态 |
| `warmup/*.json` | 未被 profiler 观察的预热输入输出 |
| `requests/*/generation.json` | 固定请求输入输出 ID，与预热输出是否一致另记在 manifest |
| `requests/*/rank-*/identity.json` | 实际 worker/runner/model 类型、源码、环境与设备 |
| `requests/*/rank-*/trace.json` | 原始 Chrome profiler 证据 |
| `requests/*/rank-*/request.json` | 调度步、输入输出元数据、dispatcher scopes、GPU 关联及未归属活动 |
| `inventory.json` | 全部唯一签名、父模块、调用次数、GPU 活动时间、未知部分及支持状态 |

首先反馈 doctor/collect/summarize 的终端输出以及 `manifest.json`；后续依据实际清单选择回放适配器。不要对这些 profiler 文件运行旧的 `performance fit`。

## 解释边界

- 元数据包括输入输出 shape、dtype、device、stride、layout；不复制模型权重、KV cache 或中间 tensor 值。未知 Python 对象只记类型，依赖它的操作尚不具备回放条件。
- 父模块来自实际模型 hooks，无固定 64 层或模块顺序假设。保留 prefill/decode、分块、缓存命中、恢复和推测 token 的调度信息。
- CPU launch 在同进程同线程内归属 dispatcher scope，再按 correlation ID 关联 GPU 活动；不跨 CPU/GPU 时间轴猜测归属。未归属或歧义活动保留在总活动时间分母中。
- CUDA Event 只记录当前 stream 区间，可能含等待；隐藏 stream 和重叠工作由 profiler 进一步解释。采样不逐算子同步，仅在请求导出边界同步。
- GPU 活动时间之和可包含重叠。跨两个 rank 相加是工作量统计，不能当作请求延迟。
- 签名按 rank、阶段、执行名、输入输出元数据及实际 kernel 名称区分；opaque 操作额外保留上下文长度等状态。操作不被拆成猜测的普通 GEMM。
- 分类来自名称提示，不是已证明的性能瓶颈。所有签名初始均为 `replay_status: not_implemented`、`equation_validated: false`。
- 本机测试使用 CPU 替身，目标 5090 上的观察兼容性和完整性需要实测确认。第一交付完成不等于整份 Eq. (1) 计划完成。

## 已修复：TorchVersion 无法通过 RPC 序列化

若日志出现 `TorchVersion is not serializable`，是观察 worker 返回版本对象的类型问题。修复将版本字段显式转换为内置字符串，不需要 `VLLM_ALLOW_INSECURE_SERIALIZATION=1`。`Cudagraph is disabled under eager mode` 是正常提示。

先在采集终端 Ctrl+C 结束本次进程，再 fetch/快进更新。保留失败目录，新建 OBS_OUT，重新运行 doctor/collect；旧 doctor 绑定旧源码，不能复用。若 Ctrl+C 后仍有本次 worker 占用 GPU，先确认并停止本次残留进程，不结束其他任务。

## 已修复：CPU/GPU 同名 annotation 被误判重复

2026-11-03 用户 trace：每 rank 211684 条 CPU `user_annotation`，151406 条同名 `gpu_user_annotation`，无缺失标记。旧代码不区分类别而误报重复；现在仅用 CPU `user_annotation` 建立 launch 所属范围，GPU annotation 保留在原始 trace 中但不当作额外 kernel。真实 CPU 重复或缺失仍报错，未知 GPU 活动继续保留。

可先离线读取旧 trace/request，调用 `rtx5090.observe_core.correlate(events, rows)` 检查关联；不要改旧失败记录的 status，不将单个请求视作九请求采集完成。完整采集仍需新目录 doctor/collect。
