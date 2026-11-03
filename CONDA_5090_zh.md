# Windows + WSL2 + Conda：手动复现与效果验收

本入口使用现有 vLLM **0.32.0 V1**、单张 5090、真实 INT4 权重。无需 Docker、`vllm serve` 或 Agent 常驻。不安装依赖、不改驱动/网络、不操作第二张卡。当前新增代码经过 CPU/mock 测试；真实 CUDA、量化模型、NVML 控制和能耗结果必须在目标机执行后验收。

“普通 DVFS”在这里是 **NVIDIA 默认自动频率**。成功条件为独立测试集的 GPU J/token 下降，配对 bootstrap 95% 区间及遥测误差余量支持下降，平均请求延迟增幅 ≤5%；同时报告 p95。结论只针对下面的单请求 512/128 token 工作负载，不是整机节能或完整论文收益。

## 0. 更新代码，保留原环境与已有验证记录

在 WSL Bash 逐条执行。以下更新不修改已有四个 `validation/` 文件；如 Git 报本地文件冲突，先保留它们，不要 `reset --hard`。

```bash
cd /mnt/d/s00958243/SparseDVFS
git status --short
git fetch origin
git switch 5090
git merge --ff-only origin/5090
conda activate vllm-qwen
PY=/home/s00958243/miniforge3/envs/vllm-qwen/bin/python
MODEL=/mnt/d/models/Qwen3.5-27B-AWQ-4bit
GPU=GPU-22d1cd22-d198-45c4-2b54-974484026ab5
LIB="$PWD/build"
RUN="$PWD/runs/conda-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN"
sd() { "$PY" -B -m rtx5090.conda "$@" --model "$MODEL" --gpu-uuid "$GPU" --library-dir "$LIB"; }
```

`sd` 只是当前 Bash 的函数，不是后台服务。每条命令退出后均可独立继续。保留这个终端中的 `RUN` 路径；新终端重新设置这些变量即可。

产生 baseline/calibration/profile/tuning/compare 证据的命令拒绝覆盖已有结果。重新实验时使用新的 `RUN` 或 `--out`，避免失败后误用上一次成功的文件。

```bash
sd doctor --out "$RUN/doctor"
sd cpu-test --out "$RUN/cpu"
```

预期：doctor 显示当前 Python/包实际路径、vLLM 六个源文件哈希、CUDA 设备及头文件/运行库候选；CPU 测试 `successful: true`。仅 CUDA 测试可跳过。这里不会给你的真实 GPU 写频率。

若 `PYTHONPATH` 仍包含 `/tmp/triton-360-test`，运行入口会拒绝。使用不执行旧启动脚本的新终端，先确认当前加载的是 Conda 的 Triton 3.7.1。不要删除临时目录或全局替换 Triton。本项目尚未取得 3.7.1 启动失败的实机 traceback；若有错误，保留完整 `process.log` 再决定隔离方案。

## 1. 构建本地动态库；先只读探测控制能力

```bash
sd build --cuda --out "$RUN/build"
sd probe --ack-default-auto --out "$RUN/capabilities"
```

构建产物：`build/libfuse_dvfs.so`、`build/libfuse_cuda.so`、`build/build.json`。构建用 g++ 和已有 CUDA Runtime，无设备 kernel 编译，不依赖 nvcc/CMake；检查实际 runtime 与 torch CUDA 的版本及两个 ABI。构建成功不等于 CUDA hook 已执行。

若缺少头文件或 libcudart，先查 doctor。可明确选择 **同一现有 CUDA 12.9** 的文件：

```bash
# 将下列两处替换为 doctor 列出的真实路径；不要原样运行占位路径。
sd build --cuda --cuda-include /已有路径/include --cudart /已有路径/lib/libcudart.so.12
```

找不到兼容文件时停止 CUDA 阶段，不自动安装 toolkit、不更换 torch。编译器 warning 与测试失败分开记录。

probe 只查询、不尝试锁频或 reset。它必须返回所选 UUID、支持频点及遥测。`set_reset_verified: false` 是正常的：实际写入能力留到第 5 步。如果错误为 NVML 不支持/权限不足，不能声称调频可用；可按后面的 Windows 桥接方法显式重试。

## 2. 准备真实模型与固定输入

当前 `/mnt/d/models/Qwen3.5-27B` 是 BF16 配置，索引引用的 11 个权重分片不存在；旧脚本的 `--load-format dummy --tensor-parallel-size 2` 不能作为真实权重基线。保留原目录。

以下命令会下载真实权重到新目录，磁盘与网络开销由你执行时承担：

```bash
sd download --out "$RUN/download"
sd dataset --out "$RUN/data"
DATA="$RUN/data/dataset.json"
```

模型固定为 `cyankiwi/Qwen3.5-27B-AWQ-4bit`、revision `9d8d63632afafe90bdfcb70164d3e2a0955619f7`。名称含 AWQ，但实际是 compressed-tensors INT4 group32；不添加 `--quantization awq`。下载完成后生成并检查模型锁及文件哈希，真实实验仍拒绝 BF16/dummy/缺失分片。

数据为 8 个训练、8 个独立测试 prompt，每个恰好 512 输入 token，固定 greedy 128 输出 token。输入由脚本构造，输出来自真实模型；不要把此单一工作负载的结果外推为通用收益。

可选的小模型 dummy 接口检查（不需要上述权重下载完成）：

```bash
sd smoke --out "$RUN/dummy"
```

预期 `synthetic: true`、`deployable: false`。它只验证原生离线 vLLM 接口，不证明 Qwen hooks、CUDAFUSE 或调频成功，也不能生成真实 profile/plan。

## 3. 原生真实权重基线

确认选定卡目前处于默认自动频率且没有其他实验负载；`--ack-default-auto` 是对此前提的明确确认，不会执行 reset。若不确定已有锁频来自谁，先核实，不能随意重置其他人的设置。

```bash
sd baseline --dataset "$DATA" --ack-default-auto --out "$RUN/baseline"
BASE="$RUN/baseline/baseline.json"
```

预期：真实模型加载、单卡 UUID 审核通过，保存 16 个 prompt 的 token 输出哈希和负载期间频率遥测。基线采用原生 Worker，无 SparseDVFS hooks。这里的 `energy_measured: false` 是预期值，正式配对能耗实验在第 7 步。

OOM、量化 kernel 错误、source hash 不符或输出不完整均停止，不自动换模型、换精度或换 Triton。错误日志先于降级建议。

## 4. 真实 CUDA + mock 调频执行链

```bash
sd chain --dataset "$DATA" --baseline "$BASE" --out "$RUN/chain"
```

预期：`result.json` 中 `real_cuda: true`、`synthetic: true`、`frequency_applied: false`、`coverage_complete: true`、`output_checked: true`、`restoration_ok: true`。检查真实 prefill + 连续 decode 的 68 个逻辑 unit；包含跨 execute/sample 的 super-block 闭合。此步骤不写硬件频率。

动态库加载、UUID、顺序、边界或输出哈希任一失败都停止。此阶段覆盖 module 主流边界，不构成 operator 路径的全 CUDA 活动覆盖证书。

## 5. 选择频点并显式标定真实切换

```bash
sd probe --baseline "$BASE" --ack-default-auto --out "$RUN/probe"
GPUCFG="$RUN/probe/gpu.json"
sd calibrate --gpu "$GPUCFG" --ack-default-auto --apply --out "$RUN/calibration"
CAL="$RUN/calibration/calibration.json"
```

频点为支持列表中最接近负载观测 p95 频率 100%、90%、80%、70% 的值，去重后至少两个。CPU/显存仍由系统管理。

**第一条实际锁频命令是带 `--apply` 的 calibrate。** 测试每一对频点的往返调用、读回和恢复。预期 `synthetic: false`、`measured: true`，所有读回在 ±15 MHz 容差内；`physical_pll_settle_verified: false` 明确表示它只证明调用与读回，不是物理 PLL 测量。切换耗时包含桥接往返，采用调用方单调时钟。

如果失败，保留错误和恢复记录，停止后续实调频，不回退 mock。标定和 profile 与环境、源码、动态库哈希绑定；更新代码、库、GPU 配置后重新运行对应前置步骤。

## 6. 真实 profile、分块及训练集选方案

```bash
sd profile --gpu "$GPUCFG" --dataset "$DATA" --baseline "$BASE" --calibration "$CAL" --ack-default-auto --apply --out "$RUN/profile"
PROFILE="$RUN/profile/profile.json"
sd plan --profile "$PROFILE" --out "$RUN/candidates"
sd tune --gpu "$GPUCFG" --dataset "$DATA" --baseline "$BASE" --calibration "$CAL" --profile "$PROFILE" --ack-default-auto --apply --out "$RUN/tune"
SELECTION="$RUN/tune/selection.json"
```

每个频点保存真实 CUDA unit 延迟、整请求能耗、遥测、输出及读回。能量优先使用 NVML 累计 mJ 计数器，转换为 J；所有能量窗口至少 30 秒。

分块复用已有 Algorithm 1 参考代码：短块或近似频率合并，合并频率取保守最大值，保留终末短块标记。三个候选的 module 延迟预算为 0%、5%、15%；**最终端到端限制始终为 5%**。这些参数和粗粒度路径不是论文未提供细节的“标准答案”。

训练阶段实际测量串行候选，再测最优串行候选的 lookahead；只有满足延迟约束且能量改善超过误差余量才选用 lookahead。最终以训练集实测整请求 J/token 选方案，不用低采样率功率硬推每层能量。`selection.json` 锁定方案、数据集、环境及标定哈希，不是收益证明。

需要逐条观察某个候选时，可在训练集单独运行（只作诊断，不替代独立验收）：

```bash
sd run --mode serial --plan "$RUN/candidates/candidate_1.json" --gpu "$GPUCFG" --dataset "$DATA" --baseline "$BASE" --calibration "$CAL" --ack-default-auto --apply --out "$RUN/manual-serial"
sd run --mode lookahead --plan "$RUN/candidates/candidate_1.json" --gpu "$GPUCFG" --dataset "$DATA" --baseline "$BASE" --calibration "$CAL" --ack-default-auto --apply --out "$RUN/manual-lookahead"
```

如果所有候选超时，命令失败且不产生新 selection；若可行方案也不节能，可以保留负结果做第 7 步验证，不能说已复现。不要反复根据测试集改候选。

## 7. 自动频率 vs 冻结方案：独立配对验收

```bash
sd compare --gpu "$GPUCFG" --dataset "$DATA" --baseline "$BASE" --calibration "$CAL" --selection "$SELECTION" --ack-default-auto --apply --pairs 10 --out "$RUN/compare"
sd report --out "$RUN/compare"
```

每个窗口在新进程中加载同一真实模型并预热 3 次；两臂使用相同请求数和输入顺序，窗口顺序随机。时长按训练输入的 pilot 确定，不根据测试收益挑选时长。自动频率窗口在预热前执行 reset，因此这里同样要求 `--apply`。测量不包含模型加载/预热，但包含请求内调频和 SparseDVFS 插桩开销。离线 profile/tune 总耗时另记，不把它抵扣进收益。

结果在 `report.json`，原始数据为 `pairs.json` 和每个窗口的 `process.log`、`result.json`、`telemetry.json`、`measured_trace.json`、`worker/`。`report` 只重新分析已有数据，不运行 GPU。

- `reproduced_for_this_workload` 且 `success: true`：能量区间通过并且平均延迟增幅 ≤5%。仍不代表整机节能或论文完整复现。
- `not_reproduced_or_inconclusive`：没有足够证据。查看能量置信区间、误差余量、延迟和切换次数，不删负结果。
- 遥测、输出、身份、边界、窗口时长或恢复不合格：明确报错，不能生成成功结论。

每个窗口同时记录温度、频率、GPU 利用率、显存和 `nvidia-smi` 进程查询。WSL 可能无法报告全部进程；实验前排除选定卡的背景负载，实验中不要启动其他 GPU 工作。后台负载或温度趋势异常时整次实验标记受干扰，不能只保留有利窗口。

## 累计能量不可用时

不支持能量计数器时，默认拒绝能耗验收。只有已实际完成遥测精度校验，才可使用 `--power-evidence /真实证据.json`，并将同一参数加到 profile/tune/compare。至少需要：所选 `uuid`、`validated: true`、真实 `reference_method`、`max_relative_error`（0～0.1）和 `max_sample_interval_s`。保存参考仪器/计数器对照数据，不能自行填一个假精度。

积分使用同一遥测源时间戳和梯形法，桥接时不跨系统相减时钟。验收另外要求下降幅度超过两臂误差余量；计数器路径也保留至少每臂 0.5% 的保守余量，它不是 NVIDIA 传感器精度认证。如果没有可靠计量条件，只交付时序/执行链结果，能耗结论为无法判定。

## WSL NVML 不支持时：Windows 用户态桥接

仅在现有网络已允许 WSL 访问 Windows loopback 时使用。连接失败就报告，不修改 WSL/Windows 网络、驱动、防火墙。Windows 必须已有 Python；否则此路线暂不可用。

在 Windows PowerShell（使用同一份更新后的仓库）：

```powershell
cd D:\s00958243\SparseDVFS
$env:SPARSEDVFS_BRIDGE_TOKEN = [Guid]::NewGuid().ToString('N') + [Guid]::NewGuid().ToString('N')
# 将此值安全地复制到自己的 WSL 终端，不写入仓库/报告。
$env:SPARSEDVFS_BRIDGE_TOKEN
py -3 -B -m rtx5090.bridge --uuid GPU-22d1cd22-d198-45c4-2b54-974484026ab5 --ack-default-auto --log bridge.jsonl
```

上面是只读桥接。WSL 终端：

```bash
read -rsp 'Bridge token: ' SPARSEDVFS_BRIDGE_TOKEN; echo
export SPARSEDVFS_BRIDGE_TOKEN
sd probe --backend bridge --ack-default-auto --out "$RUN/bridge-capabilities"
sd baseline --backend bridge --dataset "$DATA" --ack-default-auto --out "$RUN/bridge-baseline"
BASE="$RUN/bridge-baseline/baseline.json"
sd probe --backend bridge --baseline "$BASE" --ack-default-auto --out "$RUN/probe"
```

`GPUCFG` 的 backend 现在是 bridge。进行第 5 步之前，Windows Ctrl+C 关闭只读进程，用相同 token/UUID 重新启动并显式加 `--apply`；WSL 的 calibrate/profile/tune/compare 也仍需各自 `--apply`：

```powershell
py -3 -B -m rtx5090.bridge --uuid GPU-22d1cd22-d198-45c4-2b54-974484026ab5 --ack-default-auto --apply --log bridge.jsonl
```

控制连接持久复用；独立只读遥测连接不会抢占或 reset 控制卡。GPU 写入单连接独占、UUID/会话 token/请求序号检查；心跳维持连接，客户端崩溃后最多约 30 秒空闲超时尝试恢复。认证失败、断连和 reset 失败都不能返回成功。

## 失败后的恢复与证据边界

正常结束、Python 异常会关闭控制器并 reset 选定 GPU；`worker/controller/restore_status.json` 记录调用结果。SIGKILL、主机掉电或驱动失效无法保证自动恢复。停止本实验后，可显式运行：

```bash
sd recover --gpu "$GPUCFG" --ack-default-auto --apply --out "$RUN/recovery"
```

若连动态库都无法加载，可在支持 NVML 的 Windows 终端手动执行 `nvidia-smi -i GPU-22d1cd22-d198-45c4-2b54-974484026ab5 -rgc`，只恢复该卡，保留返回码。reset 返回成功也不等于独立仪器验证了原始锁频状态；本实验只支持“默认自动频率开始，结束 reset”的约定。

尚未完成：目标机 CUDA/INT4/显存验收、硬件锁频与桥接延迟、实际节能结论；论文稀疏感知功耗公式重新标定、kernel DAG/operator 活动全覆盖、多频域联合控制及整机能耗测量。GPU-only module 合并没有改变模型执行顺序，也不融合 kernel。密集模型的量化不等于有效稀疏跳算。
