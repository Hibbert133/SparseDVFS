# 第一阶段：单张 RTX 5090 的 Eq. (1) 性能模型

本阶段只做“算子测时 → 参数估计 → 公式预测 → 实测核对”。不加载 Qwen，不需要真实模型权重、vLLM worker 或 FUSE 动态库，不做功耗拟合、分块、动态调频或收益对照。固定频率只用于离线标定。

2026-11-03 目标机用户反馈：`doctor` 已 ready，2250 MHz 的 12 组完成；2527 MHz 首组在正式测时前读回 2407 MHz，采集失败，已请求 reset 且调用返回成功，物理恢复未验证。当前尚无完整标定或预测结论。下面的 `doctor → collect → fit → validate` 均为已实现入口；此次失败的诊断与重试见 [读回诊断说明](EQ1_CLOCK_DIAGNOSTICS_zh.md)。

## 依据与实现选择

- 原文：[SparseDVFS v1 §4.1.1 Eq. (1)](https://arxiv.org/html/2603.21908v1)。直接调用 `sparsedvfs.paper_core.timeline_seconds()`，不改公式。
- 补充来源：[作者代码 c524c51 的 profile_model](https://github.com/ziyangzhang96/SparseDVFS/blob/c524c51aac552fd16bec5743b1fb95309b46347b/SparseDVFS.py#L536)。参考设置频率后等待 0.5 秒、预热 5 次、测量 5 次取均值，以及 `perf_counter()` 配合同步的测时方式。
- 作者公开脚本主要是逐层 profiling，没有提供 Eq. (1) 的容量、带宽、固定开销估计器。下述 BF16、算子形状、静态数据量、独立 CUDA Event 采样及常量估计方法均是本阶段明确的实现选择，不是作者给出的参数。
- 这是单 GPU 核心频率的一维核对。CPU、显存策略保持不变；正式采样要求显存读回同档。有效稀疏度 `S_comp=S_mem=0`，不验证跳零机制。

| 量 | 本工具的定义 |
| --- | --- |
| GEMM 计算量 | `2*M*N*K` FLOP |
| GEMM 数据量 | `2*(M*K+K*N+M*N)` byte，BF16 两个输入与一个输出的逻辑字节量 |
| 加法计算量 | 元素数 FLOP |
| 加法数据量 | `3*元素数*2` byte，两个输入读、一个输出写 |
| P_peak(f) | 各频点 8192 方阵 GEMM 的 FLOP / CUDA Event 平均秒数，属于有效容量估计 |
| B_mem | 2527 MHz 下，256 MiB/缓冲区加法的逻辑字节量 / CUDA Event 平均秒数；所有核心频点共用一个值 |
| T_overhead | 标定形状的“墙钟均值减 Roofline 项”残差的中位数，下限为 0；全部算子和频点共用一个常量 |

逻辑字节量不等于实际 DRAM 流量，有效 GEMM 容量也不等于所有算子的理论峰值。本阶段不采集硬件 DRAM 计数，不进行功率或电压建模。使用预分配输入和输出，分配、输入生成、遥测与文件写入都在计时区间之外。

## 1．通过代码仓更新，释放实验卡

代码通过 `https://github.com/Hibbert133/SparseDVFS` 的 `5090` 分支管理：开发侧提交并推送，目标机器 fetch 后快进更新。同一份 Windows/WSL 共享仓库只需要更新一次。

先在自己的服务终端停止 vLLM；如果旧 Windows bridge 仍运行，也先在它的终端按 Ctrl+C，后面重新启动以加载新增的诊断字段。

在 **WSL Bash** 执行：

```bash
cd /mnt/d/s00958243/SparseDVFS
git status --short
git fetch origin
git switch 5090
git merge --ff-only origin/5090
git log -1 --oneline
```

`git fetch` 只更新远端引用，后面的 `git merge --ff-only origin/5090` 才更新当前代码。遇到本地修改冲突或分支分叉时停止，保留现场，不使用强制覆盖。此次更新补足失败读回、阶段和诊断字段，并新增显式可选的有界检查，无需编译 CUDA/FUSE 或重新安装 Python 包。

可选的纯 CPU 入口检查：

```bash
python -B -m rtx5090.performance --help
```

预期看到 `doctor, collect, fit, validate`。旧的 `rtx5090.conda calibrate` 测的是切频开销，不是本阶段入口。

## 2．启动 Windows bridge

### 2.1 网络前提：WSL 镜像网络

当前 bridge 固定监听 Windows 的 `127.0.0.1:50900`，不支持 `--host`。WSL 需使用镜像网络才能按本文地址访问它；不能只把客户端地址改成 NAT 网关。参见[微软镜像网络说明](https://learn.microsoft.com/windows/wsl/networking#mirrored-mode-networking)。

2026-11-03 目标机用户反馈：WSL 2.6.0.0、Windows build 26100.6584，最初无 `.wslconfig`；Windows 已监听但 WSL 报 `Connection refused`。启用镜像网络并重启后，TCP 连接及认证 `doctor` 均已通过，不需要重复改网络。这不代表两个频点的完整标定或物理恢复通过。

仅在连接检查失败、确认需要切换网络时执行。此操作改变该 Windows 用户下所有 WSL2 发行版的网络模式；`wsl --shutdown` 会停止所有 WSL 实例。先保存工作并正常停止自己的任务，不中断其他用户任务。已启用镜像网络的机器无需重复配置。

在 **Windows PowerShell** 检查版本和现有配置：

```powershell
wsl --version
$cfg = Join-Path $env:USERPROFILE ".wslconfig"
if (Test-Path -LiteralPath $cfg) { Get-Content -LiteralPath $cfg }
```

确认没有配置文件时，执行以下命令；如已有文件则停止，保留原配置，人工合并 `[wsl2]` 下的 `networkingMode=mirrored`，不覆盖其他设置。

```powershell
if (Test-Path -LiteralPath $cfg) { throw ".wslconfig 已存在，停止以避免覆盖" }
Set-Content -LiteralPath $cfg -Encoding ascii -Value "[wsl2]`nnetworkingMode=mirrored"
Get-Content -LiteralPath $cfg
wsl --shutdown
wsl
```

如需回退，只移除本次新增的设置，保留其他配置，并在保存工作后再次重启 WSL。重启后重新激活 Conda、核实持久目录 `/mnt/d/s00958243/triton-360`，并重设下面的环境变量、GPU UUID、输出目录与 token。此前 `/tmp/triton-360-test` 已因重启丢失，用户已改用持久目录。Windows bridge 进程不因 WSL 重启而自动重启，仍使用原 token。

### 2.2 在 Windows 管理员窗口启动服务

在 **Windows 管理员 PowerShell** 执行：

```powershell
cd D:\s00958243\SparseDVFS
Test-Path .\rtx5090\bridge.py
```

输出应为 `True`，否则先检查目录和分支。然后在同一窗口生成新 token：

```powershell
$env:SPARSEDVFS_BRIDGE_TOKEN = python -c "import secrets; print(secrets.token_hex(32))"
$env:SPARSEDVFS_BRIDGE_TOKEN
```

仅在自己的终端之间复制 token，不发送到聊天或写入配置文件。继续启动：

```powershell
python -B -m rtx5090.bridge --uuid GPU-22d1cd22-d198-45c4-2b54-974484026ab5 --port 50900 --ack-default-auto --apply --log bridge-eq1.jsonl
```

保持窗口运行。启动桥接只允许后续经过认证的写频请求，不会立即锁频。第一张卡启动前应处于默认自动频率管理状态。

### 2.3 先检查端口可达性

在 **WSL Bash** 执行，无需 token，不写频率：

```bash
python3 - <<'PY'
import socket
try:
    with socket.create_connection(("127.0.0.1", 50900), timeout=3):
        print("连接成功")
except OSError as e:
    print(f"连接失败：{e}")
PY
```

预期为 `连接成功`。若失败，在另一个 **Windows PowerShell** 检查：

```powershell
Get-NetTCPConnection -LocalPort 50900 -State Listen |
    Select-Object LocalAddress, LocalPort, OwningProcess
```

无监听时检查 bridge 启动报错；Windows 有监听而 WSL 不通时先检查镜像网络是否生效。不要反复更换 token，也不要直接关闭防火墙。TCP 检查成功后再运行 doctor，验证认证、设备与环境。

## 3．在 WSL 检查环境和连接

新开 **WSL Bash**，下面各步骤在同一个终端继续执行：

```bash
conda activate vllm-qwen
cd /mnt/d/s00958243/SparseDVFS

export PYTHONPATH="/mnt/d/s00958243/triton-360:${PYTHONPATH:-}"
export TRITON_CACHE_DIR=/mnt/d/s00958243/triton360-cache
export SPARSEDVFS_TRITON_PATH=/mnt/d/s00958243/triton-360

PERF_GPU_UUID=GPU-22d1cd22-d198-45c4-2b54-974484026ab5
PERF_OUT="$PWD/runs/eq1-$(date +%Y%m%d-%H%M%S)"

read -rsp '粘贴本机 bridge token: ' SPARSEDVFS_BRIDGE_TOKEN
echo
export SPARSEDVFS_BRIDGE_TOKEN

python -B -m rtx5090.performance doctor --gpu-uuid "$PERF_GPU_UUID" --backend bridge --ack-default-auto --out "$PERF_OUT"
```

预期输出 `status: ready`，UUID 与第一张卡一致，`idle_readback` 有核心和显存频率。详细环境与源文件哈希写入 `$PERF_OUT/doctor.json`，不保存 token。

`doctor` 不设置或恢复频率，仅创建所选卡的 CUDA 上下文并读取信息。空闲时的显存档位可以与后面的负载态不同；`collect` 会在预热后建立自己的正式采样显存基准。

遇到错误停在当前步骤：

- `Memory clock unavailable`：先确认 Windows 与 WSL 访问同一份已打补丁的仓库，并重启旧桥接进程；若驱动仍不提供读回，保留错误，不用猜测值代替。
- `Connection refused`：连接尚未建立，发生在 token 验证之前；按 2.3 检查监听和网络。超时也先检查连接路径，不自动修改防火墙或驱动。
- 认证失败：只在自己的 Windows/WSL 终端间核对同一个 token；Windows 启动 bridge 后更改环境变量不会更新已运行进程的 token。
- `Evidence already exists`：失败的 doctor 也会保留 `doctor.json`，不要删除或覆盖。修复当前错误后执行 `PERF_OUT="$PWD/runs/eq1-$(date +%Y%m%d-%H%M%S)"`，等待至少一秒避免同名，再重试 doctor；成功后 collect/fit/validate 沿用该目录。
- CUDA/UUID/Triton 路径错误：核对当前 Conda 及上述持久目录，检查实际导入路径；不覆盖已安装环境或重用旧临时路径。

## 4．采集两个固定频率下的数据

确认实验卡空闲后，在原 **WSL Bash** 继续：

```bash
python -B -m rtx5090.performance collect --gpu-uuid "$PERF_GPU_UUID" --backend bridge --rates-mhz 2250 2527 --warmup 5 --repeat 5 --ack-default-auto --apply --out "$PERF_OUT"
```

这是本阶段实际写入 GPU 核心频率的步骤。每个频点设置后等待 0.5 秒，每个算子执行两次独立测时流程：

1. 预热 5 次；测 5 次同步墙钟时间并保留全部样本。
2. 再预热 5 次；独立测 5 次 CUDA Event 时间，转为秒后保留全部样本。

每个流程前后读取核心、显存频率和温度；核心偏差超过 15 MHz、显存改变或读回缺失时停止。15 MHz 是本阶段的读回容差，不是论文参数。

默认仍是每个阶段一次严格读回。失败快照现在也保存到 `snapshots[阶段]` 和 `clock_observations`，包括原始频率、阶段、查询往返耗时及可用的只读诊断。若需评估额外稳定性检查，可以显式添加 `--clock-stability-timeout-s 5`：仅在两次正式测时**之前**，要求连续 3 次读回满足原容差；重查前执行一次同形状算子并同步，重查间隔 0.1 秒。额外操作不计入 5 次预热或 5 次正式样本，但会改变负载历史，因此记录为相对作者流程的实现差异。测时后的检查仍一次失败即停止；详见 [读回诊断说明](EQ1_CLOCK_DIAGNOSTICS_zh.md)。

| 用途 | GEMM 方阵边长 | 加法每个缓冲区 |
| --- | --- | --- |
| 容量估计 | 8192 | 256 MiB |
| 常量开销标定 | 1024、2048、4096 | 8、64 MiB |
| 预测核对，不参与拟合 | 1536、3072、6144 | 16、128 MiB |

命令会逐组显示进度，正常结束为 **24 条完整记录**、`status: collected`、`reset_returned: true`。原始数据保存在 `$PERF_OUT/measurements.json`；部分完成或失败的数据也保留在这里，但不能进入拟合。

`reset_returned` 只表示驱动恢复调用返回成功，`physical_restore_verified` 仍为 `false`；不据此证明物理 PLL 稳定或恢复成某个固定 MHz。

## 5．估计参数并核对预测

```bash
python -B -m rtx5090.performance fit --out "$PERF_OUT"
python -B -m rtx5090.performance validate --out "$PERF_OUT"
```

`fit` 写入 `$PERF_OUT/model.json`。只用容量估计和标定形状决定参数：两个频点各一个 P_peak、一个共享 B_mem、一个共享 T_overhead。标定用墙钟残差，容量用独立 CUDA Event 数据；不把两种时间混作同一个测量值。

`validate` 不改模型，输出 10 行核对结果，写入 `$PERF_OUT/validation.json` 和 `predictions.csv`。表中包含计算项、访存项、固定开销、预测时间、墙钟实测时间、辅助 CUDA 时间和相对误差。

报告统计平均/最大相对误差及 ±10% 内比例。`status: evaluated` 表示完成了数据核对，不表示已达到论文精度、复现节能或完整三域机制。低核心频率下测到的带宽只作为诊断项，不会自动变成 `B_mem(f_gpu)` 的自由参数。

`fit/validate` 不连接 GPU、不导入 torch/vLLM，也不要求功率、电压或权重。离线复制数据时需使用相同源文件版本；源哈希变化会明确拒绝继续。

## 6．失败处理与本阶段交接

- 任一步报错先停下，不运行后续命令。保留整个 `$PERF_OUT`，不要删掉不利样本或覆盖既有证据；重采使用新的时间戳目录。
- 显存自动换档会使本轮数据不满足共享带宽前提。保留失败记录，再分析稳定性；工具不自动锁显存、改 CPU governor 或放宽检查。
- `GPU reset failed` 或桥接断开：停止后续实验，在另一个 Windows 管理员 PowerShell 必要时执行：

```powershell
nvidia-smi -i GPU-22d1cd22-d198-45c4-2b54-974484026ab5 -rgc
```

- Ctrl+C/SIGTERM 会进入恢复路径；SIGKILL、断电或系统崩溃不能依靠进程内清理保证恢复，需要手动核实。
- 收集后反馈 `doctor.json`、`measurements.json`、`model.json` 和 `validation.json` 中的结果或报错即可，检查不要包含自行追加的密钥。
- 本阶段结论只覆盖单卡、BF16、显式逻辑流量及声明测时口径。实际 DRAM、非零有效稀疏度、CPU/EMC 三域、Qwen/TP=2 均未在本阶段验收。

CPU 检查命令：

```bash
python -B -m unittest discover -s tests -p test_performance_eq1.py -v
python -B -m unittest discover -s tests -p test_v4_paper.py -k eq1 -k test_peak_cannot_depend_on_emc -k test_bandwidth_cannot_depend_on_cpu_gpu -v
```

这些使用合成数据、时钟后端替身和测时 API 替身验证数学、数据隔离与恢复流程，不证明实际 CUDA、锁频能力或真实预测误差。
