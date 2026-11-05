# 常驻优化服务：一次启用 profiling，观察后建模

本入口对应 2026-11-05 的修订计划。目标机由用户手动操作；开发机只完成 CPU 检查。新入口 `python -m rtx5090.service_model` 不创建 vLLM 引擎，不安装依赖，不使用旧 `observe` 的 eager worker，不运行 FUSE 或分块。

当前阶段交付的是 HTTP 请求窗口经验模型工具。实际服务配置、路径清单、模型参数和独立误差必须由目标机新记录产生，仓库没有预填硬件结果。旧 512/30 基线、旧 eager trace 和人工方阵标定均不混入新模型。

## 1. 当前步骤：旧服务运行时预检查

在 **WSL Bash** 的仓库中先核对工作区。以下检查无写频、无服务重启：

```bash
conda activate vllm-qwen
cd /mnt/d/s00958243/SparseDVFS
git branch --show-current
git status --short
```

预期分支为 `5090`。若有用户修改，先保留并处理；不要 reset、clean 或覆盖。确认工作区可快进后：

```bash
git fetch origin
git merge --ff-only origin/5090
PROFILE_DIR="/mnt/d/s00958243/SparseDVFS/runs/service-profile-$(date +%Y%m%d-%H%M%S)"
bash scripts/serve_profiled_5090.sh --check-only "$PROFILE_DIR"
```

预期 `status=preflight_ready_not_worker_verification`，显示 vLLM 0.32.0、独立 Triton 3.6.0 的实际路径和包哈希、Torch/CUDA 版本及 profiler 配置。首先只复制这段精简输出。失败就保留当前服务，复制最后的错误，不升级或切换依赖。

预检查在现有 Conda 中实例化真实 `ProfilerConfig`，检查本地短窗口请求字段、benchmark 数据类型及 profiler 停止语义，保存所用源文件哈希。它不启动 GPU 推理，不代表 worker 的导入已被直接观察。目录自动建立，文件采用独占创建，已存在记录不会覆盖。

## 2. 当前步骤：唯一一次服务重启

仅在上一步通过后，在原服务终端正常停止自己启动的服务，等其退出。在**同一 Conda 环境、WSL Bash** 运行：

```bash
bash scripts/serve_profiled_5090.sh "$PROFILE_DIR"
```

若换终端，先把 `PROFILE_DIR` 设为上一步的实际绝对目录，不要生成另一个时间戳。脚本前台运行，完整日志留在 `$PROFILE_DIR/startup.log`。启动失败停在本步，不自行第二次重启，不添加 eager、dummy、dtype 参数或改变依赖。

脚本保留成功参数：模型 `/mnt/d/models/Qwen3.5-27B`、TP=2、4096、并发 1、显存利用率 0.90、reasoning parser、language-model-only、trust-remote-code；只增加以下 profiler 配置：

```json
{
  "profiler": "torch",
  "torch_profiler_dir": "本次独立绝对目录",
  "ignore_frontend": true,
  "torch_profiler_with_stack": false,
  "torch_profiler_record_shapes": false,
  "torch_profiler_with_memory": false,
  "torch_profiler_with_flops": false,
  "capture_torch_profiler": false,
  "max_iterations": 3
}
```

保留独立 Triton 路径 `/mnt/d/s00958243/triton-360` 和缓存 `/tmp/triton360-vllm-cache`，清除旧三个 NCCL 禁用变量及 `NCCL_DEBUG` / `NCCL_DEBUG_SUBSYS`，设置 `NCCL_CUMEM_HOST_ENABLE=0`。不添加旧 profiler 环境变量，不更改 V1/V2 runner 环境选择，不替换权重。原生按需接口没有启动请求时不会持续采样所有请求。

## 3. 当前步骤：身份验收

在另一 **WSL Bash** 终端，固定以后所有测量也从此 WSL 客户端访问相同地址。设置上一步的 `PROFILE_DIR` 后：

```bash
conda activate vllm-qwen
cd /mnt/d/s00958243/SparseDVFS
curl --noproxy '*' --connect-timeout 5 --max-time 10 \
  --fail --silent --show-error http://10.255.255.254:8000/v1/models
ID_OUT="${PROFILE_DIR}-identity"
python -m rtx5090.service_model inspect --launch-dir "$PROFILE_DIR" --out "$ID_OUT"
```

预期 `identity_ready=true`，两个 `profiler_routes` 均为 true，`profile_overrides_supported=true`。复制命令的摘要即可。需要展开配置时才运行：

```bash
python -m rtx5090.service_model report --file "$ID_OUT/identity.json" --section config
```

身份来自当前 HTTP、启动 PID/启动时刻、实际引擎配置日志和 `nvidia-smi` UUID/型号/驱动。读取实际 dtype、load_format、prefix caching、enforce_eager 和完整编译配置行；没有日志证据就拒绝拟合。`load_format=auto` 只证明自动选择，具体 loader 未直接观察时仍标未知，不能改称 dummy。记录两卡：

| 可见索引 / 标准 TP 启动映射 | UUID |
| --- | --- |
| 0 / 推定 rank 0 | `GPU-22d1cd22-d198-45c4-2b54-974484026ab5` |
| 1 / 推定 rank 1 | `GPU-e2f02606-bac6-52cd-f8a1-15a027db7da0` |

该映射依据启动可见顺序，不冒充 worker RPC 直接验证。记录日志中的 worker PID；runner 具体类型、worker 实际 Triton 导入哈希、collective 通道和 capture 节点映射缺证据时为未知。launch 预检查的 Triton 哈希与实际 worker 导入证据分开。请求窗口模型限于这次服务进程寿命，不推断跨重启可迁移性。`Using network Socket` 不能证明全部 collective 用 Socket；本轮不为补 NCCL 日志再重启。

## 4. 当前步骤：短窗口观察

**WSL Bash**：确保没有其他客户端发送请求。

```bash
OBS_OUT="${PROFILE_DIR}-observe"
python -m rtx5090.service_model observe --identity "$ID_OUT/identity.json" --out "$OBS_OUT"
python -m rtx5090.service_model report --file "$OBS_OUT/observation.json" --section coverage
```

每个输入长度 128/512/2048，先发送 2 个不计入观察的请求，再分别发送输出 64 的 prefill、decode 早段、decode 尾段观察请求。随机合法非特殊 token ID、温度 0、ignore_eos、并发 1；原始 usage 必须确认实际输入长度和输出 64。保留实际 prefix-cache 策略，窗口前后 cache/request counter 检查是否命中或混入外部请求。

使用目标 0.32 原生 `delay_iterations/max_iterations`，当前候选分别是 `(0,1)`、`(2,3)`、`(62,3)`。它们是针对一个 prefill 加 63 个 decode forward 的候选，不是对调度行为的保证；逐 trace 核对实际 `execute_*context_*generation_*` 数量和相位。chunked prefill、speculation、runner 变化或 profiler 边界不符就失败保留记录，不能把错误窗口改标成成功。需要适配时审查本地代码/trace 摘要，服务保持常驻。

每窗明确 stop，待原生导出返回并找到两个 rank 文件后才继续。每个 rank 都解析，即使另一 rank 边界失败也保留两侧摘要。仅 CPU 原生 forward annotation 用作 host launch 边界，支持一次 graph launch 对应多个 GPU kernel；只依赖 correlation ID，歧义或缺失保留 unknown。高 GPU 时间归属率不抵消尾段 forward 没有 GPU 活动的问题。

预期 `observed_sampled_windows`，每窗 `boundary_checked`。这仅表示已采样 forward 边界通过检查，不证明完整请求/完整图的所有活动均已采到。`graph_replay_observed` 来自 trace 的实际 graph launch/GPU correlation；false 只表示本窗口没建立证据。融合体以实际 kernel 名保存，通信 kernel 同样保留原名，不从耗时推定通道或网络带宽。形状、stride、capture 节点没有可靠证据就保留 null/unknown。

只复制一个长度/rank 的主要 kernel 摘要：

```bash
python -m rtx5090.service_model report --file "$OBS_OUT/observation.json" \
  --section kernels --length 128 --rank 0 --top 3
```

原始 trace 永远留在目标机；不需要上传。未启用启动 capture trace，原生 capture 选项只有 rank 0，不能充当双 rank 完整映射。profiler 时间、活动时间和归属比例不能作为正式性能/能量标签。观察失败先处理本步；不要继续堆积新 trace。

## 5. 当前步骤：只读双卡遥测试测

复用已有 Windows NVML bridge 的遥测底层。桥接和 HTTP 是不同地址：服务仍是 `10.255.255.254:8000`，已连通的桥接回环是 `127.0.0.1:50900` 和 `:50901`，不修改网络。

若没有可用桥接，在两个 **Windows 管理员 PowerShell** 终端分别启动。沿用已配置 Python 和 `SPARSEDVFS_BRIDGE_TOKEN`，token 只经环境变量传递，不写命令参数、仓库、日志或聊天。两端应有同一 token；本指南不打印它。不与仍占用该 UUID 的旧 bridge 并行启动。

```powershell
Set-Location D:\s00958243\SparseDVFS
$bridgeLog = "runs\service-bridge0-$(Get-Date -Format yyyyMMdd-HHmmss).jsonl"
python -m rtx5090.bridge --uuid GPU-22d1cd22-d198-45c4-2b54-974484026ab5 `
  --port 50900 --read-only --log $bridgeLog
```

第二个终端：

```powershell
Set-Location D:\s00958243\SparseDVFS
$bridgeLog = "runs\service-bridge1-$(Get-Date -Format yyyyMMdd-HHmmss).jsonl"
python -m rtx5090.bridge --uuid GPU-e2f02606-bac6-52cd-f8a1-15a027db7da0 `
  --port 50901 --read-only --log $bridgeLog
```

`--read-only` 不写频、不需要声明初始自动频率，也不因写频候选枚举不支持而拒绝遥测。NVML 不可用则报告真实错误；不退回 mock。模板没有 token。

在 **WSL Bash** 中沿用同一 token 环境变量：

```bash
TELEM=configs/service_telemetry.template.json
PILOT_OUT="${PROFILE_DIR}-pilot"
python -m rtx5090.service_model pilot --identity "$ID_OUT/identity.json" \
  --telemetry "$TELEM" --requests 20 --out "$PILOT_OUT"
```

预期 `status=ready`。试测只有 512/64、20 个请求，不是训练数据。预热、就绪请求和 profiler stop 在窗口外；请求清单预生成。优先两卡累计能量差值；计数器全程不可用才在本次 pilot 选择功率积分。正式数据固定每卡能量来源，间歇性计数器失败不能静默改用功率。

默认 100 ms 采样，同时保存每字段查询开始/结束、原始值、Windows/WSL 时钟映射区间、真实值变化间隔、采样缺口和温度。每次远程查询用本地请求包围区间约束时钟偏移，按字段映射到客户端单调时钟；不直接相减两台系统时钟。前后均须有样本夹住同一请求窗口，能量与耗时包含同一窗口内的 HTTP/客户端空隙，两卡能量相加。

计数器采用边界插值差值，预算包含两个边界的最大观察增量及查询时间不确定性；功率采用梯形积分，预算包含观察到的值变化间隔和查询时间不确定性。少于 3 次值变化、时间倒退、计数器复位、超过 350 ms 的采样缺口、单卡/总量预算超过 3% 等均拒绝。这里是**基于本轮观察的时间/量化误差预算**，不是传感器硬件精度证书；驱动功率平滑及系统误差仍须结合原始记录核查，无法建立分辨率就不拟合。

若只因窗口太短超预算，可依据 `suggested_requests_if_budget_failed` 在**新 pilot 目录**延长，再冻结每窗请求数。若 token、缓存、频率、计数器或时钟映射有问题，先处理原因，不用延长样本掩盖。功率标签是双卡板级总能量/窗口耗时，不是整机能耗或单算子能量。

## 6. 当前步骤：选择已验收的频率范围

默认 `--clock-mode current` 无写频，只形成当前实际频率策略下的经验模型；仍检查频率、显存频率、温度和实际身份。此前有恢复失败/未验证时必须先解决，不能用 current 模式绕过。

只有准备检验真实锁频时，正常停止自己的只读 bridge，分别用其原命令将 `--read-only` 替换为 **`--apply --ack-default-auto`**，换独立日志文件。这只重启 bridge，不重启 vLLM。`--ack-default-auto` 声明初始状态确认为系统自动频率，不能拿它代替历史恢复验证；工具不恢复任意预存的第三方锁频策略。

**WSL Bash** 双卡验收命令：

```bash
CLOCK_OUT="${PROFILE_DIR}-clock-check"
python -m rtx5090.service_model clock-check --identity "$ID_OUT/identity.json" \
  --telemetry "$TELEM" --apply --ack-default-auto --out "$CLOCK_OUT"
```

仅 `status=passed` 且双卡 2527/2250 MHz 带载读回、正常/注入异常/bridge 断连三类恢复都验证后可进入 fixed。写入前后均保存记录，锁频容差 15 MHz，先等待 0.5 秒，再最多 5 秒以额外不计入测量的请求负载核对连续 3 次读回。这是明确的实现选择，会改变负载历史，不是论文规定。

恢复除了 reset 返回，还要求每卡至少 3 次带载读回高于原锁频上界 15 MHz，并至少有读回落在原带载范围 ±15 MHz。空闲降频不足以证明解锁。不能取得该证据就标未验证并停止；这是保守验收，并非证明恢复不可能。控制顺序 2527 后 2250 有利于最终恢复见证，不据此调模型参数。

单卡 set 失败也尝试恢复所有已经触及的卡；SIGINT/SIGTERM 走清理路径，bridge 连接 EOF 由服务器恢复。强杀整个系统等情况不保证客户端能写出最终记录，须检查 bridge 日志和实际硬件。恢复失败不自动重试测量，不继续写下个频点，不把 reset 返回改写为硬件恢复成功。

## 7. 当前步骤：训练、冻结、独立验证

先完成身份、观察和 pilot 检查。下面为 **WSL Bash / current 模式**，每次确认上一步成功后再执行下一段：

```bash
TRAIN_OUT="${PROFILE_DIR}-train"
python -m rtx5090.service_model collect --identity "$ID_OUT/identity.json" \
  --telemetry "$TELEM" --pilot "$PILOT_OUT/pilot.json" \
  --clock-mode current --split train --seed 11 --out "$TRAIN_OUT"
```

预期 `status=collected`，共 9 窗：3 个长度 × 3 次重复，每窗使用 pilot 冻结的请求数。fixed 模式改为 `--clock-mode fixed --clock-check "$CLOCK_OUT/clock-check.json" --apply`，其余同上，共 18 窗。不会在 current/fixed 之间自动回退，不会使用单卡设置。

正式请求输出均为 64，客户端固定地址和进程内绕代理，实际 usage 校验长度，逐窗检查 prefix-cache 命中及服务成功请求计数。每窗保存预生成请求、回复摘要和原始遥测；训练输入长度按重复轮次轮转，预热/带载稳定请求始终在正式窗口外。正式窗口前关闭 profiler，不挂 Python 算子 hooks，不做逐算子同步，检测到新编译/capture 日志或其它质量异常即保存并停止。

冻结模型：

```bash
FIT_OUT="${PROFILE_DIR}-fit"
python -m rtx5090.service_model fit --data "$TRAIN_OUT/measurements.json" --out "$FIT_OUT"
```

得到 `model.json` 后，才生成不同种子与请求的独立验证集：

```bash
VAL_OUT="${PROFILE_DIR}-validation-data"
python -m rtx5090.service_model collect --identity "$ID_OUT/identity.json" \
  --telemetry "$TELEM" --pilot "$PILOT_OUT/pilot.json" \
  --clock-mode current --split validation --seed 101 \
  --model "$FIT_OUT/model.json" --out "$VAL_OUT"
```

若训练是 fixed，此处也使用完全相同的 fixed 参数。最后验证：

```bash
CHECK_OUT="${PROFILE_DIR}-validation"
python -m rtx5090.service_model validate --data "$VAL_OUT/measurements.json" \
  --model "$FIT_OUT/model.json" --out "$CHECK_OUT"
python -m rtx5090.service_model report --file "$CHECK_OUT/validation.json" --section cells
```

输出参数、各目标总体/分频点/分场景 WAPE、P90 相对误差、最大误差、±10% 比例、样本数和能量时间/量化不确定度。摘要与分场景表分开复制，原始文件留在目标机。退出码 2 表示检查未通过/未就绪，1 为执行错误，130 为中断；失败文件不覆盖、不拼接、不删异常窗口拟合。温度越出训练范围也标适用范围失败，不为达标回填验证集。

## 模型定义和证据界限

每个已验收频率对（或 current 策略）分别拟合 `y = a + b*(L/512)`，a、b 非负。训练窗等权，解两变量 NNLS 的内点与边界候选，目标是窗口平均 TTFT（s）、平均 TPOT（s/token）、窗口总耗时（s）、双卡总能量（J）；平均功率预测由预测能量除以预测耗时得到。TTFT/TPOT 复用已安装 vLLM benchmark 的 HTTP 流式解析和计时语义，额外读取真实 usage，不把 CPU/GPU profiler 时间当标签。TPOT 为 `(HTTP 输出延迟-TTFT)/(64-1)`。

模型保存训练数据摘要、请求/提示词哈希、种子、协议和源文件指纹。验证采集要求模型事先存在，并保存冻结模型哈希；验证不修改模型、不调整参数、不选择新形式。各频点的 4 个主要目标均须 WAPE ≤10%、P90 相对误差 ≤20%，功率单独报告误差。仅 3 窗/场景，不支持长期稳定性或 P99 结论。失败结果也交付，不靠扩大采样隐藏线性形式不足。

适用范围仅为本次实际 dtype/权重/编译图/cache 配置、两张指定 UUID、驱动、常驻服务寿命、WSL 客户端和路径、已测输入 128/512/2048、输出 64、并发 1、温度范围；current 模式还检查验证时频率范围不得超过训练观察范围 ±15 MHz。两频点不拟合任意频率曲线，1024 场景继续封存，未测中间长度也不声称已验证。

执行单元分支保留必要门槛：只以本轮实际融合 kernel/通信/可观测图段为候选；隔离回放尚无证据时不生成适配器。kernel、形状、布局、dtype、编译图环境和状态语义需逐一对齐；Eq.(1) 容量/开销独立标定，通信等待不冒充带宽；缺少电压、温度与可辨识性证据时 Eq.(2) 未复现。请求能量不按耗时比例分摊成“实测算子能量”。

分块仍缺完整图内依赖、合法控制边界、调频切换/拆图/融合变化开销，以及改变边界后的重验证。请求级模型通过不等于 SparseDVFS 算子模型完成，也不能直接决定图内分块。

## 代码与验证入口

- `service_client.py`：原生配置预检查、身份、代理隔离、token 与 benchmark 接口。
- `service_core.py`：trace 关联、数据契约、非负拟合、冻结与独立验证。
- `service_telemetry.py`：双 UUID 查询、跨时钟映射、能量预算与恢复判定。
- `service_model.py`：小批量 CLI 编排、原始记录、文本报告。
- `validation/service_model/REPORT_zh.md`：本机 CPU 验证和未完成的目标机验收。

原生接口依据：[v0.32 profiler 配置](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/config/profiler.py)、[按需 API](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/entrypoints/serve/profile/api_router.py)、[profiler wrapper](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/profiler/wrapper.py)、[worker 边界](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/worker/gpu_worker.py)。目标机预检查仍以本地实际安装源码为准。
