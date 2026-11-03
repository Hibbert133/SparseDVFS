# Eq. (1)：2527 MHz 读回失败的诊断与重采

截至 2026-11-03，目标机用户确认 `performance doctor` ready、bridge 认证及 UUID 正确。2250 MHz 的 12 组完成；2527 MHz 的 `gemm_8192` 在预热 5 次并同步后的 `before_wall` 读回 2407 MHz，正式测时尚未开始。该目录保持失败状态，不拟合、不拼接新采样。

当前证据不能区分锁频未达到、同步后的查询间隙降频或遥测更新延迟。旧运行的失败 `snapshots` 为空，不能事后补造。退出后的 2512 MHz、37℃、约 67 W、600 W 上限以及限频原因均未激活，不能反推采样时状态。累计 SW Power Capping 的 1317512 us 没有前后差值，也不能归因于这次失败。

## 先更新，保持默认流程重采

在旧 Windows bridge 窗口 Ctrl+C。若其报告恢复失败，先按主指南手动恢复。然后在 **WSL Bash** 更新共享仓库：

```bash
cd /mnt/d/s00958243/SparseDVFS
git status --short
git fetch origin
git switch 5090
git merge --ff-only origin/5090
git log -1 --oneline
```

冲突或分叉时停下，不覆盖用户修改。按 [主指南第 2.2 节](PERFORMANCE_EQ1_zh.md#22-在-windows-管理员窗口启动服务)，在 **Windows 管理员 PowerShell** 中进入仓库、生成本机新 token 并重启 bridge。必须重启才能加载新增 NVML 诊断；无需再改镜像网络。

在 **WSL Bash** 设置已确认的持久 Triton 路径，创建全新目录：

```bash
conda activate vllm-qwen
cd /mnt/d/s00958243/SparseDVFS
export PYTHONPATH="/mnt/d/s00958243/triton-360:${PYTHONPATH:-}"
export SPARSEDVFS_TRITON_PATH=/mnt/d/s00958243/triton-360
export TRITON_CACHE_DIR=/mnt/d/s00958243/triton360-cache
PERF_GPU_UUID=GPU-22d1cd22-d198-45c4-2b54-974484026ab5
PERF_OUT="$PWD/runs/eq1-diagnostic-$(date +%Y%m%d-%H%M%S)"
read -rsp '粘贴本机 bridge token: ' SPARSEDVFS_BRIDGE_TOKEN
echo
export SPARSEDVFS_BRIDGE_TOKEN
python -B -m rtx5090.performance doctor --gpu-uuid "$PERF_GPU_UUID" --backend bridge --ack-default-auto --out "$PERF_OUT"
```

确认 `status: ready` 后，继续在同一终端运行默认采集：

```bash
python -B -m rtx5090.performance collect --gpu-uuid "$PERF_GPU_UUID" --backend bridge --rates-mhz 2250 2527 --warmup 5 --repeat 5 --ack-default-auto --apply --out "$PERF_OUT"
```

默认行为仍是设频后固定等 0.5 秒，每个独立测时流程预热 5 次、测量 5 次。四个检查阶段都只读一次，不自动换频点或放宽 15 MHz。

## 新记录怎样解释

`measurements.json` 的每组保留 `snapshots` 中各阶段的最后一次读回，以及 `clock_observations` 中的全部尝试。即使读回不合格，也先附到该组，再使检查失败；外层恢复路径把失败记录写入文件。`clock_checks` 记录各阶段整体通过/失败、耗时及额外操作数。

快照的 `passed` 仅表示这一次身份/频率比较通过；阶段是否通过还要看 `within_deadline` 和 `clock_checks[阶段].status`。本次把 Graphics 查询放到遥测序列第一项，旧版是在能量/功率/温度/显存之后查询；查询顺序和新增字段都会改变检查间隙，属于诊断实现变化，不能据此认定根因或硬件问题已消失。

| 字段 | 用途与限制 |
| --- | --- |
| `failure_context` | 当前频点、算子、阶段、样本索引、已完成样本数、最近读回 |
| `active_phase` | 分配、预热、before/after_wall、墙钟采样、before/after_cuda、Event 采样等；采样索引从 0 开始 |
| `clock_mhz` / `sm_clock_mhz` / `memory_clock_mhz` | Graphics 是原检查口径；SM 和显存读回辅助解释，不替换原检查 |
| `client_last_sync_ns`、`sync_to_query_start_s` | 同一 WSL 进程里，同步返回至查询发起的间隔；不是 GPU 内部空闲时间 |
| `query_roundtrip_s` | 客户端整次遥测请求耗时，含 bridge 排队/网络/所有字段查询 |
| `timestamp_ns`、`field_query_times_ns` | 后端单调时钟与各字段查询起止时间；只能在后端内部比较，不与 WSL 时间戳相减 |
| `clock_event_reasons` / `clock_event_reasons_api` | 查询时活跃限频原因的原始位掩码及实际 API 名；缺失为 null 并保留错误，不能当作 0 |
| `performance_state`、`power_mw`、`power_limit_mw`、温度、利用率 | 同次请求内的只读诊断，字段顺序查询，不能当作原子快照；不进行功耗拟合 |
| `rate_transitions` | 写频调用返回及固定等待的客户端起止时间；写入返回不证明稳频 |
| `restoration` | reset 是否尝试/返回；`physical_restore_verified: false` 仍不能解释成物理恢复通过 |

当前限频位掩码含义依据 [NVIDIA NVML 文档](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlClocksEventReasons.html)。优先调用 `nvmlDeviceGetCurrentClocksEventReasons`，不可用时尝试旧 `...ClocksThrottleReasons`，保留原错误；参见 [查询接口](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html)。这些都是查询时状态，不是历史因果证明。相同的重复读数也不能证明驱动遥测已刷新。

若再次失败，在 **WSL Bash** 提取诊断，不启动下一步：

```bash
python - "$PERF_OUT/measurements.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
print(json.dumps({k: d.get(k) for k in
    ("status", "error", "failure_context", "restoration")}, ensure_ascii=False, indent=2))
if d.get("records"):
    print(json.dumps(d["records"][-1], ensure_ascii=False, indent=2))
PY
```

## 可选：有超时的测时前一致性检查

建议先取得上述默认流程的诊断。若需要检查短暂不匹配能否消退，使用**另一个新目录**先重跑 doctor，再对 collect 显式加 `--clock-stability-timeout-s 5`。这不是自动开启的修复，也不表示已经证实根因。

检查规则：

- 保留设频后的 0.5 秒等待，以及每个流程的 5 次预热和 5 个正式样本。
- 仅 `before_wall`、`before_cuda` 使用每阶段 5 秒预算，要求连续 3 次 Graphics 读回在请求值 ±15 MHz 内。初次读回也计入连续次数。
- 后续每次重查前先等待 0.1 秒，再执行一次**同形状、同缓冲区算子并同步**。额外操作及查询均在正式测时外，次数和失败读数全部保留。
- 不匹配会把连续次数归零。显存变化、UUID/查询错误立即失败，不做重试。
- 超时后不再发起新探测。单次已经进入的 CUDA/NVML/socket 调用不能由此预算强制中断，所以实际耗时可能越过预算；越界返回不能使检查通过。
- `after_wall`、`after_cuda` 保持单次严格检查。即使随后频率达标，也不能挽救已完成的测时样本。

此模式写入 `protocol.clock_stability.author_flow_deviation: true`。连续次数、0.1 秒间隔、预算与额外算子均为本阶段移植选择，作者公开代码只有固定等待。这些操作改变了负载历史，须在最终结果中说明；默认与可选模式的采样不能拼接成一份模型。

稳定性检查通过只代表这些读回一致，不能证明整个测时区间持续稳频或物理 PLL 已稳定。若仍失败，保留证据继续定位，不自动换频点、放宽容差或改功率上限。仅完整 `status: collected` 且 reset 返回成功的数据可进入 fit/validate；当前旧失败目录始终不拟合。
