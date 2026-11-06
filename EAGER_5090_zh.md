# 5090_eager：2026-11-05 首批实现与目标机验收

这是独立 eager 分支。`5090` 的优化常驻服务配置保持原样；本指南不授权自动停止服务。
目标是论文全部机制，但本次交付**不是已完成的全机制复现**。

## 当前已经实现与仍缺少的内容

| 部分 | 本分支软件 | 实机状态/限制 |
| --- | --- | --- |
| BF16/TP2 eager | 独立启动、全分片头/长度检查、双 worker 身份、实际 compile/graph 配置 | 尚未在目标机加载验收；文件检查不是权重载荷校验或质量评测 |
| 观察 | 真实 worker 模块顺序、实际调度、可选 dispatcher/kernel trace | 模块链不是完整 kernel DAG；Event 时间不是无扰动标定标签 |
| 双卡硬件 | 复用 bridge/DualClocks、正常/异常/断连恢复、切换上界测量 | 尚未实机验收；现有100ms工具不证明微秒/毫秒精度 |
| Eq.(1)/(2) | 同路径数据输入契约、共享板级热系数拟合、独立验证、温度约束 | **未实现通用隔离回放采集器**：须先检查本轮 trace，再为已确认 kernel/状态实现适配；没有电压/独立功率标签时不能标定 |
| Algorithm 1 | 复用原OR/max/末尾追加语义，扩展显式双GPU频率向量 | 是GPU域投影、模块链；不是CPU/GPU/EMC齐全的论文执行图 |
| 边界控制 | 两rank独立Unix通道会合、全设备完成、双卡设置/读回、再提交 | Python/主机协调的首版；不是已完成的原论文C++ FUSE接入 |
| look-ahead | 有界一块提前队列、取消与等待、实测门槛 | 无匹配串行验收及细粒度切换证据时拒绝启用；没有宣称隐藏开销 |
| 比较 | AUTO/两固定频率训练验证；AUTO/冻结最佳固定/冻结动态策略比较 | 未在目标机运行；双卡板级总能量，不是整机 |
| CPU/EMC/有效稀疏 | 原数学接口保留；能力报告明确缺项 | 当前Windows/WSL后端无CPU/EMC真实控制；Qwen有效跳零未证实；完整复现仍未完成 |

原文：[Eq.(1)/(2)、Algorithm 1、FUSE与look-ahead](https://arxiv.org/html/2603.21908v1)。
结构化规格见 `validation/eager/spec.yaml`。不得将下面的经验窗口模型当成论文算子模型。

## 0. 第一批：只读检查，服务继续运行

**WSL Bash**：

```bash
cd /mnt/d/s00958243/SparseDVFS
git status --short --branch
git log -1 --oneline
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version,clocks.current.graphics,clocks.current.memory --format=csv
PYTHONPATH="/mnt/d/s00958243/triton-360:${PYTHONPATH:-}" \
/home/s00958243/miniforge3/envs/vllm-qwen/bin/python -B - <<'PY'
import sys, torch, triton, importlib.metadata as m
print('python:', sys.version.split()[0], sys.executable)
for n in ('vllm', 'torch', 'transformers'):
    print(n, m.version(n))
print('cuda:', torch.version.cuda)
print('triton:', triton.__version__, triton.__file__)
PY
curl --noproxy '*' --connect-timeout 5 --max-time 10 \
  --fail --silent --show-error http://10.255.255.254:8000/v1/models
```

预期：两卡UUID正确、现有服务正常、独立Triton 3.6.0。失败：保留输出，先处理当前步骤，不重装或重启。

## 1. 更新与无模型预检查

确认目标工作区无需保留的未提交改动后再执行。若有改动，先审阅处理，不用强制切换/reset。
首次创建本地分支（已存在则用 `git switch 5090_eager`）：

```bash
git fetch origin
git switch --track -c 5090_eager origin/5090_eager
git merge --ff-only origin/5090_eager
conda activate vllm-qwen
export EAGER_RUN="/mnt/d/s00958243/SparseDVFS/runs/eager-$(date +%Y%m%d-%H%M%S)"
bash scripts/serve_eager_5090.sh --check-only "$EAGER_RUN"
```

预期 `preflight_ready_not_loaded_or_clock_verified`。检查保留成功NCCL设置、真实模型分片、安装源码哈希、Triton与CUDA12.9；不构造LLM，不写频。
该命令仅写新报告目录。CPU/EMC能力不可用、native链接未验收会明确报告，不伪造通过。
预检查失败时旧服务保持运行。每次新的检查使用新目录；不要删除失败目录。

## 2. 硬件初验：在旧服务上先完成

先复用 `SERVICE_MODEL_zh.md` 的 `inspect`、`pilot`、`clock-check --apply --ack-default-auto`。
其 `--identity` 必须来自仍在运行的优化服务及其实际 launch 目录；不能拿eager身份冒充旧服务，也不能把旧gate直接给eager。

Windows bridge仍是**每UUID一个进程**，端口50900和50901；配置复用 `configs/service_telemetry.template.json`。
只有用户确认当前确为AUTO、现有bridge状态及端口后，才在Windows管理员PowerShell使用既有Python启动：

```powershell
python -B -m rtx5090.bridge --uuid GPU-22d1cd22-d198-45c4-2b54-974484026ab5 --port 50900 --apply --ack-default-auto --log "<本次独立目录>\gpu0-bridge.jsonl"
```

另一窗口对应第二UUID和50901。上面目录占位符须换成实际已创建目录，不能原样执行。
token沿用既有环境传递方法，不写命令参数/JSON/日志/聊天。不自动终止既有bridge，不另装Python。
真实写频命令必须显式 `--apply`；查询通不代表写频或恢复通过。
正常、受控异常、断连后的两卡带载恢复有任一未通过，停止后续调频。

## 3. 唯一一次 eager 切换

仅在预检查和硬件初验通过后，由用户正常停止自己启动的服务。保留原优化启动命令作为恢复配置，但不要自动重启。
在保留 `EAGER_RUN` 的WSL终端：

```bash
bash scripts/serve_eager_5090.sh "$EAGER_RUN"
```

显式BF16、auto、TP2、eager、全部64层；保持prefix cache、chunked prefill及成功配置的默认值。
启动器选择UUID而非历史索引，保留Triton 3.6.0和 `NCCL_CUMEM_HOST_ENABLE=0`。
不启用旧 `rtx5090.observe collect`（它仍是历史dummy协议）。
观察与控制入口一次装入，默认关闭，之后复用常驻进程。

另一WSL终端激活相同Conda，手动设为同一个实际目录，然后：

```bash
python -B -m rtx5090.eager inspect --run "$EAGER_RUN" --out "$EAGER_RUN/identity.json"
python -B -m rtx5090.eager observe --identity "$EAGER_RUN/identity.json" \
  --seed 110501 --out "$EAGER_RUN/units-512"
```

预期双rank真实身份一致、prefill与63个decode上下文、每步完整执行单元顺序。失败即保留两rank日志；不要自动再加载模型。
通过后另选新目录执行一次带 `--trace` 的observe，取得真实kernel/shape/stride清单，作为回放适配依据；其耗时不进入性能拟合。

2026-11-05 目标机确认 vLLM 0.32 的 `/reset_prefix_cache` 仅在 `VLLM_SERVER_DEV_MODE` 下注册。
`observe` 在此接口缺失时检查当前服务 OpenAPI 的 completion 请求是否声明字符串 `cache_salt`；
支持才为每次观察生成独立盐并保存到 `request.json`，不改变 prompt token、prefix-cache 配置或开启开发接口。
盐仅作为隔离手段，不直接证明未命中：仍要求两个 rank 的调度记录从0开始、完整512-token prefill，
以及63个decode上下文全部通过。失败不标为已观察完整请求。客户端提交及源码哈希另记，保留常驻worker原身份。
该适配仅用于执行观察；正式配对窗口的清缓存要求没有放宽。93项相关CPU检查通过，不代表目标机观察已通过。

## 4. eager 本轮硬件验收与最早可得实测结果

### 当前频率批量参考（无需写频、无需重启）

`python -B -m rtx5090.eager_baseline --identity <现有eager身份> --telemetry configs/service_telemetry.template.json --requests 20 --seed 1106000 --out <新目录>`

2026-11-06新增：一次20请求pilot，随后3个reference窗及3个不同请求的holdout窗。
全部固定512输入、64输出、并发1，窗口内关闭profiler及详细观察；核实同一常驻双rank身份。
独立缓存盐只负责请求隔离，真实metrics必须验证无cache命中且无额外请求；不因此放松正式配对比较的门槛。
pilot通过后冻结累计能量/功率来源、显存频率参考和请求数，不自动扩窗或重试。
每窗写原始遥测、请求、回复与摘要，最终写 `baseline.json` 和 `REPORT.txt`；失败也保留已完成窗口。
它没有写频或检查锁频状态，所以条件名为 `current_policy_not_verified_AUTO`，不能称为正式AUTO验收。
6个独立窗口是重复性参考，未拟合模型，不把holdout名称当作模型已通过验证；不能用于宣称Eq1/Eq2或节能收益。
当前97项相关CPU检查通过，目标机批量能量结果待运行。

另：rank0 probe实机已匹配四个kernel名称集合，用户离线核对8192次相应算子的kernel次数全部一致。
实际缓存、顺序和rank1仍未通过。该次CUDA可用显存报告30535 MiB，与NVML的2755 MiB冲突；
后续probe在CUDA初始化后同时查询NVML，取两者较小值进行1024 MiB余量检查，不将CUDA读数当作物理显存已释放。

`TEL` 是审核过的实际双卡遥测JSON（模板没有凭据）：

```bash
export TEL=/mnt/d/s00958243/SparseDVFS/configs/service_telemetry.template.json
python -B -m rtx5090.eager clock-check --identity "$EAGER_RUN/identity.json" \
  --telemetry "$TEL" --apply --ack-default-auto --out "$EAGER_RUN/clocks"
python -B -m rtx5090.eager pilot --identity "$EAGER_RUN/identity.json" \
  --telemetry "$TEL" --requests 20 --out "$EAGER_RUN/pilot"
```

每条通过后才执行下一条。pilot不通过3%时间/量化预算时，按报告增大requests并换新目录试采；冻结窗口后不再改协议。
`clock-check.json` 必须 `passed` 且三个恢复案例均有两卡证据。

```bash
python -B -m rtx5090.eager collect --identity "$EAGER_RUN/identity.json" \
  --telemetry "$TEL" --gate "$EAGER_RUN/clocks/clock-check.json" \
  --pilot "$EAGER_RUN/pilot/pilot.json" --split train --seed 111000 --apply \
  --out "$EAGER_RUN/train"
python -B -m rtx5090.eager fit --data "$EAGER_RUN/train/dataset.json" \
  --out "$EAGER_RUN/window-model.json"
python -B -m rtx5090.eager collect --identity "$EAGER_RUN/identity.json" \
  --telemetry "$TEL" --gate "$EAGER_RUN/clocks/clock-check.json" \
  --pilot "$EAGER_RUN/pilot/pilot.json" --split validation --seed 211000 --apply \
  --model "$EAGER_RUN/window-model.json" --out "$EAGER_RUN/validation"
python -B -m rtx5090.eager validate --data "$EAGER_RUN/validation/dataset.json" \
  --model "$EAGER_RUN/window-model.json" --out "$EAGER_RUN/window-validation.json"
```

每条件3窗、同token清单配对、轮换条件顺序，prefix cache通过受支持接口在窗口前重置并通过metrics核实miss。
失败窗口保留并停止，不丢样本继续拟合。固定频率选择按训练窗口耗时≤AUTO×1.05后最低能量冻结。
这一步产出真实eager延迟/双卡能量及频率空间结论，**不是**SparseDVFS算子模型或动态策略收益。

## 5. 论文建模与分块门槛

### 2026-11-06 单卡矩阵kernel名称诊断

`python -B -m rtx5090.matrix_probe --identity <eager身份文件> --trace-dir <完整trace目录> --rank 0 --out <新目录>`
读取该rank最近生成的 `matrix-signatures` 文件，验证摘要哈希与四个已观察的MLP线性算子契约。
服务保持运行，但执行期间不要发其他推理/实验请求。工具持有实验客户端锁，并在前后核实常驻服务身份。
按UUID选择单卡；创建CUDA上下文前需1536 MiB空闲，之后需1024 MiB，未满足则停止。
该余量是本诊断的保守实现选择，不是论文参数或显存峰值保证。GPU1此前仅221 MiB可用，不能直接运行。
每个算子只预热一次、捕获一次，分开保存小型trace与kernel名称集合比较；不写频，不报告容量或能量参数。
`kernel_names_matched_only`只表示四项kernel名称集合一致，不验证顺序/次数、实际输入数值、缓存状态或完整TP路径。
合成BF16张量只用于派发诊断；`same_path_verified`、`calibration_ready`固定为false。
rank0通过不能替代rank1。WDDM进程显存返回N/A时不擅自终止其他用户进程。
开发侧40项相关CPU检查通过，目标GPU诊断尚待运行。

统一入口已经包含：

- `switch-check --identity ... --telemetry ... --gate ... --apply --out ...`：每方向5次切换，保留API返回、确认时间、原始样本及恢复。输出保守上界；默认不解锁look-ahead。
- `paper-fit --data ... --out ...`：显式物理单位、每rank容量/带宽与来源、执行签名、同路径复现证据、独立重复单元功率窗口、电压和温度。仅 `same_path_repeated_unit_board_windows` 输入可进入该路径。
- `paper-validate --data ... --model ... --out ...`：独立请求、冻结容量和签名；每rank/频点延迟和能量WAPE≤10%、P90≤20%。
- `plan --model ... --validation ... --coverage ... --switching ... --temperature ... --out ...`：Algorithm 1，N=5，相似阈值5%，保留短末块。`coverage` 可直接传 `observation.json`；temperature文件为实际两rank温度对象，键为 `"0"`、`"1"`，值必须来自本次读数。
- `serial --identity ... --telemetry ... --gate ... --plan ... --seed ... --apply --out ...`：一次真实512/64请求的双rank控制验收，不是收益验证。
- `serial` 追加 `--serial-evidence ...`：仅当冻结计划已有细粒度切换证据且匹配串行验收时启动look-ahead；现有100ms switch-check不满足此条件。
- `compare --identity ... --telemetry ... --gate ... --plan ... --serial-evidence ... --model ... --pilot ... --seed ... --apply --out ...`：独立三组配对窗口；`model`为冻结窗口基线模型，`plan`为论文模型生成的计划。

`paper-fit`输入的完整字段和验证代码在 `rtx5090/eager_model.py:check_data`。训练数据和验证数据须独立采集；不能手填一个 `verified:true` 来替代证据。当前未提供可直接生成这些标签的通用回放器，也未伪造可部署的示例标定文件。
若本轮trace不足以建立可靠回放、voltage读数不存在或回归秩不足，停在已取得的观测/窗口结果，明确记录缺项。

## 运行失败与结果解释

- 旧服务硬件诊断可使用 `rtx5090.service_model clock-check --diagnose-mhz 2250`（仍需 identity、telemetry、独立 out、`--apply --ack-default-auto`）。只运行该频点与带载正常恢复；成功状态为 `diagnostic_passed`，不能用于固定频率采集的完整验收证书。未执行异常/断连恢复验收。
- 首轮频率稳定检查无论成功或失败均保存 `settle-<MHz>.json` 的逐次双卡读回；保持 5 秒、15 MHz 容差与连续三次达标要求，不通过扩大容差掩盖失配。

- 任一卡失败：恢复两个已触碰的设备；API reset返回与实际恢复分开保存。恢复失败/未验证先处理，不能直接继续下一个调频命令。
- worker在执行中出错会标记poisoned；不会偷偷清状态、退回mock或自动重启。对端会合超时后不再释放新块。
- 若rank在底层CUDA/collective中挂起，Python不能保证强制解挂；需要用户正常停止自己的实验并核查双卡恢复。不得宣称任意硬故障都能自动恢复。
- 所有正式窗口计入控制、等待和必要日志开销。AUTO与固定组同样运行带禁用控制包装的eager服务。
- 双卡GPU功率不包括CPU/整机；模块级证据不升级为完整kernel图证据。
- `incremental_value`仅为本轮均值比较，不代表统计显著性。动态策略须在≤5%延迟增幅下优于满足同约束的冻结最佳固定频率；否则报告没有证实增量价值。
- 两频点都不满足延迟约束时，窗口模型的best_fixed为空；不放宽预算或改验证集取代该结论。

开发侧CPU测试结果与目标机验收分列记录于 `validation/eager/REPORT_zh.md`。
