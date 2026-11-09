# v5 事实、参考选择、假设及未验证项

## 2026-11-06：Eq.(1)/(2) 离线计算入口

- worker v2的matrix-calibrate对真实linear与同张量回放分别做短kernel trace，关闭profiler后测不同批次数。kernel序列和geometry可核对，但缓存等价仍未建立；batch截距/斜率只是摊销诊断，不作为Eq1容量或系统开销真值。完整独立参数标定尚未完成。

- `independent_capacity`是外部合成primitive独立标定的条件性估计器：tiny-copy开销代理、BF16方阵计算、逻辑copy带宽，先冻结后比较留出primitive和已有MLP。它不匹配Qwen实际kernel/缓存/批内频率，不完成严格同路径或Eq2标定，不能用于正式选频。显存门槛与未完成项见 `INDEPENDENT_CAPACITY_zh.md`。

- 后续新增 `observe --matrix-replay` 真实X/W复用计时：每rank每相位/投影首个单元、3批×32次，属于实施选择；重复访问的缓存状态与原请求不等价，不作为容量拟合/独立验证/实测算子能量。详见 `MATRIX_REPLAY_zh.md`。

- 审计原文v1 §4.1.1–4.1.2，保留max时间模型和线性温度漏电项；没有从正文的super-linear描述另造方程。
- 新入口分别计算Eq1/Eq2，缺项为null，数据/配置哈希绑定；公式数值可计算不代表参数有效。
- 逻辑bytes不自动作为D_mem；P/B/开销及电压/功耗系数无默认实机值。来源文本仅记录声明，不能授权硬件。
- 算术demo的11秒/6瓦/66焦耳是人工夹具，不代表5090性能。实机参数仍待同路径标定与独立验证。
- 详细对应关系、执行命令见 `EQUATIONS_OFFLINE_zh.md`；现有正式拟合/控制门槛保持不变。

## 2026-11-05：5090_eager 独立分支补充（优先于下面的历史单卡约定）

- 本轮模型为真实BF16、auto加载、TP=2、全部64层，固定512输入/64输出；旧dummy/单卡INT4入口保持为历史工具，新入口为 `rtx5090.eager`。
- 用户最终目标包含原论文全部机制，端到端相对eager AUTO延迟增幅≤5%。本次软件交付不等于完成该目标。
- 新双rank控制通过独立Unix通道会合，先全设备CUDA完成，再成对写频/读回。Python协调、模块链和双GPU向量属于移植选择，不是作者的完整C++协同运行时。
- look-ahead队列已实现取消/等待机制，但在匹配串行验收与细粒度切换证据缺失时禁止启用。100ms样本加固定等待仅提供保守上界；不推断真实过渡精度或开销已隐藏。
- CPU与EMC后端在当前Windows/WSL方案中不可用；不以占位频率、显存时钟或mock冒充这些域。原数学三域接口保留。
- Eq.(1)/(2)新拟合入口只接收独立的同路径重复单元实测标签。trace不自动生成隔离回放；请求平均功率不生成单元标签。通用回放采集器、目标机电压证据与真实标定仍缺失。
- 两板热回归各自共享k1/k2、每执行单元一个alpha；容量/带宽独立输入且带来源。该参数估计方法和双卡能量组合是实现选择，不是论文明确给出的双GPU方法。
- Algorithm 1复用OR/max/尾块语义；GPU投影使用真实两卡MHz向量，不注入虚假的CPU/EMC数值。N=5、相似阈值5%、验证误差标准与选频优化器的来源分开记录。
- 新窗口fit仅为512请求级经验模型，始终 `block_selection_authorized=false`；动态策略只接受另行独立验证的论文模型和本次硬件门槛。
- 本次本地测试仅CPU/模拟控制流；实际Qwen加载、两rank行为、真实锁频/恢复、能量、动态策略收益均待目标机。

详细执行与未完成项见 `EAGER_5090_zh.md`，开发侧验证见 `validation/eager/REPORT_zh.md`。

研究来源核对于 2026-10-29；v5 实施/交付记录使用 2026-10-30。用户指定无运行环境；本次仅生成 workspace 文件、执行现有 Python/静态工具和临时模拟节点测试，没有安装软件、下载模型/镜像、启动设备服务或写真实频率。

## 论文明确规定的事实

| 编号 | 事实 | 对应代码 |
|---|---|---|
| P01 | Eq. (1)：max(W*(1-S_comp)/P_peak(cpu,gpu), D*(1-S_mem)/B_mem(emc))+overhead | paper_core、paper_calibration |
| P02 | Eq. (2)：alpha(S_comp)*V²*f+(k1*T+k2)*V | paper_core；共享热系数合同 |
| P03 | Algorithm 1：拓扑顺序，短块 OR 频率近似合并，逐分量 max，尾块追加 | greedy_partition；两种粒度复用 |
| P04 | Eq. (3) 严格大于 N*T_switch，N=5 为示例设置 | 摊销标记；不改写短尾/等号结果 |
| P05 | §4.3 的提交期 CPU 协同、GPU/EMC 协同和 look-ahead | submission/event reference；原生参考适配 |
| P06 | 原论文实验面向 Orin 与 CNN/ViT 等负载 | Thor/Qwen/vLLM 是用户指定移植，非原实验 |

## 参考或用户确定的选择

| 编号 | 选择及依据 | 边界 |
|---|---|---|
| R01 | T5000/128GB、固定 Qwen revision、固定 NVIDIA 容器 digest | VERSIONS.json；本机没有拉取/运行 |
| R02 | vLLM Worker 子类、compute_logits 包装、采样独立区间 | 锁定 0.19.0 源码；不修改安装好的 vLLM |
| R03 | PyTorch TorchDispatchMode 与 vLLM profiler | 实际镜像内 API/CUPTI 可用性待验证；私有 API 不能假定跨版本稳定 |
| R04 | NVIDIA Thor CPU/GPC/EMC、功率/电压/PTO 说明 | 名字/量纲/格式在实机核实；未验证电压 scale 留空 |
| R05 | Zeus/jetson-stats 的测量/发现思路，PELM 的接口适配提醒 | 不复制新参考代码，不引入 PELM 的其他算法或宿主机改动 |
| R06 | 被引 FUSE 是 Android/llama.cpp 阶段频率控制，默认 memory governor 保留 | 不能视作 SparseDVFS 缺失的 C++/寄存器实现 |

## 本包提出的工程假设

| 编号 | 假设/默认 | 未验证部分与影响 |
|---|---|---|
| A01 | TP=PP=DP=1、单请求 eager、纯文本，正常 KV/Mamba cache 保留 | 无 Graph/并发/多 GPU 支持；实际覆盖待 Thor |
| A02 | 默认 module 68 段；operator 为 dispatcher 叶单元，融合算子保持原子 | 并非完整 kernel DAG；有绕过 dispatcher 的 GPU 工作时证据检查失败 |
| A03 | 以一个观测到的 GPU stream 串行执行图作分块输入 | 三次请求和每 OPP 诊断不能证明未来不透明内核内部永不新增隐藏 stream；正式运行只直接检查当前 stream/签名 |
| A04 | 证书绑定模型/镜像/输入输出/上下文，prefill 与各 decode 位置分开 | 只允许固定请求复现；不同输入、长度或实现须重采集 |
| A05 | 三次 post-warmup 诊断；每定向切频默认十次；T_switch 全局 p95 | 都是本包验证/统计选择，不是作者参数或物理 PLL 稳定时间 |
| A06 | 近频率采用相对 L∞，预算/容差由调用者提供；目标为 min(P*T) 且满足区间预算 | 论文未完整定义 optimizer、距离和默认门槛；区间预算不保证端到端 SLO |
| A07 | block 时间是当前合并频点下已 profile 单元时间之和 | 只适用于本包串行图，不能外推并行 critical path |
| A08 | dense BF16 的 S_comp=S_mem=0 | 未验证跳零/稀疏存储 kernel；激活零比例不作为有效稀疏度 |
| A09 | compute-bound 最低充分 EMC；sparse-memory 最高已 profile EMC | 论文“充分/高”的定量化是本包选择；dense-memory fallback 明示 proposed |
| A10 | shared k1/k2 + 每单元 alpha 的缩放 QR 拟合，每四点留出一点，至少四点/单元 | 拟合、相对秩阈值 1e-10 及误差验收规则是本包选择；需真实温度/V/f 覆盖 |
| A11 | Eq. (2) 的 power_scope、标量 V/f 映射必须显式 | 论文没有完整的三域多电压物理映射；若用 GPU rail 拟合，不能冒充整体 VIN 模型 |
| A12 | GEMM/copy 仅估计有效算力/带宽；固定 P_peak 依赖 CPU/GPU，B_mem 依赖 EMC | 缓存、频率耦合可能使模型不准确，留出验证应暴露差异；不是所有 Qwen 内核真实性能 |
| A13 | 仅对已识别 mm 推导 2MNK；逻辑 tensor bytes 不自动作为 data_byte | attention/GatedDeltaNet 等 work/真实流量未知；严格路径要求补充来源 |
| A14 | collect 的请求功率积分不能生成单元 alpha | unit_records 默认空；需要外部独立单元真实功率/窗口证据，当前未采集 |
| A15 | 提交完成用 Python/dispatcher 返回近似；CPU 释放到 run 频点 | 不等于恢复默认 governor，也不是透明逐 kernel launch 拦截 |
| A16 | 主机观察 start event 后按估计时间发 look-ahead；立即写 sysfs | 轮询/线程调度存在延迟，可能影响当前块，不等价于作者寄存器队列 |
| A17 | 相同 GPC/EMC 目标是 no-op；短/不满足严格摊销的块不提前请求 | 保守参考策略；只有 no-op/无提前请求时 report 不通过完整机制验收 |
| A18 | 单 writer mutex、flock 与顺序三域写入 | 不是原子三域提交；外部 governor、热/过流限频及不使用锁的进程可干扰 |
| A19 | 窗口功率按真实单位、刷新周期与边界积分；VIN 不与子轨相加 | 20ms 主机轮询不等于传感器刷新率，也不能解析每个短块物理频率 |
| A20 | synthetic 示例的 OPP、容量、电压、温度、功率和 trace 全部人工构造 | synthetic=true/deployable=false；数值仅验证数学和接口，不能作为设备输入 |

## 尚未验证及必须补齐的证据

- C++/CUDA 编译、链接、CUDA API 类型与 ABI=5 实际加载；结构解析不替代编译。
- 固定镜像/BSP、27B 权重加载、64 层与采样真实覆盖、TorchDispatchMode 与 CUPTI/Kineto 实际输出格式。
- 每 OPP 真实写入/读回、隐藏辅助流、CUDA 错误清理、异常恢复与全局锁行为。
- Voltage/PTO 原始节点单位、每轨范围和 sensor interval；独立单元功率、完整 FLOP/流量描述及真实拟合参数。
- 物理 PLL 稳定时间、无流水气泡、真实端到端收益以及原论文任何图表数值。

代码生成、静态检查、合成执行、设备验证和论文实验复现是五个不同状态，见 VALIDATION.json、TEST_RECORD_zh.md 与 paper2code/04_verification.yaml。

## 2026-11-03：独立 Eq. (1) 第一阶段

此阶段对应 `rtx5090.performance`，不改变现有 Qwen/Thor 计划的验收与数据量规则。操作见 `PERFORMANCE_EQ1_zh.md`。

| 编号 | 来源或实现选择 | 明确边界 |
| --- | --- | --- |
| E01 | 原文 §4.1.1 的 max(compute, memory)+overhead 直接复用数学核心；作者代码 c524c51 提供 0.5 秒等待、5 次预热、5 次同步墙钟计时取均值 | 作者公开代码未给出 Eq. (1) 的容量/带宽/固定开销估计器 |
| E02 | 用户选择先用可解释 GEMM/加法的静态输入输出逻辑字节量 | 标记 logical_tensor_bytes_not_measured_DRAM；不修改 A13 所述完整模型路径，不宣称实际 DRAM 流量 |
| E03 | BF16 8192 GEMM 估计各频点有效计算容量；2527 MHz、256 MiB/缓冲区加法估计共享逻辑带宽；容量使用独立 CUDA Event 时间 | 这些形状、容量估计器与独立 Event 采样是本阶段选择，不是作者微基准参数或硬件理论峰值 |
| E04 | 一个全局 T_overhead，取标定形状墙钟残差的非负中位数；另选形状只用于预测核对 | 保留负残差数量与未截断中位数；不按频率或验证结果增添自由参数 |
| E05 | 单张 5090，GPU 2250/2527 MHz；CPU/显存策略不变；正式读回要求同一显存档、核心误差不超过 15 MHz | 一维切片；15 MHz 是移植检查选择；reset 返回不证明物理 PLL 稳定 |
| E06 | 报告平均/最大预测相对误差及 ±10% 内比例 | 无额外精度通过阈值，不能照搬论文 CNN/ViT 命中率或据此宣称节能 |
| E07 | 本机使用合成数据和 API 替身检查数学、采样口径、数据隔离与异常恢复 | 目标 GPU 的真实 CUDA、时钟稳定性、参数值和预测误差仍未验收 |
| E08 | 默认保留 0.5 秒等待和一次严格读回；显式 `--clock-stability-timeout-s` 可在测时前加入有界、连续 3 次达标检查，重查间隔 0.1 秒且执行一次额外同形状算子 | 额外操作不计时但改变负载历史，属于移植选择；不能证明硬件持续稳频或遥测更新；测时后不重试，15 MHz 和频点保持不变 |
| E09 | 失败读回、阶段、同步至查询的客户端间隔、查询往返及可用 NVML 限频/功率上限/P-state 字段用于诊断 | Windows 与 WSL 时间不相减；查询不是原子快照，功率等只读字段不用于本阶段功耗拟合；退出后状态和无差值累计计数不能追溯原因 |

## 2026-11-03：真实 vLLM 清单优先，第一次交付

用户报告旧 GEMM/加法 24 组成功，10 个核对项平均绝对相对误差 33.926%、最大 112.332%、±10% 命中 40%；这不证明容量或开销适用于真实推理。暂停人工方阵扩样。

- 独立 `rtx5090.observe` 只观察，不使用旧 worker 固定层数/单卡量化/单 stream 假设，不接 FUSE 或调频。
- 用户成功 serve 命令只显式设置 dummy、TP=2、4096、单请求、内存 0.90、qwen3 parser；BF16/eager/V1/language-model-only 是本轮计划显式选择，不冒称原服务默认路径。离线 token-ID 请求绕过 HTTP/parser。
- 保留用户脚本三个 NCCL 禁用开关与 `/tmp/triton360-vllm-cache` 缓存；独立 Triton 包路径仍为持久目录。关闭 prefix caching 是避免预热复用 prompt 的观察选择。
- 每场景一次预热、三次观察；128/512/2048 输入，64 输出；1024 场景封存。prefill/decode 依据 scheduler 真实计数，支持部分 prefill，不预设完整请求等于一步。
- profiler 关联 CPU launch 与 GPU correlation；无法归属的活动计入总活动分母并保持未知。名称类别为提示，不是瓶颈证据。CUDA Event 只是当前 stream 诊断跨度。
- 不逐算子同步；请求末同步导出。profiler、元数据及 Event 会扰动时序，时间不得用于容量拟合或正式延迟结论；重叠时间不相加当请求延迟。
- 全部签名初始为未实现回放、未验证 Eq1；融合操作不拆分。拿到目标机清单后才实现同路径回放、输出检查和独立容量标定。
- 本机仅 CPU 替身验证；目标 TP=2 CUDA 执行、CUPTI 关联率、内存开销及实际生成仍待用户运行确认。

## 2026-11-09：严格模型选频与纯 Algorithm 1

- 用户确认双5090/Qwen平台，原文未充分规定之处允许最小明确补全。选频与分块分离；分块不要求算子或块的实测能量，不进行任何再次寻优。
- 选频实现选择：在两候选中先求最快的模型预测时间，仅在不慢于该值的候选中最小化 Eq2预测功率乘Eq1预测时间；同能量按时间及固定候选顺序破平局。没有逐算子5%阈值；端到端5%仅作为最终项目验收。
- 不使用板级混合负载功率乘Event包络作为严格算子能量，不把Event中位数当作Eq1容量模型。历史线性回放及功率候选只作为明确标记的参考证据。
- N=5及频率相似阈值5%保持固定；前者为论文示例，后者为移植选择。合并后按新频率重新计算块时间，短尾块/等号单列Eq3审计。
- 部分dispatcher序列不是完整DAG；没有依赖/资源证据时只做非部署诊断。当前stream事件采集作为后续参数核实工具保留，单独版本门槛，不自动重启旧worker。


## 2026-11-09: TP embedding structural pairing

Target installed vLLM `_custom_ops.vocab_parallel_embedding` source supplied by the user calls
`torch.ops._C.vocab_parallel_embedding(out, input_ids, weight, org_start, org_end, padding, added_start, added_end)`.
Thus dispatcher args[3]/args[4] are original vocabulary boundaries (the wrapper itself has no out argument).
The offline inventory recognizes contiguous rank-ordered original shards starting at zero only when
all other normalized fields match, tensor shapes/types and shard capacity pass checks, and added ranges are empty.
This conservative adapter is a port choice, not a SparseDVFS paper rule. Nonempty added vocabulary remains unsupported.
Original differences and signature hashes are retained. No token-content, collective, current-service identity,
full vocabulary coverage, kernel equivalence, graph join or model parameter certification follows from this check.


## 2026-11-09: bounded copy/conversion workload definitions

Port choice: only nonempty, contiguous strided, equal-shape same-device CUDA tensors
with associated GPU kernels enter the copy/conversion traffic adapter. Same-dtype copy
has zero floating arithmetic and logical bytes equal to source plus destination sizes.
This does not certify non-aliasing, actual DRAM traffic, copy bandwidth or power; existing
same-path and independent calibration gates remain required. Broadcast, cross-device,
unsupported dtype and no-kernel cases remain unresolved.
Dtype conversion records logical bytes only, with work_flop=null and explicit unresolved
conversion_compute_path. It cannot enter Eq1 prediction until its compute path is modeled
and validated; casting is not silently assigned a zero-cost compute branch.


## Stateless calibration adapter boundary

`stateless_calibration_adapter.prepare/invoke` is worker integration groundwork, not
a connected collector. Only bias-free supported linear and supported floating binary
operations are admitted. Training and holdout IDs are disjoint; a read-only dispatcher
schema and exact supplied metadata are required. The caller must derive metadata from
actual operands and independently establish calibration input provenance, kernel/cache
matching, timing, V/f/T coverage and energy-window scope. Sample IDs alone do not prove
independence. No service RPC or collection command is enabled by this adapter.


## Resident independent primitive collection v1

`eager observe --independent-primitives --apply` arms the next request; CUDA work runs
in the model thread before its first prefill, never in the RPC thread. A normal worker
restart is necessary to load this capability. No clock setter or bridge is used.
Fresh seeded BF16 tensors, train square dimensions 256/512 and held-out 384/640,
exercise aten linear/add/mul in the existing context with bounded scratch memory.
Actual tensor metadata binds the prepared call; no model tensor or weight is reused.
Three warmups and three 32-call event batches are retained, including host launch gaps.
These port-choice shapes do not certify the real MLP kernel family or cache state.
No tiny-copy overhead estimate, automatic fit, Eq2 energy label or formal parameter
certificate is produced. CUDA failures propagate with a saved partial report; low-memory
prechecks reject before allocation. Target CUDA execution remains pending.


## No-write operator rendezvous v1

The experiment binds actual pre-call tensor metadata and parent plus the source dispatcher
ordinal, then synchronizes the whole device on the model thread and rendezvous via private
Unix sockets. It tests four selected boundaries, not a full DAG or dynamic-frequency plan.
A 10s device-wait watchdog aborts the coordinator; Python cannot cancel a hung CUDA driver
call. Peer wait is 10s, RPC 12s, HTTP request timeout 120s. Timeout may poison the worker
and require manual service recovery; no automatic restart or clock writes occur.
Old collection supplies candidate contracts; the new run records fresh identity separately
and rechecks actual operands. It does not transfer old model/hardware certificates.
