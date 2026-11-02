# v5 事实、参考选择、假设及未验证项

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
