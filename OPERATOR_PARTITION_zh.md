# 算子粒度分块输入

2026-11-09：模块计时仅为辅助尺度诊断，不作为严格复现的正式节点。

SparseDVFS §4.2 Algorithm 1 从算子 DAG 的拓扑序开始，根据算子的预测最优频率合并连续节点。伪代码条件为当前块时间小于 N*T_switch 或频率相似时合并，块频率取最大值；最后追加尾块。Eq.(3) 的严格大于关系和短尾块需单独报告。N=5 是文中示例；5% 相似阈值是本移植选择。

`python -m rtx5090.operator_partition_inventory --trace TRACE --out NEW_OUT`
读取两个 rank 已保存的 request.json，默认选择 prefill、decode.0。无需 GPU、token、写频或重启。输出真实 dispatcher operator、所属模块标签、对应 kernel 名称及逐次调用 ID。融合/custom operator 保留，不臆造内部算子。缺少 kernel 映射仍标未知。

层模板按每个 rank/context 的完整签名序列分组，只是复用候选，不能证明不同权重、缓存状态或 decode 位置具有相同成本。旧 trace 的身份、原始文件哈希保留，不绑定当前常驻服务。

此输入清单不是可部署计划：dispatch 顺序不等于完整依赖 DAG；没有张量存储身份、跨 stream/collective 依赖证据时不构造虚假数据流边。每算子两频点延迟/能量保持空值，原 profiler 时间不作为标定成本。当前 worker 的模块边界控制不能自动升级为算子边界控制。后续按实际算子模型、依赖和控制边界完善，再执行 Algorithm 1，而非提前指定整层或整个 decode 为块。

论文：https://arxiv.org/html/2603.21908v1

## 严格分块接口修正

partition_dryrun 只接收上游 selected_frequencies、时间预测、切换时间及拓扑顺序，不要求能量字段，不进行选频。pair-partition-input-v2 必须明确 verified_serial 或 resource_dag 时间语义以及依赖证据；串行累加需要无重叠证据，resource_dag 的边须含 stream 资源约束和 collective 关系。v1 兼容路径只标为 legacy_serial_projection。

分块核心保留 Algorithm 1 的 OR/max/尾块语义，另审计严格 Eq3。所有离线结果 deployable=false。request_budget_partition 保留为历史替代实验，不是当前严格复现入口；其能量要求只属于该选频器。

软件示例：`python -B -m rtx5090.partition_dryrun --demo --out /absolute/new/algorithm1.json`。示例频率由夹具明确提供，不证明模型或硬件通过。

## 模型接入与下一步（离线，不重启）

```bash
python -B -m rtx5090.operator_parameter_gaps \
  --inventory runs/operator-partition-input-20261109-170604 \
  --linear runs/eager-validation-1793951462455330921/linear-two-settings-20261109-160036/collection.json \
  --out /absolute/new/parameter-gaps
```

生成 `calibration-tasks.json`、`equations-template.json` 与精简 `REPORT.txt`。任务按 rank、相位、实际 operator/kernel 家族去重，同家族跨形状的参数复用仍需验证。首层/GDN层/FlashAttention层优先作为代表。BF16无bias linear自动计算理论FLOP及逻辑张量字节数；其他算子未知工作量保持空值，不默认填零。两rank签名不匹配单列，缺依赖不虚构链。

历史linear形状匹配记录为可复用诊断证据，不等于独立容量标定。共享热参数按卡只列一组任务，不为每次算子调用重新标定。模板的所有证明字段默认未通过，严禁为运行通过而手填true。

`python -B -m rtx5090.operator_equations --input /absolute/frozen-equations.json --out /absolute/new/predictions`
执行 Eq1/Eq2 公式，不拟合、不访问服务。要求与参数哈希绑定的独立验证、原始测量证据、家族签名覆盖、已知的预测前V/f/T、同路径成对单元功耗与可辨识热参数。参数变更、训练/验证交叉、混合请求功率代理、温度或实际频率越界均拒绝。输出保留每rank时间/功率，以及上游无减速选频结果；之后只把频率与时间传给分块器。

选择规则：在最快预测时间的候选中，取最低预测能量，平局依次比较时间及固定候选顺序2250、2527。这里的MHz是控制请求标签；Eq2使用状态映射中的实际Hz，不使用标签冒充硬件实际频率。两rank max时间/共同功率窗口需要独立join证据。若该假设不适用，必须补建实际双rank依赖模型，不能强行填证明字段。

模型模板故意不包含当前service绑定、有效参数或DAG认证，所以直接运行模板会产生有原因的 `blocked_parameter_evidence`，不生成假选频。它不是新增采集失败，不需重跑已有实验。

## 已准备但尚不启用的采集能力

`linear_frequency_probe --scope operators` 是只执行一次原始算子的 current-stream Event 诊断。要求新worker的 `operator_cost_version=1`；旧服务在任何写频前被拒绝，不热替换进程。无CUPTI、无权重副本、不重复attention/KV/GDN/collective，仅采prefill和decode.0，原始512/64请求正常完成。该功能不是独立容量或功耗标定器；本轮不要求为它重启或采集。

正式标定适配器与状态保护齐备后再安排一次正常重启，复用双卡恢复事务；异常/SIGTERM保留原始数据并尝试带载恢复。无法恢复时停止后续写频。当前串行算子级部署和look-ahead尚未得到目标机证据，不输出可执行硬件计划。
