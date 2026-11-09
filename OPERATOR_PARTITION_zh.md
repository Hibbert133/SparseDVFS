# 算子粒度分块输入

2026-11-09：模块计时仅为辅助尺度诊断，不作为严格复现的正式节点。

SparseDVFS §4.2 Algorithm 1 从算子 DAG 的拓扑序开始，根据算子的预测最优频率合并连续节点。伪代码条件为当前块时间小于 N*T_switch 或频率相似时合并，块频率取最大值；最后追加尾块。Eq.(3) 的严格大于关系和短尾块需单独报告。N=5 是文中示例；5% 相似阈值是本移植选择。

`python -m rtx5090.operator_partition_inventory --trace TRACE --out NEW_OUT`
读取两个 rank 已保存的 request.json，默认选择 prefill、decode.0。无需 GPU、token、写频或重启。输出真实 dispatcher operator、所属模块标签、对应 kernel 名称及逐次调用 ID。融合/custom operator 保留，不臆造内部算子。缺少 kernel 映射仍标未知。

层模板按每个 rank/context 的完整签名序列分组，只是复用候选，不能证明不同权重、缓存状态或 decode 位置具有相同成本。旧 trace 的身份、原始文件哈希保留，不绑定当前常驻服务。

此输入清单不是可部署计划：dispatch 顺序不等于完整依赖 DAG；没有张量存储身份、跨 stream/collective 依赖证据时不构造虚假数据流边。每算子两频点延迟/能量保持空值，原 profiler 时间不作为标定成本。当前 worker 的模块边界控制不能自动升级为算子边界控制。后续按实际算子模型、依赖和控制边界完善，再执行 Algorithm 1，而非提前指定整层或整个 decode 为块。

论文：https://arxiv.org/html/2603.21908v1
