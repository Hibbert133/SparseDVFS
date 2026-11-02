# SparseDVFS v5 论文逐项对照

依据 [SparseDVFS arXiv:2603.21908v1](https://arxiv.org/html/2603.21908v1)，本包是 Thor/vLLM 的独立参考移植。作者仓库固定快照仅公开 SparseDVFS.py，没有作者 C++ FUSE。代码完整性与实验结果复现分别记录。

| 论文位置 | v5 对应实现 | 当前差异和证据边界 |
|---|---|---|
| §4.1.1 Eq. (1) | paper_core.timeline_seconds；paper_calibration | 原式 max(compute,memory)+overhead；CPU/GPU 容量和 EMC 带宽依赖受校验。Thor 容量只是待实测微基准估计 |
| §4.1.2 Eq. (2) | paper_core.thermal_power_w；calibration_fit | 一单元一个 alpha、共享 k1/k2；QR 拟合/留出规则为本包实现选择。三域的标量 V/f 与电源轨范围必须明确 |
| 有效稀疏度 | 所有 Qwen 描述与采样 | BF16 dense 统一 S_comp=S_mem=0；真实跳算/压缩机制没有证据 |
| §4.2 Algorithm 1 | paper_core.greedy_partition；graph_partitioner | 保留拓扑顺序、短块 OR 近频率合并、逐分量 max、末块无条件追加 |
| Eq. (3) | 每块 amortized_strict_gt | 严格大于；与伪代码的等号/短尾块差异直接暴露，不伪造满足条件 |
| 算子图 | module/ operator 双路径 | module 为 68 段粗链；operator 为 dispatcher 单元，融合保持原子，限已观察单流串行图；不是完整 kernel DAG |
| §4.3.1 CPU-GPU | Schedule → FUSE submission ticket | CPU 提交窗口升频、提交返回释放；Python/dispatcher 返回近似 offload 完成，非透明逐 kernel interposer |
| §4.3.2 GPU/EMC | paper_calibration.coordination_vectors | sparse-memory 高 EMC、compute-bound 最低充分 EMC；dense-memory 未写明的分支保留显式 proposed fallback |
| §4.3.3 look-ahead | CUDAFUSE.cpp 事件线程及过期保护 | 立即写 sysfs 可影响当前块；作者延迟生效寄存器指令队列未公开，不能声称等价 |
| §5 平台及测量 | Thor direct nodes、功率窗口、worker trace | 从 Orin/原工作负载移植到 Thor/Qwen/vLLM；没有实机、论文表格/节能数字验证 |

## 本次从参考资料采用的内容

- [FUSE 原论文 §6](https://arxiv.org/html/2507.02135v1#S6)帮助区分 prefill/decode 及联合频率搜索；它是 Android/llama.cpp 阶段控制，且保留默认 memory governor，不能充当本包 C++ 的源码来源。
- [vLLM 0.19.0 profiler](https://github.com/vllm-project/vllm/blob/v0.19.0/docs/contributing/profiling.md)用于诊断覆盖，正式测量关闭 profiler。
- Zeus 与 jetson-stats 仅参考设备发现、测量窗口和接口差异。单位以 Linux ABI/Thor 驱动为准，没有照搬其他 Jetson 的固定路径或量纲。
- PELM 的投机解码、早退、DQN、风扇和 nvpmodel 改动不引入本实现。

## 不能用服务启动成功代替的验收

算子证书需要固定输入/输出、三次请求、完整 scheduler 上下文、GPU 活动归属及单 stream/device；每个 profile OPP 再做诊断。运行时检查 dispatcher 签名，但不能证明不透明融合单元未来永不新增隐藏辅助流。算法服务继续限制单请求 eager；Graph、批处理、多 GPU 不在此版本范围。

严格 Eq. (1)/(2) 路径必须补齐真实 FLOPs/有效流量和独立单元功率，并满足可辨识性及留出误差门槛。整请求平均功率乘单元时间始终只是代理。当前 paper_experiments_reproduced=false，详见 assumptions.md 与 TEST_RECORD_zh.md。
