# v5 C++ FUSE / CUDA 时序审计

FUSE.cpp / CUDAFUSE.cpp 是独立参考代码，不是作者发布的 C++ 文件。ABI=5，Python binding 拒绝旧库。FUSE 在此指调频引擎，不是文件系统库。

## 保留的控制机制

- CPU submission ticket 防止旧提交者降低新提交者的频率；GPU boundary ticket 防止旧块 look-ahead 覆盖新块。
- 一个 writer mutex、配合进程共享的 flock；CPU/GPC/EMC 合法频表检查；正常/异常恢复快照。
- stream host function 负责有序边界和 CPU/sysfs 工作，CUDA 事件查询在普通线程；host function 内没有 CUDA API 调用。
- 相同 GPC/EMC 目标返回 no-op，单独记录；不计作真实资源写入或物理切换。

## 新增记录

core CSV 已分别给出排队、开始和返回时刻。CUDA CSV v5 再给出 boundary queued/started/returned、callback_duration_ns、start_observation_lag_ns、poll_late_ns。

回调耗时包括 writer 锁等待和节点操作；queue→callback 延迟还包含此前 GPU 工作，不能直接称为气泡时间。start_observation_lag 是 host 观察延迟上界的一部分，不是 GPU 精确开始时间。

switches.json 按实际 from→to、每个方向默认十次采样，源频点先就绪，排除 no-op。T_switch 采用全局 p95 请求至软件匹配读回，仍包括轮询和 keepalive，不是物理 PLL 时间。CPU cpuinfo 与可用 PTO 原始信息另列。

## 与论文仍不同的部分

论文 §4.3.3 的硬件寄存器/队列协议未公开。本包提前立即写 sysfs 约束，可能改变当前块的速度；下一块有边界兜底。其正确性与物理切换重叠需要实机时间线，不能由虚拟 ticket 测试推断。

CUDA host function 会阻塞其后入队的流工作。[CUDA 13.0 文档](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-runtime-api/group__CUDART__EXECUTION.html)明确限制回调内 CUDA 调用。本包计录回调耗时，不承诺零开销或无气泡。

两个共享库在本机没有编译、链接或执行。C++ 结构解析仅检查语法树，不能验证 CUDA API 类型、线程竞态、驱动行为或硬件恢复。需在 Thor 执行 selftest，并检查全部 native/CUDA 测试实际运行。

report 若没有任何提前资源请求，会给出 not_exercised；若只有 no-op，也不算完整 look-ahead 机制验收。即使写后 end event 仍 pending，physical_transition_overlap_verified 继续为 false。
