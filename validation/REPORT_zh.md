# 5090 虚拟验证报告

日期：2026-11-02。最终源码测试：**163 项，160 通过，3 跳过，0 失败/错误**。

执行入口为 `sh scripts/test_5090.sh`，在便携 QEMU 的 Alpine Linux 内实际编译并执行 C++。新增 5090 测试 13 项全部通过，其中 5 项运行 native 控制器；加上原版 7 项 native 测试，共 12 项 C++ 集成测试通过。

| 验证内容 | 本次证据 |
|---|---|
| 新 5090 worker、Spans、Schedule 边界 | 模拟模型与 CUDA 事件；1 prefill + 3 decode，272 unit，16 个 block 请求 |
| 新 worker → Schedule → C++ | 已贯通，使用 host serial 调试传输和 mock GPU，实际执行 C++ |
| C++ → 持久 socket bridge → mock NVML | 已贯通；Hz/MHz、去重、重置通过 |
| ticket/过期 lookahead/取消/异常 | 原版及新增 native 测试通过 |
| NVML/桥接协议 | 假 NVML 成功、不支持、权限错误、延迟、认证、UUID、序号、断连恢复、reset 失败测试通过 |
| 模拟收益 | 正/零/负场景均为假设值；不代表硬件性能 |
| 真实 CUDA/vLLM/量化模型执行 | 未执行；3 项 CUDA 测试明确跳过 |
| 5090 真实锁频、显存、性能/能耗 | 未验证；没有硬件收益结论 |
| 目标容器构建、WSL mirrored 网络、Windows NVML | 待实机验证；镜像和模型 revision 已查询并锁定 |

完整日志见 `tests.txt`，结构化结果及源码 SHA256 见 `summary.json`。`worker_5090_native_True.json` 是贯通测试轨迹；其中模型/CUDA 仍为替身，不能误读为真实 GPU 执行。

没有安装本机依赖、改变系统配置或真实 CPU/GPU 频率。编译依赖只安装到虚拟机内存系统。`isolation.json` 记录隔离方式与库哈希。

首次 socket 测试因最小 guest 没有启用 loopback 失败；只在 guest 中启用后重跑通过。最终报告使用最终源码重跑记录，不沿用旧版通过数。

本 Demo 的自动分区是实测延迟预算启发式，不等同于论文的严格能耗优化。operator 路径仍需真实 CUDA 覆盖认证。部署顺序及限制见根目录 `QUICKSTART_5090_zh.md`。
