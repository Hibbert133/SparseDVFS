# SparseDVFS RTX 5090 reference demo

当前主线面向 **Windows + WSL2、双卡 RTX 5090 TP=2、Qwen3.5-27B 常驻优化服务**，先核实实际 dtype/权重和执行路径，再建立有独立验证的性能/板级能量模型。旧单卡 INT4、eager 和 FUSE 入口保留为历史参考。
这是从 Thor v5 参考实现演进的 GPU-only 版本，不是论文作者发布的完整 FUSE 源码，也没有已测得的 5090 加速或节能结论。

- **[常驻优化服务观察与建模](SERVICE_MODEL_zh.md)**（当前入口：一次启用 profiling，后续复用服务）
- [Conda + WSL2 / vLLM 0.32 手动操作指南](CONDA_5090_zh.md)（历史部署与兼容性记录，无 Docker）
- [显式使用已有独立 Triton 3.6.0](TRITON_360_zh.md)（保留 PYTHONPATH 覆盖，不修改 Conda 包）
- [0.32 兼容性依据与真实调用链](COMPATIBILITY_032.md)
- [旧版 Docker 部署指南](QUICKSTART_5090_zh.md)
- **[Conda 更新验证报告](validation/conda/REPORT_zh.md)**：188 项，185 通过，3 CUDA 跳过；含最终源码的 Linux CI
- [旧版虚拟验证报告](validation/REPORT_zh.md)
- [版本锁定](configs/5090.lock.json)：vLLM 0.19.0 amd64 镜像、量化模型 revision 与配置。
- [虚拟测试入口](scripts/test_5090.sh)：真实 C++ 编译、调度测试、桥接测试及假设效果模型。
- Conda 入口为 `python -m rtx5090.conda`；`rtx5090.run` 保留旧 Docker 路径。原 `demo.py`、`thor_repro.sh` 和 Thor 文档仅供历史参考，不用于 5090。

历史 FUSE 参考流程：真实 vLLM 模块 hook → super-block 边界 → CUDAFUSE → C++ FUSE 队列 → GPU 后端。当前 `service_model` 不启用此流程。
后端可选 mock、直接 NVML 或 Windows NVML bridge。super-block 表示连续执行区间的频率策略，不重排模型层，也不融合 CUDA kernel。

旧 Docker 模型名称带 AWQ，但其锁定 revision 配置为 **compressed-tensors、group-size 32、INT4 权重、BF16 激活**。这不是当前双卡服务的配置；当前 dtype/权重模式以实际启动日志为准。

## 证据边界

虚拟测试运行真实 C++ 调度器和 Python 适配代码；模型、CUDA 事件及 NVML 设备使用测试替身。它可以验证请求顺序、单位、生命周期和失败处理，不能证明真实 vLLM/CUDA 时序、显存适配或硬件收益。

`rtx5090.simulate` 是假设耗时/功率的算例，包含正、零、负收益；不是硬件仿真器。
历史 Conda/FUSE 路径使用实测 CUDA event 延迟生成候选，复用 Algorithm 1 参考分块，再按训练集整请求实测能耗选方案；独立测试集对照系统默认自动频率。旧 Docker 分区器仍是延迟预算启发式。两条路径均**不冒充论文完整能耗优化公式**，尚无已测得的 5090 收益。当前先走常驻服务观察和独立模型验证，分块暂缓。

## 来源

- 论文与作者仓库：<https://github.com/ziyangzhang96/SparseDVFS>
- 基线：本项目原 Thor v5 参考 Demo；来源说明见 `NOTICE.txt`、`SOURCE_MANIFEST.json`。
- 模型：<https://huggingface.co/cyankiwi/Qwen3.5-27B-AWQ-4bit>
- WSL 限制：<https://docs.nvidia.com/cuda/wsl-user-guide/index.html>

原 `VALIDATION.json`、`SHA256SUMS`、`UNIT_TEST_RESULTS.txt` 等是 Thor v5 历史记录；旧 5090 测试保留在 `validation/`，Conda 更新记录在 `validation/conda/`。仓库不包含模型权重、系统镜像或驱动。
