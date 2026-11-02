# SparseDVFS RTX 5090 reference demo

面向 **Windows + WSL2、单卡 RTX 5090、Qwen3.5-27B INT4** 的实验 Demo。
这是从 Thor v5 参考实现演进的 GPU-only 版本，不是论文作者发布的完整 FUSE 源码，也没有已测得的 5090 加速或节能结论。

- [中文部署指南](QUICKSTART_5090_zh.md)
- [本次虚拟验证报告](validation/REPORT_zh.md)
- [版本锁定](configs/5090.lock.json)：vLLM 0.19.0 amd64 镜像、量化模型 revision 与配置。
- [虚拟测试入口](scripts/test_5090.sh)：真实 C++ 编译、调度测试、桥接测试及假设效果模型。
- 新入口为 `python -m rtx5090.run`。原 `demo.py`、`thor_repro.sh` 和 Thor 文档仅供历史参考，不用于 5090。

流程：真实 vLLM 模块 hook → super-block 边界 → CUDAFUSE → C++ FUSE 队列 → GPU 后端。
后端可选 mock、直接 NVML 或 Windows NVML bridge。super-block 表示连续执行区间的频率策略，不重排模型层，也不融合 CUDA kernel。

模型名称带 AWQ，但此锁定 revision 的配置为 **compressed-tensors、group-size 32、INT4 权重、BF16 激活**。不要强制传 `--quantization awq`。

## 证据边界

虚拟测试运行真实 C++ 调度器和 Python 适配代码；模型、CUDA 事件及 NVML 设备使用测试替身。它可以验证请求顺序、单位、生命周期和失败处理，不能证明真实 vLLM/CUDA 时序、显存适配或硬件收益。

`rtx5090.simulate` 是假设耗时/功率的算例，包含正、零、负收益；不是硬件仿真器。
实际 profile 只使用运行所得的 CUDA event 延迟；当前自动分区器按延迟预算选择最低候选频率并合并相邻区间，是明确标注的启发式，**不冒充论文严格能耗优化公式**。Thor 论文公式代码保留在原目录，未经重新标定不得用于 5090。

## 来源

- 论文与作者仓库：<https://github.com/ziyangzhang96/SparseDVFS>
- 基线：本项目原 Thor v5 参考 Demo；来源说明见 `NOTICE.txt`、`SOURCE_MANIFEST.json`。
- 模型：<https://huggingface.co/cyankiwi/Qwen3.5-27B-AWQ-4bit>
- WSL 限制：<https://docs.nvidia.com/cuda/wsl-user-guide/index.html>

原 `VALIDATION.json`、`SHA256SUMS`、`UNIT_TEST_RESULTS.txt` 等是 Thor v5 历史记录；本次结果以 `validation/` 和 `MANIFEST_5090.json` 为准。仓库不包含模型权重、系统镜像或驱动。
