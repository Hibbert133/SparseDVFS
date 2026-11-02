# SparseDVFS

独立的 SparseDVFS 部署参考实现与验证记录。

完整的 **RTX 5090 / Windows + WSL2 / Qwen3.5-27B INT4** Demo 位于 [`5090` 分支](https://github.com/Hibbert133/SparseDVFS/tree/5090)。

本项目基于 Thor v5 参考 Demo 演进，保留来源记录；**不是论文作者发布的完整 C++ FUSE 源码**。作者仓库：[ziyangzhang96/SparseDVFS](https://github.com/ziyangzhang96/SparseDVFS)。

虚拟验证覆盖适配器、super-block 边界、真实 C++ 调度与 mock/NVML 桥接协议。真实 vLLM/CUDA、5090 硬件锁频、显存、性能和能耗仍须在目标机器上验证。模拟结果不能作为硬件收益。

`main` 保存项目概览，`5090` 保存代码、中文部署指南、版本锁和验证证据。仓库不包含模型权重、驱动、凭据或虚拟机镜像。
