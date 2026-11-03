# vLLM 执行清单：第一次交付检查

日期：2026-11-03。代码基线：22cb1c7；新增观察实现以本文件所在 Git 提交为准。

本机 CPU 检查 **63 项通过，0 失败，0 跳过**。其中观察工具 18 项，既有独立性能模型 33 项、Eq. (1) 数值及依赖 3 项、后端协议 8 项、NVML 单位 1 项。选择列表见 `cpu-tests.json`。可单独复核新工具：

```bash
python -B -m unittest discover -s tests -p test_observe_vllm.py -v
```

既有 CI 的 `run_virtual_validation.py` 自动发现 `tests`，会包含新测试；本次没有在本机重跑全部 native/CUDA 套件。

检查涵盖双 rank/UUID、真实调度计数的部分 prefill/decode、签名与留出隔离、融合身份保持、多 stream correlation、无法归属工作保留、输入输出长度、整个模拟 RPC 生命周期、异常 profiler 清理，以及无逐算子同步。模拟数据不构成目标机硬件证据。

目标机尚需运行 `VLLM_OBSERVE_zh.md`：核对真实 worker/runner、双卡 CUDA profiler、调用元数据及覆盖率。未运行目标模型、未修改频率、未验证 Eq. (1) 参数或精度；未实现回放适配器。原服务命令未显式 BF16/eager/V1，本轮明确这些设置并记录差异。

用户提供的旧 performance 结果已记入 `alignment/eq1.yaml`：24 组采集完成、10 项预测核对的 ±10% 命中率为 40%；开发侧未取得原始数据复核。reset 返回成功仍不等于物理恢复已验证。

## RPC 版本类型修复

2026-11-03 用户实机反馈 `process_output_sockets` 无法序列化 `torch.torch_version.TorchVersion`。修正 identity 中的版本字段为内置 str，保留安全序列化默认值。新增 str 子类夹具，调用真实 observation_identity 并递归检查返回树的精确内置类型。修复后 19 项观察工具 CPU 测试全部通过；没有在本机安装 torch/msgspec 或执行真实 RPC/CUDA，目标重采待确认。上述 63 项为前次检查记录。
