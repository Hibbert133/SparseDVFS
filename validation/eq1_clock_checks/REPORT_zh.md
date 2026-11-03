# Eq. (1) 读回诊断检查记录

2026-11-03 在开发主机运行 45 项 CPU 检查：45 通过，0 失败，0 跳过。没有在目标 GPU 上采样或改变频率。本记录对应当前提交中的实现。

覆盖范围：33 项独立性能模型测试，保留 3 项 Eq. (1) 数值/依赖检查，另有 8 项 bridge/协议回归及 1 项 NVML 能量整数宽度/功率单位检查。新增测试使用合成频率序列、虚拟时钟和 API 替身，检查失败快照及阶段、有界连续读回、超时拒绝、测时后不重试、错误与恢复、离线数据门控及只读 NVML 回退。

在仓库根目录复核：

```bash
python -B -m unittest discover -s tests -p test_performance_eq1.py -v
python -B -m unittest discover -s tests -p test_v4_paper.py -k eq1 -k test_peak_cannot_depend_on_emc -k test_bandwidth_cannot_depend_on_cpu_gpu -v
PYTHONPATH=tests python -B -m unittest test_5090.ProtocolTests test_energy_5090.EnergyTests.test_nvml_energy_width_and_power_units -v
```

最后一条是 WSL Bash 语法，仅临时将 tests 加入模块搜索路径。Windows PowerShell 可使用本轮交接目录保存的运行记录；以上检查均不依赖 CUDA。

目标机证据仅来自用户反馈：doctor ready，2250 MHz 的 12 组完成，2527 MHz 首组在 `before_wall` 读回 2407 MHz 而失败；旧实现未保存该快照。reset 调用已返回，物理恢复未验证。修复后的 NVML 字段、时钟行为及有界模式仍需目标机新目录复核；本机通过的检查不能证明故障根因已找到、真实稳频或公式预测精度已验证。
