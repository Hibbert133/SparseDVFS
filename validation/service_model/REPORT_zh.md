# 常驻服务工具验证记录

日期：2026-11-05。开发机 Windows / Python 3.14.5；不是目标机 WSL2、vllm-qwen 或 RTX 5090 环境。

## 已完成

- 新增 `tests/test_service_model.py` 47 项 CPU 检查全部通过。
- 与原观察、性能模型、bridge、能量、实验入口相关回归合计 **133 项：128 通过，0 失败，0 错误，5 跳过**。
- 5 项跳过均因本机没有 Linux native library，详情保存在 `cpu-results.json`；不是目标 GPU 验收通过。
- 新 Python 文件及 bridge 通过 Python 3.12 语法解析；启动脚本通过 Git Bash `bash -n`；Git diff 空白检查通过。
- CPU CLI 用隔离的测试 fixture 完成 fit → 冻结 → 新验证文件 → validate，验证模型文件未被改写。fixture 不是实测参数，不纳入研究结果。

覆盖内容包括：进程代理隔离和 urllib 显式绕代理、实际 token usage 与 benchmark 计数不符、cache hit/指标缺失/外部请求、双卡 UUID 错配、跨主机单调时钟区间映射、能量计数复位/间歇失败/功率回退、窗口过短和缺样、图 replay 一个 launch 多个 kernel、尾段活动缺失但归属率仍为 100%、trace dropped 元数据、单卡写频失败后双卡恢复、重设频率清除旧恢复凭据、训练与验证泄漏/协议/温度/频率范围检查。

编排检查覆盖流式请求中断保存部分回复与原始遥测、profiling 启动失败后仍执行 stop、原生 profiler 日志异常阻止正式测量、无效正式窗口触发双卡恢复且失败数据仍不能拟合、`/v1/models` 动态 created 时间不污染常驻身份。

完整逐项日志：`cpu-tests.txt`。在仓库根目录复核相同范围：

```bash
python -c "import sys,unittest; sys.path.insert(0,'tests'); s=unittest.TestLoader().loadTestsFromNames(['test_service_model','test_observe_vllm','test_performance_eq1','test_5090','test_energy_5090','test_experiment_5090']); r=unittest.TextTestRunner(verbosity=1).run(s); sys.exit(not r.wasSuccessful())"
bash -n scripts/serve_profiled_5090.sh
```

## 尚未完成的目标机验收

2026-11-05 后续文件名修复：用户文本确认同一 prefill 窗口已导出 rank0/rank1 两份 gzip trace（387509/387498 bytes）。旧匹配只识别 `_tp`，现支持原生 `_rank`，新增 `analyze-observation` 离线恢复，不改原失败记录，不代表 trace 内容已经验收。新增 3 项检查；本次运行 `python -m unittest tests.test_service_model tests.test_observe_vllm -q`，72 项全部通过。此前完整回归日志仍保留为原交付记录。

| 项目 | 本次证据 |
| --- | --- |
| 已安装 vLLM profiler 字段/原生迭代语义 | 源码检查与预检查工具已实现；目标安装待运行 |
| 唯一一次服务重启、接口注册、真实 dtype/加载方式 | 未执行；不能沿用历史 dummy/BF16 |
| 两 rank 实际图 replay、融合 kernel/通信清单 | 未采集；形状/layout/节点映射仍可能缺失 |
| 双卡真实锁频、稳定读回、正常/异常/断连物理恢复 | 未执行；历史 reset 返回不构成本次通过 |
| 双卡同窗口能量、遥测分辨率与误差预算 | 未测；3% 预算不代表硬件传感器精度 |
| 本轮经验模型参数、独立 WAPE/P90 | 未产生；测试 fixture 不是测量结果 |
| Eq.(1) 同路径容量/开销与 Eq.(2) 参数 | 未完成；无证据就不拟造回放或算子能量 |
| 图内分块/合法控制边界/切换开销 | 未验证 |

新工具只提供请求窗口经验模型，不修改原论文模型，不代表 FUSE 或收益实验已经接入默认优化服务。目标机按 `SERVICE_MODEL_zh.md` 从旧服务仍运行时的 `--check-only` 开始，每步输出精简文本；失败停止当前步骤，保留原始目录，不重复加载模型绕过问题。
