# 独立资源模型采集与验证

本交付要求 resident worker 的 `resource_calibration_version=2`，支持可配置时长。旧常驻服务不能热加载，需先 preflight，再由用户正常停止旧服务并启动一次新服务。没有驱动、依赖或权重变更。

## 方法与当前界限

- Eq1 使用 `h + max(a*W/f_Hz, b*D)`：计算能力随频率线性缩放、固定显存频率，是明确移植假设。通过多个算术强度样本辨识两条路径；无法辨识即返回 unidentified，不填人为峰值。copy 的 W=0 只辨识内存路径。转换、特殊函数/归约保留工作量未知，不虚构 FLOP。
- 延迟为独立资源负载的重复 Event 窗口；保存 N=1/8/32/128 每档三次，模型用 N128 中位数。不能当成单次 Qwen 执行模型。CUDA stream 中可能包含 launch gaps；开销参数不冒称纯启动延迟。
- 资源负载都在 worker 模型线程执行；不创建额外 CUDA context。linear 使用独立 BF16 输入和小型权重；不修改模型张量；attention、cache、collective 不回放。
- 两 rank 通过私有 Unix RPC 在 GPU 完成后开始/结束 Windows 本地电压/能量窗口。每种负载默认30秒，`--window-seconds` 支持10至120秒，窗口包含协调和同步空隙。不能称为纯 kernel 能量。
- 每卡共享 k1/k2，每类资源独立 alpha，按积分 `V²f, TV, V` 拟合 Eq2。未加入隐藏截距、正则化或参数裁剪。报告条件数、能量预算对参数的精确边界敏感性、逐窗口残差。
- 默认两种训练形状、两频率2250/2527、三次交错重复；默认 linear/add/mul/copy 共48个训练窗口。中间共同支持频率、独立形状和种子共12个留出窗口。热范围由实际观测决定；不能保证自然温差足以辨识热参数。
- 单个窗口除30秒负载外还有请求/预热/恢复开销，60个正式窗口仅负载即30分钟，总批次更长；每个窗口立即输出进度。不自动重试失败窗口。
- 采样实际 V/f/T 与计时批次不是同一时段，因此预测标为条件诊断。没有传感器误差证书、Qwen同路径/缓存证据，始终 `parameters_validated=false, deployable=false`。这是仍需完成的验证，不因数值误差低自动放行。
- 输出 `ncu-query-metrics.txt` 仅证明指标查询情况，不是计数器采集或kernel核实。ncu不存在/不支持不安装依赖。
- `resource_model validate` 只读取冻结参数；拒绝训练ID、训练形状和训练频率重复，不删除坏样本。报告WAPE/P90和未解析行；统计达标与参数证据达标分开。

## WSL Bash：检查和运行

服务仍运行时可先执行：

```bash
python -B -m rtx5090.resource_calibration check \
 --run "$RUN" --telemetry configs/service_telemetry.template.json \
 --out "$PWD/runs/resource-check-$(date +%s%N)"
```

旧 worker 会返回 `NEEDS_NORMAL_RESTART`，此时无写频。按既有 `scripts/serve_eager_5090.sh --check-only "$NEW_RUN"` 完成 preflight 后，在原服务终端 Ctrl+C 正常退出，再 `bash scripts/serve_eager_5090.sh "$NEW_RUN"` 启动。管理员 PowerShell 桥接继续运行，不需要重启。

新服务启动完成后，在 WSL 客户端终端运行（token仅环境变量）：

```bash
bash scripts/resource_model_5090.sh "$NEW_RUN" \
 "$PWD/runs/resource-model-$(date +%s%N)" --apply
```

脚本顺序：check → pilot/恢复 → train/恢复 → fit/冻结 → holdout/恢复 → validate。pilot只采一个2250/linear窗口，独立种子，不进入拟合。原始窗口、失败原因、恢复状态均留存。任何命令失败立即停止；首先检查该阶段 `collection.json` 的 error 和 restoration。恢复未 verified 时不得继续写频。

已知10秒试采在两卡预算分别5.2936%和5.2176%，最大采样间隔约126ms、查询不确定性小于4ms。以误差预算和平均功率近似不变估算，30秒约1.8%；这只是窗口设计依据，不承诺通过。3%门槛不变，旧失败数据不改状态。可先手动只执行 `resource_calibration pilot --run "$RUN" --telemetry configs/service_telemetry.template.json --out "$OUT" --window-seconds 30 --apply --ack-default-auto`。

`operator_equations` 默认改为显式 `operator_latency_budgets_s` + `budget_provenance`，缺预算拒绝选频。旧最快模式只能显式 `selection_policy=legacy_no_slowdown`。本交付不自动分配请求预算，不伪造图依赖或生成动态部署计划。

## 参考来源固定（2026-11-09）

独立实现，没有复制外部代码或采用外部GPU参数：

| 来源 | 固定提交 | 许可证核对 | 使用的方法 |
|---|---|---|---|
| https://github.com/USTC-ADA/EMO | 0acb0c2a6ae5165ff1753212cb3c7ff4042d17a9 | Apache-2.0 | 按资源分类、多频率、重复功耗窗口；不运行硬编码sm80脚本 |
| https://github.com/ekondis/gpuroofperf-toolkit | 2d611a2e2b422385716ae46e68e5843ac131cb26 | MIT | 硬件参数与目标kernel特征分离；不引入nvprof |
| https://github.com/hpc-ulisboa/gpupowermodel | babdfe88f3e58192fff14d843a2887534378d4cc | 仓库LICENSE含引用论文附加条款 | 多频域功耗作用范围审查；不引入其模型替代Eq2 |

论文定义仍以 https://arxiv.org/html/2603.21908v1 §4.1 为准。稀疏系数保持0，不将零值比例直接等同硬件跳算。
# 2026-11-10：中间频率采集修复

旧版资源采集复用了只允许2250/2527的控制器；2385首次写入前被拒绝，随后恢复检查对空目标计算，掩盖原错为 `NoneType + int`。新版本将本轮 probe 交集核实的频点显式传给控制器，默认旧接口限制不变；写入前拒绝不会执行无目标恢复检查，部分写入失败仍恢复双卡。仅客户端修改，无需重启 worker。

保留已完成的48窗训练和 frozen-model.json，仅在新目录补 holdout 后验证，不重新拟合。12点 sweep 也应使用此修复。采集间保持互斥，等待正在运行的采集结束后更新代码。软件回归通过不代表目标硬件验收。

