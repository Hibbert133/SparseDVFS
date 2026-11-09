# 双卡切换与分块尺度离线检查

2026-11-09：`python -B -m rtx5090.switch_partition_screen` 仅读取现有
`report.json`、Nsight SQLite 和冻结阶段 `stage-freeze.json`，不连接服务、bridge 或 GPU。
Windows Python 3.8 和 WSL 均可执行，不需要安装依赖。

参数：`--capture` 指向已完成且恢复通过的常驻桥接采集目录；`--stage` 指向
冻结阶段目录；`--value-unit hz --unsigned32` 显式选择本轮候选解码；`--out`
必须是新目录。可选 `--hypothetical-request-s` 只用于假设的 5% 切换预算，
不是本次实测请求延迟。输入文件均记录 SHA256，原始数据不修改。

## 语义与实现选择

- GPU metrics 按 typeId 低八位映射到 TARGET_INFO_GPU，再核对 UUID；不能沿用
  nvidia-smi GPU 索引。映射说明：
  https://docs.nvidia.com/nsight-systems/AnalysisGuide/
- 负整数按 uint32 解释与 Hz 单位仍明确标为候选，未声称单位校验完成。
  NVIDIA 对有符号导出问题的说明：
  https://forums.developer.nvidia.com/t/nsight-systems-gpc-frequency-profiling/360987
- 使用已保存的墙钟/QPC 锚点对齐 Nsight session 时间，输出偏移变化；此范围不代表
  完整同步误差上界，不能做亚毫秒精度声明。
- 每卡写频前 200ms 为前平台，后 2–4s 为后平台。中点连续 2ms；后平台
  p10/p90 加减 15MHz 连续 10ms 为诊断规则。间隙超过 3ms 会打断连续性。
  这些都是移植侧诊断选择，不是论文稳定判据，不含活动率门槛。
- 双卡时间统一从该轮第一张卡写频之前的时间戳起算，以两卡事件最大绝对时间结束。
- SparseDVFS Algorithm 1 / Eq.(3) 的 N=5 沿用项目选择，严格 `T_block > N*T_switch`。
  5/10/20/50ms 是敏感性假设，不是由六个样本认证的稳定上界。
- 冻结 linear 表只用于同形状连续调用的尺度估算；不外推完整64层，不按请求板级
  能量分摊算子能量，不补造2250/2527的逐模块成本，不据此产生在线计划。
- 假设请求时长下的5%预算只计算切换次数上限，排除了计算减速、协调、初始设置和恢复；
  不是完整延迟约束验收。

输出 `audit.json` 和 `REPORT.txt`。保留 `deployable:false`、候选解码、未验证
capture/service 身份绑定及缺失成本。软件 SQLite fixture 测试不代表实机验收。

下一阶段输入仍需同服务、两个频点的逐执行单元或整块延迟/能量，以及实际双rank合法边界。
在这些输入完整之前，现有 `partition_dryrun` 不能发出可执行硬件计划。

## 复用常驻 worker 补两频点 linear 响应

`python -B -m rtx5090.linear_frequency_probe --run RUN --telemetry CONFIG --out OUT`
默认只读；实际采集必须另外提供 `--apply --ack-default-auto`。此命令在 WSL 执行。
先获取服务 client.lock，实时检查两个 worker 的身份和 matrix_replay_version，
不更改 worker、不重启服务。采用已有 matrix_replay（不启用 kernel profiler），
保留每个频点三个独立请求、每请求每种形状三批 N32 的全部结果和读回。
两频点顺序按重复轮换，正式模型拟合/独立验证另行冻结协议，当前输出仅为响应诊断。

每次设置后运行三次512/64请求预调节，再采一条真实张量回放请求；预调节不是精确锁频
验收，也不把本轮GPC观测转换为通用等待上界。成功和异常都尝试恢复，并由三条带载请求
核实。失败保留数据，禁止丢样本拟合。bridge EOF恢复是补充保护；强杀或断电不保证验证。

范围仍仅第一处MLP的prefill/decode gate_up/down linear；不是完整64层成本。
kernel匹配和缓存状态不会因运行此命令而变成已通过。遥测只描述当时的频率/温度等上下文，
不输出算子能量，不把插桩请求耗时作为正式服务基线。
