# Conda / vLLM 0.32 更新验证报告

验证日期：2026-11-03。**没有实测 5090 的节能或加速结论。**

实现提交：`2334cb65a76a7585b97472260e9ec4dabf57af6a`。兼容性基础另提交为 `25d1fc33c7e3b68a0d472df360058f0690aee0a1`。后续报告/打包提交不改变运行实现。

## 实际执行结果

| 环境 | 实际执行内容 | 结果 |
|---|---|---|
| 隔离 QEMU Linux x86_64，Python 3.12.16，g++ 14.2 | 真实 C++ 编译、调度/桥接 mock、Python 执行链与新增统计/拒绝测试 | 188 项：185 通过、0 失败、3 CUDA 跳过 |
| GitHub Ubuntu 24.04 runner，Python 3.12.15，GCC 13.3 | 最终提交的 Conda `cpu-test` 原生入口、真实 C++ 构建与完整回归 | 188 项：185 通过、0 失败、3 CUDA 跳过 |
| 本地 Windows 实现主机，Python 3.14.5 | Python/mock 回归、入口解析、诊断命令 | 188 项：173 通过；12 Linux native + 3 CUDA 跳过 |
| 用户 WSL2 / RTX 5090 | 未连接到该机器，未运行实机实验 | 待用户执行手动指南 |

GitHub CI：[run 41569581167，success](https://github.com/Hibbert133/SparseDVFS/actions/runs/41569581167)。CI 实际运行 `python -B -m rtx5090.conda cpu-test --out runs/ci`，并验证必需 CPU/native 项未跳过。

CI 日志与源码哈希在 [ci/tests.txt](ci/tests.txt)、[ci/summary.json](ci/summary.json)。其中包含 **12 项真实 C++ integration 测试**。已逐个核对 summary 中的源码 SHA256，全部与交付实现一致。GitHub artifact SHA256：`88f3640bfdb65bf7d4c881fe6bf572d2579f26f11136f6c8bff25d5d028b477f`。

QEMU 日志在 [qemu/tests.txt](qemu/tests.txt)、[qemu/summary.json](qemu/summary.json) 和 [编译日志](qemu/compiler.txt)。QEMU 快照比最终实现少两处收尾修改：Conda CPU 入口隐藏 CUDA 设备/检查跳过项，以及单独保存 measured trace。最终两处修改已由上述 Linux CI 验证。QEMU 运行真实 FUSE C++，CUDA、模型与 NVML 是替身；没有 GPU 直通。

原验证 runner 的 `date` 字段沿用历史基线日期；本次实际执行时间以 CI run 时间及本报告为准。没有改写旧四个 `validation/` 结果文件。原有 C++ misleading-indentation/unused-function warning 仍保留在原始编译日志中，不被当作真实硬件验收。

## 验证矩阵

| 检查 | 无 GPU 证据 | 尚需目标机器证据 |
|---|---|---|
| vLLM 0.32 V1 调用链 | 用户源文件 SHA 与上游匹配；签名/顺序审计、mock execute/sample 闭合 | 实际安装包启动、64 层 hooks、量化 kernel |
| 路径/环境/GPU 选择 | 模型/tokenizer 参数、子进程 argv、UUID 拒绝、临时 Triton 拒绝、源文件变更拒绝测试 | worker 实际 UUID、包加载路径、显存 |
| C++ 调度器 | 真实编译，队列/ticket/取消/去重/过期/错误/恢复通过 | CUDA host callback 与 GPU 执行时序 |
| 桥接 | 持久 socket、认证/序号、独占写入、独立遥测、断连/失败路径通过 | Windows NVML 权限、WSL 可达性、真实开销 |
| 权重身份 | 缺锁/未量化/缺失路径拒绝，保持锁定 revision 校验 | 完整 INT4 权重加载与输出哈希 |
| 分块 | 复用 Algorithm 1；顺序覆盖、短块合并、GPU-only 向量验证 | 真实 profile 与切换开销 |
| 能量计量 | mJ→J、计数器回退/重置、30s 窗口、功率证据门槛测试 | NVML 能量计数器或有效功率精度校验 |
| 效果验收 | 正/零/负收益统计算例、延迟超限、dummy/mock/训练集/输出/身份/恢复不合格拒绝 | 冻结方案的至少 10 组真实配对实验 |

统计单元测试使用人为构造数据验证判定器；其中“正收益通过”的断言**不是 5090 测量结果**。没有生成可冒充目标机性能报告的虚拟产物。

## 改动与限制

- 支持当前 Conda，明确 V1，保留真实模型、输出、覆盖与配置检查；没有全局包替换、网络/驱动改动或 Docker 依赖。
- 真实写频率与 reset 需要显式 `--apply`。恢复失败不发布可用标定；已存在证据不静默覆盖。
- 默认系统自动频率作为原生 worker 对照，SparseDVFS 插桩/调频开销进入请求测量。
- 能量方案在训练输入上选择，测试输入只用于冻结方案的配对验收；置信区间与误差余量均参与判定。
- module 级、GPU-only；没有实现完整论文稀疏感知功耗模型、kernel DAG、多频域联合控制和整机能耗。

实际手动流程、预期输出、失败判断和恢复命令：[CONDA_5090_zh.md](../../CONDA_5090_zh.md)。如果实机无法锁频、无法可靠计量，或未达到能耗下降且平均延迟增幅 ≤5%，结论必须为未复现或无法判定。
