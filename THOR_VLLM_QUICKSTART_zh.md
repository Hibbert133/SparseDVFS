# Thor 上快速复现 SparseDVFS v5

本指南是以后在 Thor 执行的步骤。本机交付阶段未安装软件、下载权重或镜像、改系统服务、写真实频控节点。v5 是独立参考移植，原论文 Orin/CNN/ViT 的数值尚未复现。

## 0. 固定配置和文件校验

目标：Jetson T5000 / 128GB、Jetson Linux R38.2、Qwen3.5-27B dense BF16、纯文本。模型固定为 `Qwen/Qwen3.5-27B`，revision `fc05daec18b0a78c049392ed2e771dde82bdf654`。

基础镜像：

```text
ghcr.io/nvidia-ai-iot/vllm@sha256:b587dd56b4cb076209ad5156a626ac75f5a976d0e8e7d1e6a9fccd56d1bd65e8
```

对应 tag 为 `0.19.0-r38.2.arm64-sbsa-cu130-24.04`；来源为 [NVIDIA 镜像发布页](https://github.com/orgs/nvidia-ai-iot/packages/container/package/vllm)。实际容器内容/BSP 兼容性仍要到机验证。不会自动升级为 latest 或切换模型/量化。

```bash
unzip sparsedvfs_thor_qwen35_demo_v5.zip
cd sparsedvfs_thor_qwen35_demo_v5
chmod +x thor_repro.sh
python3 -B tools/verify_package.py
python3 -B v5_demo.py --output runs/synthetic_v5.json
```

模型默认在 `models/Qwen3.5-27B`，结果在 `runs`。建议准备约 150–200GB 可用 NVMe 空间，实际需求以下载和镜像大小为准。所有命令可追加 `--results /absolute/results --model-dir /absolute/model`，后续保持相同路径。

v5 派生镜像 tag、worker protocol 和 FUSE ABI=5 均独立；不能复用 v4 二进制、prepare、profile、plan 或证书。

## 1. doctor：识别已有系统

```bash
sudo ./thor_repro.sh doctor
```

检查 `runs/doctor.json` 的型号、内存、BSP、Docker/runtime、磁盘、CPU/GPC/EMC、功率和电压观测节点。不会自动刷机、切换 nvpmodel、修改风扇或 governor。

计算域应是 `/sys/class/devfreq/gpu-gpc-0`，不要使用视频域 `gpu-nvd-0`。若 GPU 尚未初始化导致节点缺失，先完成 prepare/selftest/smoke，再重试 doctor。若仍缺失，按该机实际驱动/持久化状态排查；不要猜测其他域作为替代。

prepare 要求 R38.2、约 128GB 内存以及已有 Docker。缺依赖时手动参考 [Docker Ubuntu 安装](https://docs.docker.com/engine/install/ubuntu/) 与 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)，核对 JetPack/BSP 已有组件。脚本不改宿主机包源、Docker daemon 或驱动。

## 2. prepare：固定镜像及模型

```bash
sudo ./thor_repro.sh prepare
```

拉取固定 digest、构建 v5 镜像、只在镜像内编译两个 C++ 扩展；记录运行库版本并测试 BF16 matmul，然后下载固定 revision、核对 safetensors 分片和 tokenizer/配置哈希。已有权重也要通过模型锁检查。

通过条件：`runs/prepare.json` 的 `adapter_version=5`、正确 image ID/revision；`runs/container_runtime.json` 的 BF16 matmul 通过；模型目录有 `sparsedvfs_model_lock.json`。dispatcher/profiler 能力也会记录，CUDA profiler 不可用时算子证据路径不能通过，但层级路径仍可单独检查。

## 3. selftest：原生组件使用模拟节点

```bash
sudo ./thor_repro.sh selftest
```

容器不映射真实频控目录。日志在 `runs/selftest.log`，原生 C++/CUDA 测试只写临时模拟 sysfs 节点。本机测试记录中的 9 项跳过不能当作 Thor 通过；到机应实际编译并执行这些测试。

## 4. smoke：建立原生输出参考

```bash
sudo ./thor_repro.sh smoke
```

启动普通 eager vLLM worker，等健康接口、warm-up 两次、完成一次 512 输入/128 输出生成后退出。检查 `runs/scenario_512_128/smoke_latest.json`，须有真实 token IDs 和正确 usage。后续 trace 用这个输出 SHA256 做比对。

复现参数：BF16、`--language-model-only --enforce-eager`、TP=PP=DP=1、`max_model_len=4096`、`max_num_seqs=1`、`max_num_batched_tokens=4096`、显存比例 0.70。关闭 prefix caching、chunked prefill、async scheduling 和 speculative decoding；正常 KV/Mamba cache 保留。种子 0、greedy、固定长度且忽略 EOS。

OOM 时停止并查看真实 worker 的 `worker_failure.json`、server.log 和内存报告，不会自动更换量化或小模型。

## 5. module trace：先验证原有入口

```bash
sudo ./thor_repro.sh trace --granularity module
```

两次 warm-up 后连续三次请求；每一步须覆盖 embedding、64 层、norm、compute_logits、sample，共 68 段，输出与 smoke 一致。结果为 `runs/scenario_512_128/trace.json`。trace 禁止 `--apply`。

## 6. module 路径快速调频及五组对照

```bash
sudo ./thor_repro.sh profile --granularity module --grid quick --repeat 3 --apply
./thor_repro.sh plan --granularity module --selector measured --max-slowdown 0.05 --similarity-tolerance 0 --factor 5
sudo ./thor_repro.sh bench --granularity module --repeat 5 --apply
./thor_repro.sh report --granularity module
```

quick 固定当前合法 EMC，取两个 CPU × 两个 GPC 频点。完整定向切换按每对 10 次记录，包含源频点重新就绪；`--switch-repeat` 可显式调整。每个 OPP 的微基准及 warm-up 均在正式窗口之外。

这里的 5% 是示例用户时延预算，频率容差 0 表示精确相等；两者不是论文作者参数。N=5 是论文示例值。每区间预算不等于整请求 SLO。

五组是原生 eager、仅插桩、固定参考频率、serial、lookahead。若所有块都过短、只有一个块或目标不变化，可能没有有效提前请求；report 会标记 `not_exercised`，不会把它称为 look-ahead 已验收。不要为制造收益随意缩小真实切频开销。

## 7. operator trace：新增算子路径

```bash
sudo ./thor_repro.sh trace --granularity operator --backend torch
```

沿用步骤 4 的原生参考。三次请求均记录 dispatcher 调用、scheduler 上下文、所属模块、shape/dtype/stride；CPU launch 通过同线程作用域归属，再用 correlation ID 连接到 CUDA kernel/memcpy/memset 活动。融合单元不拆开。

通过后生成 `runs/scenario_512_128/operator/coverage.json` 和 `trace.json`，原始 profiler 文件留在相应 trace 子目录。重复 trace 会把旧 coverage 移为 obsolete；失败后不能继续使用旧证书。

漏归属 GPU 工作、多个 GPU stream/device、调用签名变化、缺少 correlation 数据或输出不一致都会停止。若某直接调用 Triton/CUDA 的融合路径未经过 dispatcher，不能把它忽略以通过检查；保留失败证据并继续使用 module 路径。

诊断 trace 可能较大且明显减速。这里不测节能率，也不把 profiler 时间当成正式 latency。

## 8. operator profile、plan、bench

```bash
sudo ./thor_repro.sh profile --granularity operator --grid quick --repeat 3 --apply
./thor_repro.sh plan --granularity operator --selector measured --max-slowdown 0.05 --similarity-tolerance 0 --factor 5
sudo ./thor_repro.sh bench --granularity operator --repeat 5 --apply
./thor_repro.sh report --granularity operator
```

profile 先验证初始证书，再在每个 OPP 下 warm-up 两次、做三次独立 profiler 诊断，确认同一布局和单流归属；最后关闭 profiler，执行 `--repeat` 指定的采样。计划必须携带所用 OPP 的覆盖证据。

每个 decode 上下文单独存表/分块，不把 127 个 decode 位置平均成同一算子成本。运行时调用原始算子一次，在块首/尾调用 Schedule/FUSE；未知上下文或签名先报错，不重放层，不清理正常 cache。

operator 与 module 的结果目录分开。各自用对应的仅插桩组衡量开销，不能把两种粒度的 profile/计划混用。算子观测、事件和 Python 开销可能很大，未承诺节能或加速。

## 9. full 三域实验

需要至少两个经过实机/BSP 核实的 EMC 频点。若 `possible_rates` 缺失，提供如下文件，并填真实数据：

```json
{
  "device_model": "与 hardware_host.json 完全一致",
  "jetson_release": "与 hardware_host.json 完全一致",
  "rates_hz": [],
  "verified_discrete_opps": true,
  "source": "该设备/BSP 的真实离散频表依据"
}
```

```bash
sudo ./thor_repro.sh profile --granularity operator --grid full --emc-evidence /absolute/emc_evidence.json --repeat 3 --apply
./thor_repro.sh plan --granularity operator --selector measured --max-slowdown 0.05 --similarity-tolerance 0 --factor 5
sudo ./thor_repro.sh bench --granularity operator --repeat 20 --apply
./thor_repro.sh report --granularity operator
```

full 会覆盖 quick 的当前 profile，旧计划失效，必须重建。没有合法 EMC 频表就停止 full，不能把 quick 叫作完整三域实验。2048/128 场景须从 smoke/trace 重新生成，所有命令追加 `--input-tokens 2048`，结果独立。

## 10. 严格公式路径：collect → 补真实证据 → fit → validate

measured 路径验证调度机制；严格 Eq. (1)/(2) 还需要真实标定。使用以下命令会重新采集 applied profile：

```bash
sudo ./thor_repro.sh calibrate --stage collect --granularity operator --grid full --emc-evidence /absolute/emc_evidence.json --sensor-config /absolute/sensors.json --repeat 3 --apply
```

生成 `observations.json` 和 `workloads.template.json`。采集包含 OPP 延迟/请求功率积分、只读电压/温度/频率记录及 GEMM/copy 容量估计。微基准额外使用约 0.5GB 缓冲和 GEMM 张量；OOM 会停止并恢复。

`sensors.json` 的结构如下，所有路径、单位、scale 和名字须按实际 Thor 验证，不能直接运行占位值：

```json
{
  "read_only_observations": [
    {"name":"verified_gpu_voltage", "kind":"voltage", "path":"/sys/实际电压节点", "scale_to_v":null, "evidence":"单位依据"}
  ],
  "calibration_scope": {
    "voltage_sensor":"verified_gpu_voltage", "frequency_domain":"gpc",
    "temperature_sensor":"实际 thermal type", "power_scope":"明确的电源轨范围",
    "scalar_vf_scope":"明确说明三域到标量 V/f 的选择及局限"
  }
}
```

collect 不自动生成单算子功率，`unit_records` 默认是空数组。必须另提供真实、同电源轨的独立单元测量，并说明测量方法；短算子通常需要独立重复测量，不能直接用低刷新率传感器给每次 kernel 分配功率。每个记录须有：

```json
{
  "unit":"严格使用 workloads.template.json 的键",
  "triplet":[0,0,0], "voltage_v":null, "effective_frequency_hz":null,
  "temperature_c":null, "latency_s":null, "power_w":null,
  "power_scope":"与 observations.power_scope 一致",
  "power_attribution":"isolated_unit_measured",
  "power_evidence":"原始测量文件、方法及范围",
  "window_duration_s":null, "sensor_interval_s":null
}
```

以上是格式而非数据。每个单元至少四个独立点，四分之一留出；仍须足够温度/V/f 变化使设计矩阵满秩。采样窗口至少覆盖两个真实 sensor interval。恒温下 k1/k2 不可辨识会停止。

workloads 中仅对已识别的 ATen mm 自动计算 `2*M*N*K`。`logical_tensor_bytes` 与 `data_byte` 分开；后者必须有真实 DRAM 流量或显式有效流量估计的依据。attention/GatedDeltaNet 等融合单元默认未知。不能用激活零比例填写有效稀疏度。

```bash
./thor_repro.sh calibrate --stage fit --granularity operator --observations /absolute/observations.json --workloads /absolute/workloads.json --max-relative-error 0.10
./thor_repro.sh calibrate --stage validate --granularity operator --observations /absolute/observations.json --calibration runs/scenario_512_128/operator/calibration.json --max-relative-error 0.10
./thor_repro.sh plan --granularity operator --selector paper --calibration runs/scenario_512_128/operator/calibration.json --max-slowdown 0.05 --similarity-tolerance 0 --factor 5
```

10% 只是这里显式选择的验证误差示例，不是论文指标。缺数据、拟合秩亏、留出误差过大、synthetic 输入或 profile 哈希变化均不得部署。validate 是额外重查，不会把未通过留出验证的标定升级为通过。

## 11. 服务与恢复

```bash
sudo ./thor_repro.sh serve --granularity operator --mode lookahead --apply
sudo ./thor_repro.sh serve --granularity module --mode lookahead --apply
sudo ./thor_repro.sh serve --profile production-native
```

前两个服务只接受已 profile 的 token IDs、固定输出长度及采样设置，适合机制验证；默认端口 8000，三个服务分别运行。production-native 使用普通优化 vLLM 配置，单独报告，不能混入 eager 节能比较。

容器 root 直接写窄范围映射的 CPU/GPC/EMC 目录；模型只读、结果可写、功率/电压节点只读。共享 flock 只约束使用该锁的控制进程；三域写入并非原子操作。未使用 privileged 或宿主机后台控制 daemon。

正常退出先停止调频线程/等待 CUDA，再恢复快照并检查。Ctrl+C 按同一路径关闭。SIGKILL、掉电或恢复失败时，确认控制进程已退出，再用对应快照：

```bash
sudo ./thor_repro.sh restore --snapshot /absolute/results/实际运行目录/worker/controller/clock_snapshot.json --apply
```

快照旁须保留匹配的 `mounts.json`，并使用创建它的 results 根目录。不要恢复另一场运行的快照。

## 12. 如何判定复现程度

依次确认：原生输出一致；worker 内真实覆盖；两种粒度身份/上下文正确；每个已使用 OPP 有证据；旧 ticket 不能覆盖新块；CPU boost/release 与块边界存在；look-ahead 确实发出资源请求；恢复读回成功。

report 分别给出客户端 TTFT/TPOT、worker 区间和、请求 J/EPT、频率读回、no-op/过期/错过请求及回调耗时。INA238 VIN 表示模块+载板整体，不与模块子轨相加。20ms 轮询不代表传感器有 20ms 刷新率；PTO 原始读数与软件请求另列。[Thor 官方接口](https://docs.nvidia.com/jetson/archives/r38.2/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonThor.html)

节能可以为零或负值。请求已写入、CUDA end 尚未完成、软件频率匹配，都不能单独证明物理 PLL 切换开销被隐藏；完整结论仍需 Thor 实测。作者寄存器队列、原始 Orin 实验数值与完整 kernel DAG 均未复现。
