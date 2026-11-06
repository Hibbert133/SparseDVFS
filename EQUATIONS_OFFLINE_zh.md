# 5090_eager：Eq.(1)/(2) 离线预测

## 2026-11-06：Windows NVAPI带载电压采样

目标机已报告只读接口0x465F9BCF成功：PCI 02为0.935 V，PCI 83为0.920 V。
这仅证明单点查询成功，不代表电压曲线、传感器周期或Eq2参数已验证。

`rtx5090.host_load_check --voltage`现在可在已有20请求负载期间采集电压。
WSL客户端自动启动Windows Python采样器并等待ready；不用手动协调两个终端。
Windows侧通过nvidia-smi的UUID/PCI清单绑定NVAPI bus ID，重复总线或缺卡拒绝。
每条NVML样本附加voltage字段，包含原始微伏、换算伏值、返回码、绑定关系和独立QPC查询起止。
NVML与NVAPI按顺序查询，并非原子同时测量。采样帧包含请求前后边界，不称作精确算子窗口。
电压错误、零值、丢失或查询间隙超350ms均不通过；不放宽原NVML间隙门槛。
电压保持常数也可能是有效读数，但不能据此推断传感器更新周期。

在目标机WSL Bash（普通用户）执行：

```bash
(
set -e
cd /mnt/d/s00958243/SparseDVFS
export PYTHONPATH="/mnt/d/s00958243/triton-360:${PYTHONPATH:-}"
export SPARSEDVFS_TRITON_PATH=/mnt/d/s00958243/triton-360
export TRITON_CACHE_DIR=/tmp/triton360-vllm-cache
RUN="$PWD/runs/eager-1793872005954254496"
OUT="$RUN/host-voltage-$(date +%s%N)"
echo "本次输出：$OUT"
/home/s00958243/miniforge3/envs/vllm-qwen/bin/python -B \
  -m rtx5090.host_load_check --voltage \
  --identity "$RUN/check-1793876461713035188/identity.json" --out "$OUT"
)
```

预期`loaded_host_sampling_passed`，各UUID含voltage的range_v、distinct_values、max_query_ms、max_gap_ms。
预计一个20请求窗口约95秒，另有预热；不重启服务、不写频、不安装依赖。
产物为voltage-bindings.json、windows-samples.json、windows-summary.json、diagnostic.json。
失败保留原始数据及windows-stderr.log；不重试、不把成功的部分拼成通过记录。
本次新增及相关56项CPU检查通过；未在开发机宣称完成目标Windows带载采样。

来源：LibreHardwareMonitor `LibreHardwareMonitorLib/Interop/NvApi.cs` 中
NvGpuClientVoltRailsStatus（76字节、版本1、0x28/0x2c电压字段）及NvidiaGpu.cs读取路径。
该私有NVAPI接口不具备公开稳定ABI保证；不调用任何电压或频率setter。

2026-11-06 审计范围：论文 v1 §4.1.1–4.1.2，八组双rank MLP输入。
原文：https://arxiv.org/html/2603.21908v1#S4.SS1

## 公式与代码

- Eq.(1)：`max(W*(1-Sc)/P, D*(1-Sm)/B) + overhead`。
  `sparsedvfs/paper_core.py:timeline_seconds` 算术一致；单位为 FLOP、byte、FLOP/s、byte/s、s。
- Eq.(2)：`alpha*V**2*f + (k1*T+k2)*V`。
  `sparsedvfs/paper_core.py:thermal_power_w` 算术一致；f 为 Hz，不是 MHz。
  论文式中温度项是线性的；不能因正文提及 super-linear 而擅自加入平方或指数项。
- `rtx5090.eager_model.predict_latency/predict_power` 分离两式；已有 `predict` 返回值兼容。
- `rtx5090.equation_predict` 为不控制硬件的独立入口。输出始终禁止授权分块/调频。

## 明确差异与缺项

1. 论文P依赖CPU/GPU、B依赖EMC；这里每行显式输入一个运行条件的P/B，是GPU投影。CPU/EMC未复现。
2. 论文未完整给出微基准和alpha(S)函数。原拟合器按单元拟合alpha、按rank共享k1/k2，是实现选择。
3. BF16路径没有跳零证据，Sc=Sm=0。alpha只能对应该稀疏条件，不能推断alpha随稀疏性变化。
4. 张量逻辑bytes不能自动当DRAM流量。模板中data_byte为空；如做逻辑流量假设，必须填写data_semantics并标为条件性预测。
5. 容量/带宽/开销、V/f、温度、热系数均不填硬件规格或历史失败标定的默认值。parameter_source说明来源；字段存在本身不构成验收。
6. 功率作用域必须填写power_scope。单元模型能量为P*T，不是实测算子能量，也不求和冒充TP请求能量。
7. 现有正式fit/validate/plan的两频点、恢复、同路径、独立验证门槛未放宽。

## 在目标机 WSL Bash 运行（普通用户，无需管理员）

以下纯CPU离线操作，不请求模型、不分配GPU、不写频、不重启服务。

```bash
(
set -e
cd /mnt/d/s00958243/SparseDVFS
PY=/home/s00958243/miniforge3/envs/vllm-qwen/bin/python
TRACE="$PWD/runs/eager-1793872005954254496/trace-512-1793876976880886186"
OUT="$PWD/runs/eager-1793872005954254496/equations-offline-$(date +%s%N)"
"$PY" -B -m rtx5090.matrix_workloads --trace-dir "$TRACE" --out "$OUT/workload-input"
"$PY" -B -m rtx5090.equation_predict prepare \
  --workloads "$OUT/workload-input/workloads.json" --out "$OUT/real-inputs"
"$PY" -B -m rtx5090.equation_predict demo --out "$OUT/arithmetic-demo"
printf '\n本次输出：%s\n' "$OUT"
)
```

预期：八组真实算子输出`parameters_missing`及缺项；这表示公式输入尚不齐全，不表示采样失败。
示例输出`illustrative_predictions`，手算为11秒、6瓦、66焦耳。该数字故意采用算术夹具，绝不是5090预测。
失败时保留目录；入口不会自动重试。`failure.json`记录参数错误；原始trace不修改。

## 填参后计算

复制 `real-inputs/parameters.json` 为单独文件修改，保留原模板。
每行Eq1需data_byte、peak_flops、bandwidth_Bps、overhead_s及data_semantics、capacity_condition、parameter_source。
每行Eq2需alpha、voltage_v、frequency_hz、k1、k2、temperature_c及power_scope、parameter_source。
不改workload、identity_fingerprint和workloads_sha256。不同频率条件使用独立参数文件；不进行隐式插值。

```bash
# WSL Bash；OUT设为上一步实际目录，参数文件应由有依据的数据填写。
"/home/s00958243/miniforge3/envs/vllm-qwen/bin/python" -B \
  -m rtx5090.equation_predict predict \
  --parameters "$OUT/parameters-filled.json" \
  --workloads "$OUT/real-inputs/workloads.json" \
  --out "$OUT/predict-$(date +%s%N)"
```

仅Eq1齐全会输出延迟而功率/能量为null；仅Eq2齐全会输出功率而延迟/能量为null。
两式齐全输出`conditional_predictions_unvalidated`，仍须独立实测验证才能谈参数有效。

## 本次软件检查

69项CPU测试通过（equation_predict、eager、v4_paper、matrix_workloads）。
包含手算、Hz单位、缺参数独立运行、NaN/无穷/负值拒绝、负漏电拒绝、双rank和输入哈希绑定。
真实5090参数标定、同路径回放、独立误差验收及控制硬件均未由这些检查证明。
