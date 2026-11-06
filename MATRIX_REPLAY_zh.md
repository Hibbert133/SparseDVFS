# 常驻worker内真实MLP张量回放计时

## worker v2：kernel核对与批次数采集

新增 `observe --matrix-calibrate`，需要worker的matrix_replay_version>=2。
当前v1进程已经缓存旧MatrixReplay类，git更新不能热替换；需按本文check-only→用户正常停止→新目录启动流程加载。
无需外部CUDA进程或复制权重，但短profiler会额外占用主机/GPU资源，目标机尚需验证。

- 分别对首个真实linear调用和同X/W回放抓取短CPU/CUDA trace，调用前后全设备同步以隔离待执行工作。
- 保存 `rank-R-PHASE-MODULE-original.json` 与 `...-replay.json`，只提取真实kernel活动。
- 拒绝缺失kernel、未知或多GPU/多stream活动；比较名称、数量、时间顺序及可用的grid/block。
- 这是受同步/profiler干预的原始调用，不声称自然请求的缓存、重叠行为已完全复现。
- 关闭profiler后，交错测量1/8/32/128次调用，每档3批。参数拟合只用1/32/128档，8档作为批次数留出。
- 拟合 `batch_time = intercept + repeats*slope` 是计时摊销诊断。intercept不是Eq1的系统开销真值，slope不是独立容量参数。
- kernel_mapping_passed只代表名称/数量/顺序一致；kernel_launch_geometry_passed单列；任何一个false都不应宣称完整路径核实。
- Eq1独立P/B和Eq2功耗参数仍未完成，calibration_ready=false保留；此入口先补同路径采集的必要证据，不伪造拟合。
- 57项相关CPU测试通过；尚未在目标机验证CUDA profiler、峰值显存、vLLM双rank行为。

新服务inspect后，用以下命令代替本文旧的matrix-replay命令：

```bash
"$PY" -B -m rtx5090.eager observe --matrix-calibrate \
  --identity "$CHECK/identity.json" --seed 11062301 \
  --out "$RUN/matrix-calibrate-$(date +%s%N)"
```

主输出仅显示matrix_summary；完整张量元数据和各批时间保存在observation.json及各rank-matrix-replay.json。
失败保留trace与rank失败证据，不自动重启、降级profiler或删批重试。

2026-11-06；仅5090_eager。入口是 `rtx5090.eager observe --matrix-replay`。
新增worker方法必须重新加载服务；旧进程不会因git更新热替换。
客户端在预热/请求前检查两rank的matrix_replay_version，旧worker拒绝执行。

## 已实现的最小范围

- 仅真实BF16、TP2、512输入64输出、eager的aten.linear.default。
- 每rank捕获prefill/decode首个gate_up_proj/down_proj，共4个case，记录实际模块路径。
- 原始linear先正常执行并保留返回值；复用该调用的真实X/W，各重复3批，每批32次。
- CUDA Event记录批耗时及均值；包含主机提交造成的GPU空隙，不声称纯kernel延迟。
- 不复制权重、不创建额外CUDA进程、不新增collective、不改变频率。额外输出/库workspace仍需显存，不能保证不会OOM。
- profiler关闭；请求的端到端耗时被测量干预，不能作为正式性能基线。
- 三批共享同一输入，不是独立训练/验证集；缓存已被重复访问，kernel名称未重新核实。
- 不从本次耗时反算P再预测同一耗时；不将请求板级能量分配给算子。
- 返回的是参数采集前的真实张量回放计时证据，不是Eq1/Eq2标定完成。

## 1. 旧服务仍运行时：WSL Bash检查

```bash
conda activate vllm-qwen
cd /mnt/d/s00958243/SparseDVFS
export NEW_RUN="$PWD/runs/eager-replay-$(date +%s%N)"
bash scripts/serve_eager_5090.sh --check-only "$NEW_RUN"
echo "新服务目录：$NEW_RUN"
```

预期preflight_ready_not_loaded_or_clock_verified。保留NEW_RUN。
检查失败先处理，不停止旧服务；不要更新代码后复用旧preflight。

## 2. 用户同意额外一次重启后

由用户在原服务终端正常Ctrl+C，确认其正常退出。在上一步WSL终端运行：

```bash
bash scripts/serve_eager_5090.sh "$NEW_RUN"
```

仅此一次启动尝试；失败保留startup.log，不自动重启。真实权重、TP2、eager、显存比例0.90和成功环境沿用原启动器。
此文档不授权停止其他用户进程。旧数据和旧identity保留，不能拿旧identity用于新worker。

## 3. 新服务ready后，在另一WSL Bash终端

把RUN赋值为步骤1实际输出的新服务绝对路径。

```bash
conda activate vllm-qwen
cd /mnt/d/s00958243/SparseDVFS
export PYTHONPATH="/mnt/d/s00958243/triton-360:${PYTHONPATH:-}"
export SPARSEDVFS_TRITON_PATH=/mnt/d/s00958243/triton-360
export TRITON_CACHE_DIR=/tmp/triton360-vllm-cache
# RUN=上一步打印的新服务目录，不能填旧eager目录
test -n "$RUN" && test -f "$RUN/launch.json" || exit 1
PY=/home/s00958243/miniforge3/envs/vllm-qwen/bin/python
CHECK="$RUN/check-$(date +%s%N)"
"$PY" -B -m rtx5090.eager inspect --run "$RUN" --out "$CHECK/identity.json"
"$PY" -B -m rtx5090.eager observe --matrix-replay \
  --identity "$CHECK/identity.json" --seed 11061201 \
  --out "$RUN/matrix-replay-$(date +%s%N)"
```

预期observation.json为complete且coverage.passed=true；各rank-matrix-replay.json含四组、每组三批时间。
cache_state_matched、kernel_names_verified、calibration_ready保持false，不视为执行失败。
若worker失败会保留rank-*-matrix-replay-failed.json与服务失败记录；先检查，不继续向失败worker发采样请求。

## 开发侧检查

56项相关CPU测试通过；覆盖四形状、布局/bias拒绝、重用参数、保持原始输出、重复计数、缺批和无效计时。
实际vLLM/CUDA dispatcher路径、显存峰值、双rank负载影响仍需目标机运行验证。
