# 独立微基准采集器（条件性Eq1，不是同路径认证）

2026-11-06。当前驻留worker只支持固定真实张量回放，没有可热插入的独立容量微基准。
本入口保留服务，在单独短生命周期CUDA子进程中逐卡测量；不重启vLLM，不调频，不安装依赖。
显存检查创建上下文前至少1536MiB；上下文后CUDA和NVML均至少1024MiB。不通过则保存blocked状态，不尝试分配测试张量。
原来GPU1只有384MiB，因此预计可能只取得rank0；这不是双卡标定完成。

## 参数与证据

- launch_proxy：一个BF16元素的copy_，仅作提交/小拷贝开销代理，不是纯系统固定开销的独立真值。
- gemm_train：3072方阵BF16 mm，逻辑2N^3 FLOP。
- copy_train：64MiB输入和输出，计读+写逻辑字节量。
- 每个微基准5次预热、3批各32次；CUDA Event批时间含提交空隙，另存host时间。
- P=W/(训练GEMM中位时间-h)，B=D/(训练copy中位时间-h)。h来自launch_proxy。
- 参数保存后再测2048方阵和96MiB拷贝作为primitive_holdout，再读已有MLP时间。
- MLP_transfer_diagnostic只表示跨算子迁移误差；不是同kernel、同缓存、同频率验证。
- 记录批前后实际频率、温度和显存；未证明批内频率恒定，不做频率插值。
- 各项均为本项目估计器选择，不是论文作者的明确标定流程。
- 不输出Eq2参数：独立算子功耗窗口及可辨识热回归仍待实现。
- 不将该结果自动填入正式eager_model，也不能授权分块/调频。

这些微基准使用合成张量，不能称为Qwen同路径容量。数据用于检验简单容量移植是否成立。
不根据迁移误差改参数、不删批次或重试。真正同路径标定仍需扩展worker并重新加载，当前入口并未完成该部分。

## 目标机WSL Bash

先更新5090_eager；保留现有服务、不同时发送其他请求。用当前服务重新inspect保存identity，不能用旧进程身份。

```bash
conda activate vllm-qwen
cd /mnt/d/s00958243/SparseDVFS
export PYTHONPATH="/mnt/d/s00958243/triton-360:${PYTHONPATH:-}"
export SPARSEDVFS_TRITON_PATH=/mnt/d/s00958243/triton-360
export TRITON_CACHE_DIR=/tmp/triton360-vllm-cache
RUN="$PWD/runs/eager-replay-1793940606903409051"
REPLAY="$RUN/matrix-replay-1793946767333088898"
OUT="$RUN/independent-capacity-$(date +%s%N)"
PY=/home/s00958243/miniforge3/envs/vllm-qwen/bin/python
"$PY" -B -m rtx5090.eager inspect --run "$RUN" --out "$OUT/identity.json"
"$PY" -B -m rtx5090.independent_capacity --identity "$OUT/identity.json" \
  --replay "$REPLAY" --out "$OUT/collection"
```

成功的rank输出parameters、primitive_holdout和MLP_transfer_diagnostic；blocked保留具体显存数据。
collection.json汇总所有rank，rank-*/measurement.json保存原始三批时间和前后读数，parameters.json是在查看holdout前冻结的参数。
失败保留worker.log；不要自动停止服务释放显存。50项相关CPU检查通过，实际CUDA采集仍待目标机验证。
