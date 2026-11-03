# 显式使用已有独立 Triton 3.6.0

适用于用户报告“普通 Qwen 启动需要 `/tmp/triton-360-test` 覆盖”的机器。保留现有 Conda/vLLM/torch，不执行全局 pip 替换，也不清除需要的 PYTHONPATH。这里只支持指定已有包，未在开发主机上执行真实 Triton kernel 或 Qwen 推理。

在目标 WSL Bash 中先更新 5090 分支，退出旧 vLLM 实验进程，再执行：

```bash
cd /mnt/d/s00958243/SparseDVFS
git fetch origin
git switch 5090
git merge --ff-only origin/5090
conda activate vllm-qwen
PY=/home/s00958243/miniforge3/envs/vllm-qwen/bin/python
test -f /tmp/triton-360-test/triton/__init__.py
```

建议在正式 baseline **之前**把已有包复制到持久目录，避免重启/临时目录清理后丢失。保留原包，不重新下载或安装，不修改当前 Conda：

```bash
TRITON_ROOT="$HOME/.local/opt/sparsedvfs-triton-3.6.0-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$(dirname "$TRITON_ROOT")"
cp -a /tmp/triton-360-test "$TRITON_ROOT"
export PYTHONPATH="$TRITON_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TRITON_CACHE_DIR="$HOME/.cache/sparsedvfs/triton-3.6.0"
mkdir -p "$TRITON_CACHE_DIR"
```

如果本次只先诊断现有临时包，可以暂用 `TRITON_ROOT=/tmp/triton-360-test`，仍传 `--triton-path "$TRITON_ROOT"`。开始正式实验前再迁移，并从 baseline 重新采集；迁移前后的 profile 不混用。

检查实际导入，预期版本 3.6.0，路径指向所选目录：

```bash
"$PY" -B - <<'PY'
import sys, triton, torch, vllm
print('Python:', sys.executable)
print('Triton:', triton.__version__, triton.__file__)
print('torch:', torch.__version__, 'CUDA:', torch.version.cuda)
print('vLLM:', vllm.__version__)
PY
```

设置新的实验目录和函数。后面的主指南命令都使用这个函数，不要再替换为不带 `--triton-path` 的旧定义：

```bash
MODEL=/mnt/d/models/Qwen3.5-27B-AWQ-4bit
GPU=GPU-22d1cd22-d198-45c4-2b54-974484026ab5
LIB="$PWD/build"
RUN="$PWD/runs/triton360-$(date +%Y%m%d-%H%M%S)"
BACKEND=nvml
mkdir -p "$RUN"
set -o pipefail
sd() {
  "$PY" -B -m rtx5090.conda "$@" \
    --model "$MODEL" --gpu-uuid "$GPU" --library-dir "$LIB" \
    --backend "$BACKEND" --triton-path "$TRITON_ROOT"
}
sd doctor --out "$RUN/doctor" 2>&1 | tee "$RUN/doctor.log"
```

`doctor.json` 的 `environment.packages.triton` 应包含实际 `version: 3.6.0`、所选 `path`、`package_sha256`；`distribution_version` 单独记录，不能用它代替实际导入版本。选错路径/版本直接拒绝。worker 与窗口子进程必须保持同一身份。

然后按 [主指南](CONDA_5090_zh.md) 顺序运行 cpu-test → build --cuda → probe → download/dataset → baseline → chain → calibrate --apply → profile → tune → compare。也可先执行 `sd smoke --out "$RUN/dummy"` 检查小模型原生接口，它不证明真实 Qwen 或节能。

新 Triton 配置必须使用新的 RUN：原 3.7.1 或其他路径下生成的 baseline、calibration、profile、plan 全部不能混用。比较两臂始终使用相同 Triton、缓存路径、模型与生成配置。源码/动态库/包内容/路径变化后重新采集。

Windows NVML bridge 只加载 Windows NVML，不需要在 Windows 安装 Triton。保持 WSL 的本函数及覆盖设置，按主指南切换 `BACKEND=bridge` 即可。

仍未取得目标机器的完整 3.7.1 traceback；因此本次改变是支持用户明确选择的运行配置，不声称已经定位上游缺陷。真实单卡 INT4、CUDAFUSE、硬件锁频和节能仍需实机分阶段验收。

Validation: [override tests](validation/triton_override/REPORT.md).
