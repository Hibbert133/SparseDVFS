# 作者 ViT 脚本：RTX 5090 单文件运行

本交付位于 [Hibbert133/SparseDVFS 的 `5090-vit` 分支](https://github.com/Hibbert133/SparseDVFS/tree/5090-vit)，直接从作者 [c524c51](https://github.com/ziyangzhang96/SparseDVFS/commit/c524c51aac552fd16bec5743b1fb95309b46347b) 建立，保留原 `SparseDVFS.py`，只新增 ViT 的 5090 脚本、说明和测试。它不继承原 `5090` 分支中的 Qwen/vLLM 移植历史。

`SparseDVFS_5090.py` 可单独复制执行，不导入本项目任何包；不需要桥接服务、token、端口、Windows Python 或 pynvml。依赖仍是作者使用的 torch、torchvision、numpy、pandas、matplotlib、seaborn、tqdm。缺失或不兼容时报告真实错误，不自动安装/升级。

## 1. 更新代码，进入现有环境

**Windows 管理员 PowerShell：**进入 WSL，传递宿主机管理权限。WSL 内 sudo 不能替代 Windows 管理员权限。

~~~powershell
wsl
~~~

**WSL Bash：**首次使用时克隆到独立目录，保留原 Qwen/vLLM 仓库与工作区。

~~~bash
cd /mnt/d/s00958243
git clone --single-branch --branch 5090-vit https://github.com/Hibbert133/SparseDVFS.git SparseDVFS-vit
cd /mnt/d/s00958243/SparseDVFS-vit
conda activate vllm-qwen
~~~

已有 `SparseDVFS-vit` 独立目录时，不重复 clone；先在该目录确认工作区干净、分支为 `5090-vit`，再运行 `git fetch origin` 和 `git merge --ff-only origin/5090-vit`。遇到其他分支或用户修改时先处理，不覆盖。

保留现有 Triton/PYTHONPATH 等环境变量。正式采样需要所选 GPU 空闲；若 Qwen TP=2 服务仍占用该卡，先由用户安排停止自己的任务，脚本不会停止服务。

## 2. 只读预检

**WSL Bash：**

~~~bash
python SparseDVFS_5090.py \
  --uuid GPU-22d1cd22-d198-45c4-2b54-974484026ab5 \
  --check-only
~~~

预期打印 `PRECHECK PASS` 与独立输出目录。检查解释器、依赖版本/路径、CUDA UUID、Windows 工具、板级功率、各显存档上报的 GPU 频点并集、Windows 管理权限及原作者权重缓存。导入 torch 前按 UUID 限定唯一可见卡，再用 CUDA 返回的 UUID 核对。

不写频、不 reset、不下载或加载模型。`preflight.json` 中 `set_reset_verified:false` 是正常结果；查询成功不能证明实际锁频权限。`windows_admin:false` 时只读预检仍可通过，正式运行会停止，需从管理员 Windows 终端重新进入 WSL。

若找不到工具：确认已有 `nvidia-smi.exe`、`powershell.exe` 可在 WSL PATH 找到。脚本也检查 `/mnt/c/Windows/System32` 及其 `WindowsPowerShell/v1.0` 子目录。找不到、不支持、N/A、UUID 不符或导入错误均停止并保留输出，不退回 Linux 版 nvidia-smi、mock 或固定功率。

## 3. 正式运行

先确认所选 GPU 初始为默认自动频率模式；`--ack-default-auto` 确认此前提，因为此接口不能保证读取并还原任意已有锁频配置。实际写频必须显式加 `--apply`。

**WSL Bash：**

~~~bash
python SparseDVFS_5090.py \
  --uuid GPU-22d1cd22-d198-45c4-2b54-974484026ab5 \
  --apply --ack-default-auto
~~~

第二张卡可显式改为 `GPU-e2f02606-bac6-52cd-f8a1-15a027db7da0`；不会同时调两张卡。不指定 `--uuid` 时默认第一张已记录 UUID。可加 `--out /新的/输出目录`；已有目录会被拒绝，避免覆盖原始数据。

正式运行允许 torchvision 按原方式下载缺少的 `IMAGENET1K_V1` 权重。下载失败直接报告，不改用随机/dummy 权重或其他模型版本。只读预检不会触发下载。

预期逐层遍历**全部上报候选频点**，有效单元逐条落盘，失败单独留痕。总单元数为作者选中层数 × 候选频点数，可能较长。上报不代表本次负载能达到该频点：沿用作者两次 0.5 秒稳定等待及 5 次预热，预热后查询实际频率，超过目标 ±15 MHz 时记录失败并跳过；不重试或删改原始样本。驱动/权限/采样错误立即终止并恢复。

## 与作者代码的准确关系

- 原样保留 `calculate_sparsity`、`get_layer_input_tensor`、`get_all_layers`，以及原稀疏度绘图与样式函数。选层规则不补选 attention、norm 等未被作者名称规则匹配的层，也不将父子模块去重。
- 保留 `vit_b_16(IMAGENET1K_V1).cuda().eval()`、默认精度、随机 `(1,3,224,224)` 输入、作者梯度开关范围、5 次预热、5 次同步 `perf_counter` 测量和“平均功率 × 平均延迟”。不引入 autocast、compile 或新的推理执行模式。
- CPU/显存由系统管理；移除 Jetson CPU/EMC 写入。CSV 的不可读 CPU 频率留空；热图采用“CPU system-managed”单列，跳过 CPU 扫频图。保留稀疏度分布、稀疏度/性能、逐层 GPU 延迟/板级功率及延迟/功率热图。
- 全程使用一个 `nvidia-smi.exe --loop-ms=20` 查询进程。脚本用系统 PowerShell 的 `Start-Process -WindowStyle Hidden` 启动、取得 Windows PID，结束只清理命令行含本次唯一 CSV 路径的采样进程，避免只结束 WSL 中继或误停别人的进程。用户只运行一个 Python 入口。
- 每条采样记录 Windows 工具时间戳和 WSL 收到该行的单调时刻。均值仍取作者监视窗口（含测量前 0.1 秒）中的有效样本。20 ms 是查询请求间隔；实际查询、文件刷新和跨系统传递有开销，未证明传感器有 50 Hz 独立更新，未做跨系统时钟精度标定。
- 能量是**单卡板级平均功率 × 平均层延迟的估计**，不是整机能量或独立仪器验证的孤立算子能量；保留作者估计器，未改成长窗口或累计计数器方法。
- 增加启动前约 1 秒、reset 后约 1 秒的全模型带载读回，位于所有层级测量之外，仅用于恢复证据；每单元仍为 5 次预热/5 次正式测量。±15 MHz 和恢复检查是移植选择，不是作者论文参数。

## 输出与失败处理

| 文件 | 内容 |
| --- | --- |
| `preflight.json`、`target_layers.json` | 当前环境、UUID、候选频点、缓存与作者选中层 |
| `profiling_results_ViT-Base.csv` | 逐单元追加的有效结果；含原指标、目标/实际 GPU 频率、显存频率、UUID 与口径 |
| `power_samples.csv`、`power_samples.jsonl` | 原始 CSV 与带 WSL 收到时刻的采样 |
| `clock_readbacks.jsonl`、`measurement_windows.jsonl` | 预热后读回、5 次延迟、功率均值对应的采样索引 |
| `failures.jsonl`、`commands.jsonl`、`dvfs_profiler.log` | 失效单元、真实命令返回码/输出和运行日志 |
| `restore_status.json`、`power_sampler_cleanup.json`、`status.json` | 恢复、Windows 采样进程清理及最终状态 |

正常结束、Python 异常、Ctrl+C、SIGTERM 均进入统一清理：先恢复 GPU、记录带载读回并停止采样，再绘图。强制杀进程/断电不能依赖 Python finally。

- **退出码 0**：只读预检通过，或采样/输出完成且恢复证据通过。
- **退出码 2**：`-rgc` 返回且数据保留，但物理恢复证据不足。`completed_restore_unverified` 不表示恢复失败，也不表示硬件已验证；先处理恢复状态，再运行下一轮调频。
- **退出码 1**：运行、恢复或清理失败，查看具体记录；异常中断另保留 130/143，恢复/清理错误优先报 1。
- 恢复核验保守要求：至少三条恢复后带载读回高于上次锁频上界，并有读回回到本轮自动频率基线范围。最后候选频点较高时可能取不到证据，此时明确保留未验证，不能用空闲降频充当证明。

若恢复或清理失败，先保留上述记录，不继续下一轮 `--apply`。可在 **Windows PowerShell**只读核对这张卡的当前状态：

~~~powershell
nvidia-smi.exe -i GPU-22d1cd22-d198-45c4-2b54-974484026ab5 --query-gpu=uuid,clocks.current.graphics,clocks.current.memory,power.draw --format=csv
~~~

将 `restore_status.json`、`power_sampler_cleanup.json` 和具体命令错误用于处理当前恢复步骤。只读查询成功或空闲降频不能代替带载恢复证据；脚本不会自行扩大权限或切换控制方式。

## 开发侧验证边界

控制检查使用 CPU fixture，覆盖只读无写入、UUID、频点并集、5 次预热/5 次测量及能量式、错误功率拒绝、Windows 采样进程归属、异常/Ctrl+C/SIGTERM 恢复。源码检查核对保留函数与作者版本一致。

2026-11-11 开发检查：独立分支的 21 项 CPU 检查通过；作者保留函数核对通过。此前同一脚本的 5 张合成数据 PNG 绘制和 Windows PowerShell 启动/清理命令语法检查通过，未执行 GPU 写频。

可在本分支根目录运行 `python -B -m unittest discover -s tests -p test_vit_5090.py`，无需原 `5090` 分支的代码或测试辅助模块。

目标机 Windows/WSL 互操作、真实 CUDA ViT、全部频点锁频/读回、采样更新与硬件恢复均须本轮实机验收；CPU 检查不代表这些已通过。

接口依据：[NVIDIA nvidia-smi](https://docs.nvidia.com/deploy/nvidia-smi/index.html)、[Microsoft WSL 互操作](https://learn.microsoft.com/en-us/windows/wsl/filesystems#run-windows-tools-from-linux)。
