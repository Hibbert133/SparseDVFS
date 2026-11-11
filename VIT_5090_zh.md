# 作者 ViT 脚本：RTX 5090 单文件运行

本交付位于 [Hibbert133/SparseDVFS 的 `5090-vit` 分支](https://github.com/Hibbert133/SparseDVFS/tree/5090-vit)，直接从作者 [c524c51](https://github.com/ziyangzhang96/SparseDVFS/commit/c524c51aac552fd16bec5743b1fb95309b46347b) 建立，保留原 `SparseDVFS.py`，只新增 ViT 的 5090 脚本、说明和测试。它不继承原 `5090` 分支中的 Qwen/vLLM 移植历史。

`SparseDVFS_5090.py` 可单独复制执行，不导入本项目任何包；不需要桥接服务、token、端口、Windows Python 或 pynvml。依赖仍是作者使用的 torch、torchvision、numpy、pandas、matplotlib、seaborn、tqdm。缺失或不兼容时报告真实错误，不自动安装/升级。

调频使用 Windows `nvidia-smi.exe`；实时功率使用 Python 标准库 `ctypes` 直接调用 WSL `/usr/lib/wsl/lib/libnvidia-ml.so.1`。NVML 按 UUID 初始化一次，每个测量单元启动作者结构的 Python 采样线程，不再启动 Windows 功率采样进程或追踪跨系统文件。

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
conda activate vit
~~~

已有 `SparseDVFS-vit` 独立目录时，不重复 clone；先在该目录确认工作区干净、分支为 `5090-vit`，再运行 `git fetch origin` 和 `git merge --ff-only origin/5090-vit`。遇到其他分支或用户修改时先处理，不覆盖。

本次目标机日志确认使用 `/home/s00958243/miniforge3/envs/vit/bin/python`，沿用 `vit` 环境，不重装或切换回 Qwen 环境。保留现有 Triton/PYTHONPATH 等环境变量。正式采样需要所选 GPU 空闲；若 Qwen TP=2 服务仍占用该卡，先由用户安排停止自己的任务，脚本不会停止服务。

## 2. 只读预检

**WSL Bash：**

~~~bash
python SparseDVFS_5090.py \
  --uuid GPU-22d1cd22-d198-45c4-2b54-974484026ab5 \
  --check-only
~~~

预期先打印 `NVML power check: 20/20 windows passed`，线程与 NVML 清理成功后打印 `PRECHECK PASS` 与独立输出目录。检查解释器、依赖版本/路径、CUDA/NVML/Windows UUID、Windows 工具、板级功率、核心频率、各显存档上报的 GPU 频点并集、Windows 管理权限及原作者权重缓存。实际运行 20 个 100 ms 的线程采样窗口，每个都必须有有效功率并正常停止；打印窗口样本数、查询耗时及采样间隔的中位数/最大值。导入 torch 前按 UUID 限定唯一可见卡。

不写频、不 reset、不下载或加载模型。`preflight.json` 中 `set_reset_verified:false` 是正常结果；`power_sampling.windows_passed:20` 只表示短窗口采样通过，清理证据另见 `power_sampler_cleanup.json`。这不证明传感器每 20 ms 更新，也不证明实际锁频权限。`windows_admin:false` 时允许执行只读预检，正式运行会停止，需从管理员 Windows 终端重新进入 WSL。

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

正式运行允许 torchvision 按原方式下载缺少的 `IMAGENET1K_V1` 权重。主加载失败时，恢复作者原有 `torch.hub.load('pytorch/vision:v0.10.0', 'vit_b_16', pretrained=True)` 备用路径，并记录两条路径的错误；备用路径也失败则退出，不使用随机/dummy 权重。备用路径在现有环境是否可用仍须实测。只读预检不会加载任一路径或触发下载。

正式运行也先验证 20 个短窗口，再加载模型；每单元创建并停止自己的采样线程。保留作者标题、模型加载成功、频率范围、逐层输入准备、`[当前/总数] Profiling`、测量成功、稀疏度及保存路径打印；CPU/EMC 不打印虚构频率。终端显示失败阶段与简明原因，完整 traceback 保存在 `errors.log`，完整命令和超时前输出保存在 `commands.jsonl`。

预期逐层遍历**全部上报候选频点**，成功单元逐条落盘，失败单独留痕。总单元数为作者选中层数 × 候选频点数，可能较长。恢复作者顺序：设置频率并等待 0.5 秒 → 额外等待 0.5 秒 → 读取实际频率 → CUDA 同步 → 5 次预热并同步 → 启动功率窗口、等待 0.1 秒 → 5 次同步测量。取消预热后的读频与 ±15 MHz 单元过滤，不重试或删改样本。上报候选、锁频命令返回和预热前读数均不证明测量期间频率已稳定。

目标机曾报告 28,786 个单元，仅两次 0.5 秒稳定等待合计约 8 小时，另有采样等待、命令与计算时间。保留作者逻辑及全部候选频点后，总运行时间仍以小时计。

按作者行为，捕获输入失败/模块不存在跳过该层，单元预热或计算失败记录后继续下一个频点；驱动、权限、采样、输出写入错误及中断仍终止并恢复，避免产生假功率或丢失记录。CSV 成功行只表示完成计算和采样，不表示目标频率验收通过。

## 与作者代码的准确关系

- 原样保留 `calculate_sparsity`、`get_layer_input_tensor`、`get_all_layers`，以及原稀疏度绘图与样式函数。选层规则不补选 attention、norm 等未被作者名称规则匹配的层，也不将父子模块去重。
- 保留 `vit_b_16(IMAGENET1K_V1).cuda().eval()`、默认精度、随机 `(1,3,224,224)` 输入、作者梯度开关范围、5 次预热、5 次同步 `perf_counter` 测量和“平均功率 × 平均延迟”。不引入 autocast、compile 或新的推理执行模式。
- CPU/显存由系统管理；移除 Jetson CPU/EMC 写入。CSV 的不可读 CPU 频率留空；热图采用“CPU system-managed”单列，跳过 CPU 扫频图。保留稀疏度分布、稀疏度/性能、逐层 GPU 延迟/板级功率及延迟/功率热图。
- 功率线程恢复作者 `start/stop` 结构：直接读取 NVML 功率，存入内存，`sleep(0.02)`，停止时等待已开始的查询完成并取平均。NVML 返回 mW，转换为 W；不支持、错误或非正值停止，不退回旧进程或固定功率。PowerShell 只保留 Windows 管理权限查询。
- 每条功率样本记录 WSL 单调时钟的查询开始/结束时间。计时窗口保留原来的 0.1 秒等待；停止请求时已开始的查询仍允许完成并计入均值，保存真实结束时间，不伪造为窗口内更早的样本。停止并 join 后才批量写原始 CSV/JSONL；不在实时采样线程中写盘，也不按文件到达时间筛选。20 ms 是查询后的休眠时间，未证明传感器有 50 Hz 独立更新。
- 能量是**单卡板级平均功率 × 平均层延迟的估计**，不是整机能量或独立仪器验证的孤立算子能量；保留作者估计器，未改成长窗口或累计计数器方法。
- 扫描前约 1 秒、reset 后约 1 秒的全模型读回位于层级测量之外：每次完整前向并 CUDA 同步后直接读取 NVML 核心频率，标记 `after_synchronized_forward`；不再筛选文件接收窗口或利用率。此证据描述同步前向后的驱动读数，不是独立 PLL 测量。保留的 ±15 MHz 仅供恢复判据使用，不参与单元筛选；恢复检查是移植选择，不是作者论文参数。

## 输出与失败处理

| 文件 | 内容 |
| --- | --- |
| `preflight.json`、`target_layers.json` | 当前环境、UUID、候选频点、缓存与作者选中层 |
| `profiling_results_ViT-Base.csv` | 逐单元追加的有效结果；含原指标、目标/实际 GPU 频率、显存频率、UUID 与口径 |
| `power_samples.csv`、`power_samples.jsonl` | NVML 功率 W、UUID、查询开始/结束单调时间、样本索引；停止线程后批量保存 |
| `clock_readbacks.jsonl`、`measurement_windows.jsonl` | 预热前读回；预检/测量窗口起止、延迟、采样索引、线程状态和失败原因，零样本窗口也留痕 |
| `failures.jsonl`、`commands.jsonl`、`dvfs_profiler.log`、`errors.log` | 失效单元、真实命令返回码/输出、作者风格运行日志和完整异常栈 |
| `restore_status.json`、`power_sampler_cleanup.json`、`status.json` | 恢复、Python 采样线程/NVML 清理及最终状态；清理文件沿用旧名称，backend 为 WSL_NVML_thread |

正常结束、Python 异常、Ctrl+C、SIGTERM 均进入统一清理：先停止当前采样线程，再尝试 GPU 复位/读回，最后关闭 NVML 并绘图。线程 join 最多等待 2 秒；未退出时报错，仍尝试 Windows GPU 复位，但不与该线程并发读取或关闭 NVML。强制杀进程/断电不能依赖 Python finally。

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

控制检查使用 CPU NVML fixture 和真实 Python 线程，覆盖只读无写入、UUID、功率单位、频点并集、5 次预热/5 次测量及能量式、20 个短窗口、错误功率拒绝、停止时在途查询、延迟写盘、异常/Ctrl+C/SIGTERM 恢复及禁止并发关闭 NVML。源码检查核对六个保留函数与作者版本一致。

2026-11-11 前次修订：恢复作者预热前读频顺序、移除测量单元频率容差过滤、恢复层计算失败继续和 ViT 原始备用加载路径；当时 23 项 CPU 检查通过。

2026-11-11 旧进程修复：将超时的 Start-Process 改为 Popen，当时 30 项检查通过，包括 3 项 Windows 子进程检查。但目标机随后记录 Windows 查询间隔中位数 31 ms、WSL 接收最大间隔 1599.867 ms，功率文件批量到达导致第一测量窗口无样本。旧进程已正常清理，旧运行恢复证据仍未验证，不修改旧数据状态。

2026-11-11 当前依据：目标机直接 NVML 功率查询中位/最大耗时 0.503/2.295 ms、采样间隔 20.605/21.102 ms、功率 67.125–67.179 W、核心频率 2527 MHz。此次移除旧功率进程，改为直接 NVML 线程；30 项本地控制/线程检查通过。目标机只读片段通过不代表修订脚本完整 ViT 扫描或硬件恢复已通过。本次未修改绘图算法。

可在本分支根目录运行 `python -B -m unittest discover -s tests -p test_vit_5090.py`，无需原 `5090` 分支代码或 GPU；已删除旧 Windows 功率进程测试。

目标机 Windows/WSL 互操作、真实 CUDA ViT、全部频点锁频/读回、采样更新与硬件恢复均须本轮实机验收；CPU 检查不代表这些已通过。

接口依据：[NVIDIA NVML](https://docs.nvidia.com/deploy/nvml-api/nvml-api-reference.html)、[NVIDIA nvidia-smi](https://docs.nvidia.com/deploy/nvidia-smi/index.html)、[Microsoft WSL 互操作](https://learn.microsoft.com/en-us/windows/wsl/filesystems#run-windows-tools-from-linux)。
