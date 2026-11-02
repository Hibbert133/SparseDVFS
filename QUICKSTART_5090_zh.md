# 5090 / WSL2 部署与验收

## 1. 前提与构建

目标机需要现有 NVIDIA Windows 驱动、支持 CUDA 的 WSL2、Docker GPU 透传和 x86_64 Linux。本文不会自动安装或修改 Windows、WSL、驱动与网络配置。
Windows bridge 仅监听 `127.0.0.1`，要求 WSL2 已启用 mirrored networking；不支持把控制端口开放到 LAN。若不满足，先运行 mock 链，或在原生 Linux 上使用直接 NVML 后端。
容器还需支持 `--network host` 并能访问该回环端口；Docker Desktop 的 host networking 能力需在目标环境自行确认。`probe` 是连通性验收，脚本不会自动更改这些设置。

```bash
bash scripts/run_5090.sh build
bash scripts/run_5090.sh download --out /model
bash scripts/run_5090.sh baseline --out /results/baseline --repeat 3
```

下载过程锁定 revision，记录模型文件 SHA256；运行前验证。27B INT4 实际显存和量化 kernel 兼容性仍需实机验证。初始参数为 4096 上下文、512 输入 token、32 输出 token、单请求、eager、70% GPU 显存预算。若 OOM，保留错误报告；不要自动切模型、卸载权重或声称运行成功。改变参数后须重新生成 baseline/profile/plan。

## 2. 真实 vLLM/CUDA + mock 调频

```bash
bash scripts/run_5090.sh profile --gpu /configs/gpu_mock.json --baseline /results/baseline/baseline.json --out /results/mock_profile
bash scripts/run_5090.sh plan --profile /results/mock_profile/profile.json --out /results/mock_plan.json
bash scripts/run_5090.sh run --mode serial --gpu /configs/gpu_mock.json --baseline /results/baseline/baseline.json --plan /results/mock_plan.json --out /results/mock_serial
```

此阶段真实执行量化模型及 CUDAFUSE，但频率请求仅写入 mock 对象。profile 的各“频点”并非真实变频测量；plan 标记 synthetic，禁止转用于真实 apply。检查 output token hash 与 baseline 一致、prefill/decode 顺序、68 个 module 的覆盖和边界日志。采样的最后一个 block 必须跨 execute_model/sample_tokens 正确闭合。

## 3. Windows bridge 和真实 GPU 频率

在 Windows 用 `nvidia-smi -L` 查 UUID。确保实验开始前采用默认自动频率；本 Demo 无法可靠读取并恢复其他程序预先设置的锁频区间，所以必须显式确认这一前提。不要同时使用其他锁频工具。

在 Windows PowerShell（按驱动要求以管理员运行）用现有 Python：

```powershell
$env:SPARSEDVFS_BRIDGE_TOKEN = python -c "import secrets; print(secrets.token_hex(32))"
python -m rtx5090.bridge --uuid GPU-YOUR-UUID --ack-default-auto --log bridge.jsonl
```

将同一会话 token 设置为 WSL 中的 `SPARSEDVFS_BRIDGE_TOKEN`，不要保存到仓库或截图。复制 `configs/gpu_bridge.template.json` 为 `configs/gpu_target.json`，填 UUID。

```bash
bash scripts/run_5090.sh probe --gpu /configs/gpu_target.json --out /results/capabilities.json
```

模板的 900/1500/2100 MHz 仅为占位。根据返回的 supported grid 选至少两个频点，修改 `gpu_target.json` 后再继续。查询、锁频或 reset 任一接口不支持，真实 DVFS 路径就不能通过验收；不要将模式静默切为 mock。

```bash
bash scripts/run_5090.sh calibrate --gpu /configs/gpu_target.json --apply --out /results/switches.json
bash scripts/run_5090.sh profile --gpu /configs/gpu_target.json --apply --calibration /results/switches.json --baseline /results/baseline/baseline.json --out /results/real_profile
bash scripts/run_5090.sh plan --profile /results/real_profile/profile.json --out /results/real_plan.json
bash scripts/run_5090.sh run --mode serial --gpu /configs/gpu_target.json --apply --calibration /results/switches.json --baseline /results/baseline/baseline.json --plan /results/real_plan.json --out /results/serial
bash scripts/run_5090.sh run --mode lookahead --gpu /configs/gpu_target.json --apply --calibration /results/switches.json --baseline /results/baseline/baseline.json --plan /results/real_plan.json --out /results/lookahead
```

标定测的是调用方单调时钟下的请求至匹配读回时间，包含桥接/轮询开销，不是物理 PLL 稳定时间；不相减 Windows 与 Linux 的时钟。真实 lookahead 必须有同一配置的实测标定。
NVML 回调当前经 ctypes/Python 执行，包含 GIL/IPC 开销。细 block 下该开销可能超过收益，必须计入对比，不能借用论文的开销数字。

## 4. 实机必须检查

- 相同量化模型与输入的原生、instrumented、fixed、serial、lookahead 输出一致；每种模式至少重复 3 次并保留原始结果。
- 真实 CUDA stream/event/callback 无乱序或死锁；module 覆盖不代表 operator/kernel 覆盖。operator 路径保留为实验接口，缺少匹配的真实覆盖证书时禁止实际调频。
- 用 `fuse_trace.csv`、CUDAFUSE trace 和 bridge 日志区分请求提交、接口返回、软件读回；接口返回不等于硬件已稳定。
- 检查每次退出的 `restore_status.json` 与 Windows bridge 日志。连接断开/超时尝试 reset；reset 失败会锁住服务拒绝新会话。强制终止服务、驱动崩溃或断电不保证恢复，需手动核查。
- 当前自动报告请求耗时和 token 一致性，不声称测得能耗。若要报告节能，另采同一时间域、明确覆盖请求窗口的 GPU 功率/能量数据，积分后报告 J/token、测量分辨率、温度和功率限制，并将其与整机能耗区分。
- Thor 的 CPU/GPC/EMC 三域联合策略、能耗与切换延迟不由该 5090 Demo 证明。

## 5. 无 GPU 回归

在已有 g++/Python3 的 Linux 环境：`sh scripts/test_5090.sh`。此命令不安装依赖、不访问真实 GPU。
在 Windows 执行 `python -B run_virtual_validation.py` 仅运行可用的 Python 部分，native/CUDA skip 必须如实保留；不能作为完整 C++ 验收。
