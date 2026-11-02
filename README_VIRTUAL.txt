SparseDVFS v5 纯软件验证补充包
验证日期：2026-10-31（Asia/Shanghai）

本包是原 v5 的独立副本，增加 6 项虚拟链路测试和两个复跑入口。
没有修改原 v5，没有安装依赖、下载权重或操作物理频控节点。
原包 VALIDATION.json 等文档记录的是历史验证；本轮以本文件和
virtual_validation_results/summary.json、tests.txt 为准。

本机实际结果
149 项测试：140 通过，9 跳过，0 失败，0 错误。
原有 143 项中 134 通过；新增 6 项全部通过。
跳过的 9 项：C++ core mock 6 项、CUDA adapter ABI 1 项、CUDA integration 2 项。

新增链路
真实 SparseDVFSWorker 适配器方法 → 真实 Spans hooks → 真实 Schedule
→ RecordingController（Python 测试替身）。
vLLM 基类、模型模块、torch/CUDA event 均为测试替身。
通过原有 WorkerTests 的 instrumented fixture 注入合成 plan，未走生产
plan 身份/profile 验证入口；不能把该合成计划作为可部署计划使用。

模拟 1 次 prefill + 3 次 decode：
每步 68 个单元，划分为 4 个多单元 super-block，总计 272 次单元访问。
预期与实际均为 16 次 begin、16 次 end；块内无多余调度。
末块包含 norm/logits/sample，确认其可跨 execute_model/sample_tokens。
确认 phase 切换、下一块映射、serial 禁用 lookahead、短块资格检查。
确认缺失映射、单元错序、未闭合块在对应检查点报错。
这些事件仅表明调度请求顺序，不能证明异步 GPU 执行或硬件调频效果。
fixture 的计时为人工固定值，禁止用来比较性能、延迟或节能比例。

本机限制
Windows/Python 3.14；未发现可用 C++ 编译器、Linux/WSL 运行环境。
因此本轮没有编译或执行真实 C++ FUSE，也没有执行真实 CUDA/vLLM。
已有 native mock 测试并未通过，只是因缺少库而跳过。

复跑纯软件测试（Python 标准库即可）
在解压目录执行：
  python -B run_virtual_validation.py
默认结果写入 virtual_results/。
virtual_chain.json 含合成 plan、块边界事件和目标频率请求，不含真实频控回读。

在已有 Linux + C++17 + CMake + Python 环境补跑真实 C++ mock
  bash run_native_mock.sh
该脚本编译原 C++ core，关闭 CUDA，运行全套测试，并另外要求
CppTests 与 NativeCore 共 6 项全部通过且不能跳过。
原生测试使用临时普通文件模拟 CPU/GPU/EMC 节点，执行 set、队列、
cancel、过期 ticket 拒绝、no-op 和异常恢复等检查。
不会安装依赖，不要求 GPU，不使用 --apply，不写真实 /sys。
该 Linux 入口已作语法检查，本机没有执行其编译与原生测试步骤。

本轮新增文件
  tests/test_virtual_chain.py
  run_virtual_validation.py
  run_native_mock.sh
  README_VIRTUAL.txt

下一阶段实机边界
真实 GPU：验证 CUDAFUSE stream/event、真实 vLLM 内核及输出一致性。
Thor：验证频控接口、物理时钟生效、切换延迟、整机能耗与策略收益。
