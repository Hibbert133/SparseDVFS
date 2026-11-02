# v5 检查记录（2026-10-30，Asia/Shanghai）

仅使用本机已有 Python/标准库、Git Bash 和预先存在的 tree-sitter 环境；没有安装依赖、下载镜像/权重或访问真实频控节点。

| 检查 | 实际结果 |
|---|---|
| 单元与模拟集成测试 | 143 总计，134 通过，9 跳过，0 失败/错误 |
| Python AST | 48 文件通过 |
| Bash 语法 | 7 脚本通过；文档 15 个 bash 代码块通过语法检查 |
| C++ 结构解析 | 6 文件通过；宏/extern-C guard 归一化；没有编译、类型、链接或 CUDA API 验证 |
| JSON/结构化阶段记录 | JSON 实际解析；4 个阶段文件采用 YAML 1.2 的 JSON 子集；其余 YAML 手动核对，没有安装解析器 |
| v5 合成入口 | 实际执行 3 请求/2 上下文、每上下文 2 GPU 单元、2 个块；无真实张量 |
| 已知参数拟合 | k1=0.009999999999999995，k2=0.19999999999999982，留出最大相对误差 5.66e-16；全为合成数据 |
| 恒温失败案例 | 实际检测到秩亏，保持 deployable=false |
| 旧版本 | 四个旧 zip 哈希和其内部清单重新校验通过；没有修改 |
| Thor/CUDA/vLLM/Qwen/调频/真实能耗/论文结果 | 全部未执行 |

9 项跳过属于原生 C++/CUDA/ABI 测试：本机没有对应 Linux 共享库和 GPU。它们必须在 Thor selftest 真正运行，不能用静态检查替代。

新增测试覆盖：算子原始调用一次及返回对象不变、融合单元、缺失 launch/kernel 归属、跨线程归属错误、辅助 stream、缺 correlation、三请求/完整 decode、原生输出比较、签名/上下文变化、每使用 OPP 的证据、旧 schema/跨粒度拒绝、共享系数与留出验证、未知流量/电压、单位和电源轨错误、传感器窗口、定向切频/no-op 排除、没有实际提前请求时报告不验收。

完整日志：UNIT_TEST_RESULTS.txt；机器可读结果：VALIDATION.json 和 STATIC_CHECKS.json。合成结果在 examples/v5_*.json，全部 nondeployable。CHANGES_FROM_V4.diff 在临时 v4 副本上执行 git apply 检查及前向应用，应用后文件按字节比较；细节见 DIFF_VALIDATION.json。

源码 diff 排除自身、SHA256SUMS 和 DIFF_VALIDATION.json 以避免自引用；删除旧参考正文使用前向 literal-empty 二进制删除片段，不在 diff 中重新分发旧正文。应用时使用 `git -c core.autocrlf=false apply --binary CHANGES_FROM_V4.diff`；该 -c 参数只影响当前进程，不改全局配置。原始 v4 仍保留，可用于恢复。
