# 资源模型到 Qwen 的离线迁移检查

这两个命令不访问服务、不写频、不改变正在采集的任何数据，不需要重启。
`prepare` 在资源采集尚未完成时即可运行；`validate` 在 frozen-model.json 存在后运行。

## 输入与契约

- 输入为完整 `two_setting_operators_timing_complete` collection.json，要求6窗口、两个频率各3次、恢复通过、同一算子集合及参数/parent契约。
- 默认复用 `runs/eager-operator-v2-20261109-203829/operator-two-settings-20261109-210122/collection.json`。
- 只将密集连续 BF16 linear/add/mul 映射到对应资源模型。linear重新推导2MNK及逻辑字节，算术检查广播与alpha参数。不同dtype、非连续布局、错误rank设备不迁移。
- Attention/cache/gdn需状态及融合路径证据；collective需双rank通信模型；转换/copy/alias暂不自动绑定。未映射项保留，不填零。
- 核实输入输出元数据不是核实kernel；所有映射保留kernel/cache/重复调用与单次调用时间口径的差异。源历史服务身份与新资源模型身份不同会单列阻塞，不合并身份。
- 预测使用历史请求带载读回中位数，不把2250/2527请求值当实频；中位实频超资源训练范围拒绝外推，完整读回范围越界单列。
- 按rank/context/family/请求频率报告WAPE、P90，并保留三次标签。零Event标签保留为未解决，不能参与相对误差分母。
- 映射与预测均绑定来源哈希。validate不拟合，不改参数，不声称历史数据是事先冻结的独立留出。
- 功耗与能量预测均保留null：没有Qwen同路径算子窗口标签，不能把资源功率乘Qwen Event耗时充当验证。结果始终非部署证书。

## WSL Bash：现在准备映射

```bash
cd /mnt/d/s00958243/SparseDVFS
PY=/home/s00958243/miniforge3/envs/vllm-qwen/bin/python
MAP="$PWD/runs/qwen-resource-map-$(date +%s%N)"
"$PY" -B -m rtx5090.qwen_resource_transfer prepare \
 --input "$PWD/runs/eager-operator-v2-20261109-203829/operator-two-settings-20261109-210122/collection.json" \
 --out "$MAP"
printf '%s\n' "$MAP" > runs/next-qwen-resource-map.txt
```

## 资源参数冻结后：迁移诊断

`FORMAL` 使用本次 resource-formal 输出目录，不混用旧模型：

```bash
MAP="$(cat runs/next-qwen-resource-map.txt)"
"$PY" -B -m rtx5090.qwen_resource_transfer validate \
 --input "$MAP/mapping.json" --model "$FORMAL/frozen-model.json" \
 --out "$MAP/validation-$(date +%s%N)"
```

先检查资源参数是否可辨识，再看跨Qwen形状/执行路径的残差；若迁移不通过，模型限制在其已验证的资源负载范围，不用这些历史标签回填参数。
