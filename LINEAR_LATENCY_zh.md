# 当前形状的实用时延模型

用户2026-11-06调整要求：先给近似准确的模型。新增 `rtx5090.linear_latency_model`。
每个rank/phase/linear形状建立一个常数：全部三批N=32单次均值的中位数；N=128全部三批用于回顾性误差检查。
不丢弃任何训练或检查批次。N=1/8原始证据保留，但模型明确不适用于孤立/短批调用。
该协议是在查看各档数据之后选定的，因此不是预注册协议或独立验证，不能把本次误差称为未来预测精度保证。
不外推频率/形状/缓存条件，不估计功耗或整请求时间，不乘64层推导端到端。
这是同形状连续执行的经验时延表，不是论文Eq1参数拟合，不授权DVFS。

```bash
# 目标机普通WSL Bash；纯离线，无需重启
python -B -m rtx5090.linear_latency_model \
  --replay runs/eager-kernels-1793949231347413179/matrix-calibrate-1793949482255323717 \
  --out runs/linear-latency-$(date +%s%N)
```

产物：model.json（八组预测及形状/kernel/身份绑定）、check.json（每组全部三批误差）、REPORT.txt。
纯CPU夹具已检查中位数、留出标签不参与参数计算、误差门槛和未验证状态；目标文件由用户在目标机执行。
