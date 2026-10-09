# 运行日志：probe_rv1_domain_adaptation

- **状态**：成功
- **开始**：2026-10-09T07:44:13.257140+00:00
- **耗时**：4.6 秒
- **命令**：`C:/Users/famsun/Attention/OldIron/python.exe scripts/probes/domain_probe.py`
- **git 提交**：`4d73343`（main，有未提交改动）
- **备注**：RV-1 修复后重跑：折安全校准（协议 v1.3 方案 A，静息段前30窗），校准窗从评估集剔除；替代已作废的 probe2_domain_adaptation

## 文件

| 文件 | 内容 |
|---|---|
| `meta.json` | 机器可读的完整元数据（命令、环境、依赖版本、git 状态、退出码） |
| `stdout.log` | 完整标准输出 |
| `stderr.log` | 完整标准错误 |
| `combined.log` | 按到达顺序合并的时间线，人工回溯看这个 |

## 关键结论

> ⚠️ **本次运行已被 `20261009-154628_probe_rv1_domain_adaptation_v2` 取代，请勿引用本页数值。**
> 原因：本页跑在「N2 按通道分组」修正**之前** —— 当时的 `block` 参数是无效参数，
> N2 与 N1 数学上完全相同（本页两行都是 0.6967 即为证据）。修正后 N2 = 0.6733 ≠ N1。

```
本页数值：N0 = 0.6348 / N1 = 0.6967 / N2 = 0.6967（N2 为静默重复行，已作废）
```

## 复现方式

```bash
C:/Users/famsun/Attention/OldIron/python.exe scripts/probes/domain_probe.py
```
