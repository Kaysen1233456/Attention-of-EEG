# 运行日志：probe_rv1_cnn_domain

- **状态**：失败(exit=1)
- **开始**：2026-10-09T07:47:05.002810+00:00
- **耗时**：28.1 秒
- **命令**：`C:/Users/famsun/Attention/OldIron/python.exe scripts/probes/cnn_domain_probe.py`
- **git 提交**：`4d73343`（main，有未提交改动）
- **备注**：RV-1 修复后重跑：折安全校准（协议 v1.3 方案 A），校准窗从评估集剔除；替代已作废的 probe3_cnn_domain

## 文件

| 文件 | 内容 |
|---|---|
| `meta.json` | 机器可读的完整元数据（命令、环境、依赖版本、git 状态、退出码） |
| `stdout.log` | 完整标准输出 |
| `stderr.log` | 完整标准错误 |
| `combined.log` | 按到达顺序合并的时间线，人工回溯看这个 |

## 关键结论

> ❌ **本次运行失败（exit=1），已被 `20261009-154822_probe_rv1_cnn_domain` 取代。**

```
RuntimeError: expected scalar type Double but found Float
  → torch/nn/modules/normalization.py:217  F.layer_norm

根因：折安全校准模块内部按 float64 计算，返回时未还原输入 dtype；
      float32 的波形 + float64 的频域特征送进同一模型 → LayerNorm dtype 冲突。
修复：apply_subject_stats(..., out_dtype=feats.dtype) / calibrate_by_subject 保持输入 dtype。
验证：float32 输入 → float32 输出；N1/N2 最大差异 1.0587（分组仍生效）。
```

## 复现方式

```bash
C:/Users/famsun/Attention/OldIron/python.exe scripts/probes/cnn_domain_probe.py
```
