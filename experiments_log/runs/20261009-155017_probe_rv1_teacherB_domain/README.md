# 运行日志：probe_rv1_teacherB_domain

- **状态**：成功
- **开始**：2026-10-09T07:50:17.811214+00:00
- **耗时**：74.1 秒
- **命令**：`C:/Users/famsun/Attention/OldIron/python.exe scripts/probes/teacherB_probe.py`
- **git 提交**：`4d73343`（main，有未提交改动）
- **备注**：RV-1 修复后重跑：折安全校准（协议 v1.3 方案 A），校准窗从评估集剔除；替代已作废的 probe4_teacherB_domain

## 文件

| 文件 | 内容 |
|---|---|
| `meta.json` | 机器可读的完整元数据（命令、环境、依赖版本、git 状态、退出码） |
| `stdout.log` | 完整标准输出 |
| `stderr.log` | 完整标准错误 |
| `combined.log` | 按到达顺序合并的时间线，人工回溯看这个 |

## 关键结论

> **RV-1 修复后的 Teacher B 域适应探针。** 校准为「静息段前 30 窗、折安全拟合」，校准窗已从评估集剔除。

```
T0_original_no_std(calib_excluded)  n=2890  BA = 0.7130  AUC = 0.7684  折间 0.7124±0.0528  min折 0.638  >0.6 27/32
T1_matched_no_std                   n=1922  BA = 0.6899  AUC = 0.7080  折间 0.6914±0.0455  min折 0.637  >0.6 24/31
T2_matched_+subject_std             n=1922  BA = 0.7180  AUC = 0.7719  折间 0.7221±0.0642  min折 0.665  >0.6 25/31

对照：作废的旧值 T0 0.7071 / T1 0.6613 / T2 0.7308(+7.0)
```

**结论**
1. **T2 相对 T1 的增益从 +7.0 缩到 +2.8 分** —— C10 泄漏修掉后的诚实值。
2. **T1 从 0.6613 升到 0.6899（+2.9 分）**：因为时长匹配采样现在从**非校准**静息池（61 窗）里抽，
   而非全部 91 窗。评估集口径变化是主因，不是模型变了。
3. T0（全量非校准窗）0.7130 仍高于 T1（时长匹配）0.6899 —— **时长匹配会损失约 2.3 分**，
   这是 D2「用少量分数换公平对比」的实际代价，应在引用 D2 时一并说明。

## 复现方式

```bash
C:/Users/famsun/Attention/OldIron/python.exe scripts/probes/teacherB_probe.py
```
