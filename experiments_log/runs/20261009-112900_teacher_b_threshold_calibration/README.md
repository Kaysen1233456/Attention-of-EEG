# 运行日志：teacher_b_threshold_calibration

- **状态**：成功（exit=0）
- **命令**：`python scripts/run_with_log.py --name teacher_b_threshold_calibration -- python scripts/calibrate_teacher_b_threshold.py --output artifacts/teacher_b_calibration`
- **git 提交**：`4d73343`（main）
- **捕获方式**：⚠️ 事后回填（backfilled）
- **耗时**：约 612 秒

## 目的

检验「教师性能瓶颈是否来自决策阈值」。用**嵌套 CV**（8 个外层折 × 4 个内层折 × 3 seed）
在内层选阈值、外层评估，与固定阈值 0.5 对比。**不接触 test 集。**

## 结果

| 口径 | BA | AUC |
|---|---:|---:|
| 固定阈值 0.5 | **0.6693** | 0.7309 |
| 嵌套 CV 校准阈值 | **0.6681** | 0.7309 |
| 差值 | **−0.0012** | 0（阈值不影响 AUC） |

逐 seed：

| Seed | fixed 0.5 BA | calibrated BA | AUC |
|---:|---:|---:|---:|
| 42 | 0.6531 | 0.6511 | 0.7236 |
| 43 | 0.6930 | 0.6910 | 0.7457 |
| 44 | 0.6619 | 0.6623 | 0.7234 |

内层选出的 24 个阈值：min 0.380 / max 0.620 / **mean 0.494**（几乎就是 0.5）。

## 关键结论

1. **阈值校准无收益**（−0.0012，远小于折间 std ±0.04）。
2. **这是最重要的诊断**：按 `TEACHER_PROTOCOL.md` 自身写死的判定规则 ——
   > 若阈值校准无收益，则瓶颈是**表征 / 跨被试域偏移**，不是决策阈值，**必须重做模型结构**。
3. **独立印证了探针实验的结论**：问题在特征表征层，不在决策层。
4. **重要副产物**：本次嵌套 CV 给出了教师**最诚实的成绩单 0.6693**
   （对比项目此前记录的 0.7166，差 4.7 分，差额主要来自「用外层验证集选 epoch」的选择偏差）。

## 对后续计划的影响

- ❌ 停止一切阈值调优工作
- ✅ 转向表征层：被试级校准（`CALIBRATION_PROTOCOL.md`）、空间注意力、域对抗
- ✅ **项目对外汇报统一改用嵌套 CV 口径**

## 复现方式

```bash
python scripts/run_with_log.py --name teacher_b_threshold_calibration -- \
    python scripts/calibrate_teacher_b_threshold.py --output artifacts/teacher_b_calibration
```

产物：`artifacts/teacher_b_calibration/summary.json`
