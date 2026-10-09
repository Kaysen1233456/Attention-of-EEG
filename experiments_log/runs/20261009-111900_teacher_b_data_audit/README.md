# 运行日志：teacher_b_data_audit

- **状态**：先失败（NumPy 2.0 兼容性）→ 修复后成功
- **命令**：`python scripts/run_with_log.py --name teacher_b_data_audit -- python scripts/audit_teacher_b_data.py --data data/processed/mental_arithmetic_250hz_8ch --output artifacts/teacher_b_data_audit.json`
- **git 提交**：`4d73343`（main）
- **捕获方式**：⚠️ 事后回填（backfilled）
- **耗时**：约 18 秒

## 一、发现并修复的 bug（这是它长期零产物的原因）

```text
AttributeError: `np.trapz` was removed in the NumPy 2.0 release.
    Use `scipy.integrate` trapezoid function instead.
```

`scripts/audit_teacher_b_data.py` 第 39 / 43 行使用了 `np.trapz`，
在 NumPy ≥ 2.0 环境下**直接崩溃**，所以该脚本自写好后**从未产出任何结果**。

修复（2 行兼容垫片，兼容 numpy 1.x 与 2.x）：

```python
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid
```

## 二、审计结果（好消息）

| 检查项 | 结果 |
|---|---|
| 被试数 × run 数 | 32 × 2 = 64，**全部配对正确**（train 28 + val 4；**不含 test 4 人**） |
| 结构错误（标签顺序 / run 不连续） | **0 条** |
| 非有限值样本 | 0 |
| 常量窗口 | 0 |
| 全零窗口 | 0 |
| 数据形状 | (3850, 8, 500) |
| 标签分布 | rest 2858 / focus 992（2.88:1，与采集设计一致） |

**结论**：上一轮担心的「被试身份靠 rest/focus 连续段推断、可能静默错位」，
**在当前数据上经审计确认是安全的**（32 被试 × 2 run 全部符合 rest→focus 交替且连续）。
仍建议后续显式落盘 `subject_id`，不要长期依赖推断。

## 三、异常发现（fold 5 为什么是困难折）

| 量 | fold5（**Subject13 / Subject09 / Subject01 / Subject27**） | 其他 28 被试 |
|---|---:|---:|
| 平均窗口数 | 91.0 | 89.07 |
| 平均 RMS | 8.80e-06 | 1.02e-05（**fold5 低 14%**） |
| focus − rest 的 RMS 差 | **+2.72e-07**（focus 更响） | **−1.25e-06**（focus 更轻） |

> **勘误（2026-10-09）**：本表原先写作「fold5（被试 1,10,19,26）」，**这是错的**。
> `1 / 10 / 19 / 26` 是 `metadata.json` 里 `train_subjects + val_subjects` 拼接列表的
> **位置索引**，不是被试编号。经核对 `scripts/audit_teacher_b_data.py:25` 的
> `FOLD_5 = {1, 10, 19, 26}` 与 `subject_names()` 的取序逻辑，真实映射为：
>
> | 位置索引 | 1 | 10 | 19 | 26 |
> |---|---|---|---|---|
> | **被试名** | **Subject13** | **Subject09** | **Subject01** | **Subject27** |
>
> 之所以容易搞错，是因为 `metadata.json` 里被试的排列顺序是**打乱的**（不是 00,01,02…）。
> 详见 `experiments_log/README.md` §6「被试命名约定」。

**振幅方向反转**：其他 28 个被试里 26 个 focus 段更轻，fold5 里 4 个有 2 个反过来。

但**任务响应本身更强更一致**（用 FFT 相对频带功率实测）：
- theta 上升 4/4
- alpha 下降 4/4

**所以 fold5 不是信号弱，是「任务响应的形态和别人不一样」** —— 这是典型的
**跨被试域偏移**，而不是数据质量问题。它和「阈值校准无收益」的结论指向同一处。

## 四、其他异常

- `Subject31` 的 rest 段只有 **40 个窗口**，其余被试都是 91 个 → 建议单独核查该被试原始 EDF。

## 对后续计划的影响

- ✅ 数据可信，可以继续投入
- ⚠️ fold5 / Subject31 需要在报告里单独标注
- ✅ 「域偏移」的诊断再次被独立印证 → 支持 `CALIBRATION_PROTOCOL.md` 的方向

## 复现方式

```bash
python scripts/run_with_log.py --name teacher_b_data_audit -- \
    python scripts/audit_teacher_b_data.py \
    --data data/processed/mental_arithmetic_250hz_8ch \
    --output artifacts/teacher_b_data_audit.json
```

产物：`artifacts/teacher_b_data_audit.json`
