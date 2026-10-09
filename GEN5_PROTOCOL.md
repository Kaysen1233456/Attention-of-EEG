# Gen5 Fixed Protocol

更新时间：2026-10-09（batch size 冻结 + 扫描点归档）

## 目标

建立跨被试的心算专注度二分类模型。模型选择只使用开发被试。
历史 test holdout 已因 batch-size 扫描和三 seed 复现而暴露；后续决策不得再使用它。

## Gen5 的起点

Gen5 继承并修复了旧 Gen4 的数据、评估和模型问题。旧 Gen4 的问题记录如下：

旧 Gen4 存在以下问题：

1. 微调阶段没有复用预训练阶段的 train-subject normalization。
2. 阈值搜索把 logit 当作 probability 使用。
3. 阈值来自最后一轮验证结果，而不是最佳 checkpoint。
4. checkpoint 依据 AUC 选择，和项目主协议的 balanced accuracy 不一致。
5. 预训练 decoder 与分类 decoder 维度不同，却使用 strict load，导致迁移失败。
6. 心算数据使用了不合理的左右通道分组、差值和乘积特征。
7. 模型容易利用整体振幅差异，验证被试有效，测试被试失效。

## 固定模型

当前固定模型为 Gen5 轻量双分支融合模型：

```text
窗口级 z-score EEG
  -> temporal CNN

同一窗口
  -> 相对 delta/theta/alpha/beta/gamma 功率
  -> spectral MLP

temporal branch + spectral branch
  -> MLP classifier
```

固定关闭：

```text
左右差值特征
左右乘积特征
IILP pooling
PMoE
旧 AAMP/Gen4 预训练 encoder
```

## 交叉验证结果

使用 32 个开发被试进行 8-fold GroupKFold：

| 模型 | Balanced Accuracy | Macro-F1 | ROC-AUC |
|---|---:|---:|---:|
| 相对频谱 Logistic | 0.644 +/- 0.042 | 0.606 +/- 0.080 | 0.718 +/- 0.043 |
| CNN + 相对频谱 | 0.673 +/- 0.046 | 0.645 +/- 0.063 | 0.729 +/- 0.076 |

CNN + 频谱模型已经固定为 Gen5。它在开发集交叉验证中优于纯频谱 baseline，但仍存在被试间波动。

## 最终训练协议

1. 使用全部 train 和 val 被试训练。
2. 使用固定 15 个 epoch，不使用 test 选择 epoch、阈值或超参数。
3. 使用窗口级 z-score 和相对频带功率。
4. 使用 class-balanced cross entropy。
5. 使用 `batch_size = 256`（冻结值，见「冻结超参数」）。
6. 训练结束后加载最终 epoch 权重。
7. 历史 test 结果仅作为已暴露 holdout 记录；后续结构、超参数、校准和教师决策只使用开发集嵌套 CV。

### 冻结超参数（Frozen Hyperparameters）

| 超参数 | 冻结值 | 来源 |
|---|---|---|
| `batch_size` | **256** | 手动扫描（bs128 / 256 / 512，seed42）中的最优值 |
| `epochs` | 15 | 固定协议 |
| `optimizer` | `AdamW(lr=1e-3, weight_decay=1e-4)` | `scripts/final_spectral_cnn.py` |
| `grad_clip` | 2.0 | 同上 |
| `seed` | 42 / 43 / 44 | 见「三 Seed 复现结果」 |
| `normalization` | 窗口级 z-score，clip 8 | `normalize_windows` |
| `loss` | class-balanced cross entropy | 同上 |

> ⚠️ **Gen5 最终训练和历史复现入口必须使用 `batch_size = 256`。**
> `scripts/train_gen5_seeds.py` 的默认值已是 256；`scripts/final_spectral_cnn.py` 的默认值已同步为 256。
> `scripts/cv_spectral_cnn.py` 是开发 CV 工具，运行时必须显式传入冻结值；它的历史默认值不代表最终协议。
> 显式传入 `--batch-size 256` 是最稳妥的做法。

### batch size 扫描记录

| `batch_size` | 状态 | 说明 |
|---:|---|---|
| 128 | **未采用** | commit `5617d1e` 中的扫描点（仅 seed42），非最终配置 |
| 256 | **已采用（冻结）** | Gen5 baseline 最终配置，与三 seed 复现结果一致 |
| 512 | **未采用** | commit `5617d1e` 中的扫描点（仅 seed42），非最终配置 |

> **方法澄清**：Gen5 的 batch size 通过**手动扫描**（bs128 / 256 / 512 × seed42）确定，**不是** Sobol + 贝叶斯搜索。
> Sobol + 贝叶斯搜索实际用于 AAMP 预训练、路线 C 微调、Teacher B 超参选择。**仓库中不存在 Gen5 的搜索产物**，此处记录以避免后续误引。

## 代码记录

- `scripts/finetune_freq.py`: 修复 Gen4 归一化、checkpoint、概率阈值和评估流程。
- `src/attention_model/models/mini_neuript.py`: 迁移时跳过不兼容 decoder 参数。
- `scripts/audit_mental_arithmetic.py`: 生成被试级振幅和频谱分布审计。
- `scripts/baseline_spectral.py`: 纯频谱 baseline。
- `scripts/cv_spectral.py`: 纯频谱 GroupKFold。
- `scripts/cv_feature_fusion.py`: 时域统计 + 频谱融合对照，结果未采用。
- `scripts/cv_spectral_cnn.py`: 当前固定模型的 GroupKFold 验证。
- `configs/gen4_baseline.yaml`: 窗口标准化的无左右先验实验配置。
- `scripts/final_spectral_cnn.py`: Gen5 历史暴露 holdout 复现入口（默认 `batch_size=256`，必须显式确认）。
- `scripts/train_gen5_seeds.py`: 三 seed 历史暴露 holdout 复现入口（默认 `batch_size=256`）。

## 三 Seed 复现结果

云 GPU 使用固定协议重新训练了 seed `42`、`43`、`44`。三个结果使用同一个 test label hash：

```text
631b6555f252336b255bd8b36849200e910f0e6835465264507cdde675bda3ab
```

| Seed | Balanced Accuracy | Macro-F1 | ROC-AUC |
|---:|---:|---:|---:|
| 42 | 0.7204 | 0.6477 | 0.7521 |
| 43 | 0.6736 | 0.6254 | 0.7196 |
| 44 | 0.6822 | 0.5613 | 0.7753 |
| Mean | 0.6921 | 0.6115 | 0.7490 |
| Std | 0.0203 | 0.0366 | 0.0229 |

逐被试检查显示 `Subject20`、`Subject07`、`Subject28` 表现相对稳定，`Subject14` 是明确的困难被试，三个 seed 的 BA 约为 `0.545`、`0.567`、`0.561`。后续教师模型和蒸馏实验必须继续报告逐被试指标，不能只报告总体均值。

Gen5 现在冻结为 `student_baseline`。历史 test holdout 已暴露，后续实验不得使用它选择结构、超参数、epoch、阈值或校准方案。

## 教师模型阶段：待讨论方案

教师训练应该只使用原始 train 和 val 被试组成的开发集，并在开发集内部执行嵌套 GroupKFold。
历史 test holdout 不再用于教师结构、超参数、校准或蒸馏决策。

候选流程：

```text
开发被试
  -> GroupKFold
  -> 每个 fold 训练教师
  -> 只用 fold validation 选择 epoch
  -> 汇总 fold/seed 的 BA、Macro-F1、AUC 和逐被试指标
  -> 确认教师稳定超过 Gen5
  -> 固定教师后再进行一次 test 评估
```

教师只有在交叉验证平均 BA、AUC 和困难被试表现都优于 Gen5 时，才进入蒸馏阶段。教师模型的具体容量、输入分支、损失和 fold 数量在训练前单独确定。

## 评估口径登记（避免数字口径混淆）

同一个模型在不同评估口径下的 BA 会有系统性差异。**任何报出的 BA 必须注明口径**：

| 口径 | 含义 | 相对偏差 |
|---|---|---|
| 单 seed + 外层验证早停 | 用同一折的验证集选最佳 epoch | 偏高（约 +1.8 分） |
| 多 seed + 外层验证早停 | 3 seed 平均，仍用验证集选 epoch | 偏高（约 +2.5 分 vs 嵌套） |
| **多 seed + 嵌套 CV** | 内层选 epoch，外层评估，泄漏安全 | **最诚实，作为对外口径** |

> 例：Teacher B v2 的 0.7166 属于第一行口径；同模型在第三行口径下约 **0.6693**。
> **项目对外汇报统一使用「多 seed + 嵌套 CV」口径；历史 test 结果标记为 exposed holdout。**
