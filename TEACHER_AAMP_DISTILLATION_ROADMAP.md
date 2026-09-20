# Ear-SAAD AAMP 教师与 CNN 蒸馏执行路线

更新时间：2026-09-20

## 1. 目标与硬约束

本项目的最终目标是在 Ear-SAAD 耳 EEG 数据上建立一个可靠的 Transformer
教师模型，再将教师知识蒸馏到可部署的轻量 CNN 学生模型。教师不是最终部署
模型；教师的作用是提供更强的分类 logits 和双耳表征。

本路线采用 subject-disjoint 的开发集验证。模型选择、超参数选择和架构决策
只能使用 train/validation subjects，test subjects 在最终方案锁定前不得参与
训练、搜索或 checkpoint 选择。

教师接受的唯一分类门槛是：

1. 固定三折开发集；
2. 每折运行 seed 42、43、44；
3. 汇总 window-level balanced accuracy；
4. 三折、三 seed 的平均 BA 必须达到 `0.70` 或更高；
5. 每个 fold/seed 不得发生类别坍缩；
6. 同时报告 macro-F1、ROC-AUC、train-validation gap 和 subject-level 结果。

只有全部条件满足，才允许保留教师并开始 CNN 蒸馏。任何单次高分、单折高分、
训练集高分或 AAMP reconstruction loss 下降，都不能替代这个门槛。

## 2. 模型结构

当前 AAMP/Transformer 编码器结构为：

```text
19-channel Ear EEG, 5 seconds, 20 Hz, 100 samples
  -> 3D electrode embedding
  -> temporal_pool=4
  -> MiniNeurIPT encoder
       4 Transformer layers
       temporal attention + spatial attention + PMoE
       d_model=96, 8 heads, d_ff=384
  -> AAMP reconstruction decoder (pretraining only)
```

AAMP 迁移到分类教师后，结构为：

```text
MiniNeurIPT encoder initialized from AAMP checkpoint
  -> learned left/right IILP pooling
  -> [L, R, |L-R|, L*R]
  -> SwiGLU classification head
  -> binary logits
```

当前 AAMP 编码器约 1.65M 参数；完整 Transformer 分类教师约 1.99M 参数。
CNN 学生使用独立的双分支卷积结构。Transformer 参数不能直接复制到 CNN 卷积层。

## 3. 七个固定步骤

### Step 1：完成 AAMP 自监督预训练

AAMP 使用无标签 EEG 学习可迁移的波形重建表征。正式开发协议可以使用 train
和 validation 的无标签波形，但 test 波形不进入预训练。归一化统计只来自
训练 subjects。

AAMP 的 reconstruction loss 是诊断指标，不是分类准确率。loss 下降只能说明
模型在当前掩码重建任务上的拟合能力改善，不能证明注意力分类 BA 达到 70%。

当前搜索协议为：

- 先进行 8 个 Sobol 准随机 trial，覆盖 learning rate 和 batch size；
- 再基于已观测结果进行 8 个 Matern Gaussian Process + EI 贝叶斯 trial；
- learning rate 在 `[1e-5, 5e-4]` 的 log 空间搜索；
- batch size 从 `[16, 32, 48, 64]` 中选择；
- 每个 trial 最多 25 epochs，validation 连续 5 epochs 无改善则早停；
- 目标为 `best_val_loss + 0.25 * max(0, val_loss - train_loss)`，越低越好；
- 搜索阶段使用 `train_only`，validation 仅计算固定掩码重建损失，test 不读取。

搜索完成后必须保存 `search_summary.json` 和 `recommended_params.json`。推荐参数
还要进行一次正式 `train_val` 复训，得到用于迁移的 AAMP checkpoint。

### Step 2：用 AAMP checkpoint 初始化 Transformer 分类教师

使用正式 AAMP 复训得到的 checkpoint 初始化相同结构的 MiniNeurIPT encoder，
再连接左/右耳 IILP pooling 和分类头。

必须记录：

- AAMP checkpoint 路径和 SHA-256；
- `d_model`、层数、头数、`d_ff`、`temporal_pool`；
- encoder 是否冻结、冻结多少 epoch、encoder learning-rate scale；
- 分类训练使用的 fold、seed、epochs 和所有优化器参数。

必须保留 random-init Transformer 对照。AAMP 初始化教师只有在严格分类验证中
证明有效，才能替代 random-init 教师。

### Step 3：三折 × seed 42/43/44 分类训练

固定 `configs/development_folds.json` 的三折划分。每个 fold 分别运行：

```text
seed=42, fold=0/1/2
seed=43, fold=0/1/2
seed=44, fold=0/1/2
```

每个运行必须创建全新的模型、DataLoader、优化器、scheduler 和 AMP scaler，
不得在 seed 之间复用内存状态。输出目录必须包含 fold 和 seed，避免覆盖。

### Step 4：统一汇总与报告

每个运行记录同一最佳 checkpoint 上的 train 和 validation 指标，并生成统一表：

| 层级 | 必须报告 |
|---|---|
| run | fold、seed、best epoch、checkpoint、train-validation gap |
| window | balanced accuracy、macro-F1、ROC-AUC、混淆矩阵 |
| class | 每类 precision、recall、F1、预测数量和预测占比 |
| subject | 每个 subject 的 BA、macro-F1、ROC-AUC、窗口数 |
| aggregate | 三折三 seed 的均值、标准差、最小值和最大值 |

类别坍缩定义为某个运行几乎只预测一个类别，或某类预测数量为零，或某类召回
率长期为零。此类运行不能被平均值掩盖，必须在汇总中单独标记。

### Step 5：教师保留决策

决策规则固定如下：

```text
mean(window-level validation BA) >= 0.70
AND every fold/seed has non-collapsed class predictions
=> 保留教师
```

若通过，再检查 macro-F1、ROC-AUC、subject-level 稳定性和 train-validation gap，
确认 70% 不是由单一类别或少数 subject 支撑。

若不通过，结论必须写成：

```text
AAMP reconstruction effective but downstream classification unqualified
```

这时不能蒸馏，不能用 test 结果挽救，也不能把 reconstruction loss 当成教师
准确率。应回到数据协议、AAMP 掩码、迁移方式或分类结构继续诊断。

### Step 6：教师通过后进行 CNN 蒸馏

只有 Step 5 通过后才启动。学生使用轻量双分支 CNN，教师保持 eval 模式并冻结。
蒸馏目标至少包括：

```text
L = CE(student_logits, labels)
  + lambda_kd * KL(student_logits/T, teacher_logits/T)
  + lambda_feat * bilateral_feature_loss
```

蒸馏必须和 CNN supervised-only baseline 使用相同 folds、seeds、训练预算和
评价指标。最终比较学生是否在明显更低参数量和计算量下保持可接受 BA。

### Step 7：未通过时的处理

若教师未达到 BA ≥ 0.70 或发生类别坍缩，AAMP 只记录为重建任务有效，不能进入
蒸馏。下一轮只能一次改变一个因素，例如：

- AAMP checkpoint 选择和冻结策略；
- temporal_pool=2 或 temporal_pool=1；
- mask ratio 和 amplitude-aware masking；
- normalization、zero-channel policy 和标签/窗口对齐；
- 分类头或双耳池化方式。

每次变化都必须重新使用固定三折、固定 seeds 和相同报告模板。

## 4. 当前状态与已完成工作

已完成的工程工作包括：

- AAMP 预训练入口支持 train-only、train+val 和显式 transductive 模式；
- AAMP 使用无标签 waveform 包装，训练循环无法消费标签；
- AAMP 输出 metadata、history、checkpoint SHA-256 和固定验证 mask 记录；
- AAMP masking 改为 PyTorch 向量化实现，避免 CPU/NumPy 逐通道瓶颈；
- 增加 Sobol 准随机和 Bayesian Optimization 搜索入口；
- 搜索结果记录学习率、batch size、最佳 epoch、validation loss 和 gap；
- 修复训练入口的 `zero_channel_policy` 配置兼容和 `temporal_pool` CLI；
- 保留固定三折、多 seed、subject-level 和类别稳定性报告协议；
- 既有 B0/B1 严格验证已经证明 random-init Transformer 路线需要谨慎，不能直接
  进入蒸馏。

截至本文件更新时间，AAMP 搜索应以 `artifacts/aamp_lr_bs_search_v2` 的实际
文件为准。若搜索尚未生成完整的 8 个 Sobol 和 8 个 Bayesian 结果，Step 1
仍未正式闭合。

## 5. 禁止事项

- 不把 masked reconstruction loss 写成分类准确率；
- 不在教师 BA ≥ 0.70 前训练 CNN 蒸馏学生；
- 不把单折、单 seed 或单次 trial 当作最终结果；
- 不把 test subjects 用于搜索、早停、checkpoint 选择或改结构；
- 不把 Transformer encoder 权重直接加载进 CNN；
- 不提交原始 EEG、下载数据、`.pt` checkpoint 或大型 artifacts 到 Git；
- 不用未完成搜索的中间推荐参数作为最终配置。

## 6. 推荐执行命令模板

AAMP 搜索：

```bash
PYTHONPATH=src python scripts/search_aamp_lr_bs.py \
  --config configs/pretrain_aamp_ear_saad.yaml \
  --output artifacts/aamp_lr_bs_search_v2 \
  --quasi-trials 8 --bayes-trials 8 \
  --epochs 25 --patience 5 --device cuda
```

AAMP 完成后，分类教师命令必须使用正式 checkpoint，并将 fold/seed 写入独立
输出目录。具体命令以 checkpoint SHA 和最终配置确认后生成，不能提前启动。
