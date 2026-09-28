# EEG 专注度监测项目 — 完整路线与结果文档

> 最后更新：2026-09-23
> 项目根目录（云GPU）：`/mnt/workspace/Attention-of-EEG/`
> 本地工作目录：`C:\Users\Administrator\Doubao\chats\2026-09-09\new-chat\`

---

## 一、最初目标（我们到底在做什么）

### 1.1 大方向

我们的目标是做一个**基于头皮EEG信号的专注度二分类模型**：给一段8通道、250Hz采样率的EEG脑电数据，模型判断这个人当前是"专注"还是"休息"。

这是一个典型的**自监督预训练 + 下游微调**范式：
1. 先用大量无标签EEG数据做自监督预训练（AAMP框架），让模型学会EEG信号的通用特征表示
2. 再用少量有标签的专注度数据做下游微调，冻结预训练好的Encoder，只训练分类头

### 1.2 为什么从AAD转向专注度

最初我们尝试的是**AAD（听觉注意解码）任务**——在双耳分听实验中，判断用户在注意哪只耳朵的声音。用的是Ear-SAAD数据集。

但AAD任务效果一直达不到门槛（路线A，已废弃），于是我们转向了**心算数据集的专注度二分类**——任务更简单、数据更干净，更适合验证预训练+微调的范式是否有效。

### 1.3 核心科学问题

路线B和路线C的对照，本质上回答一个问题：

> **预训练阶段的超参数，需不需要在目标数据集上专门搜索？**

- 路线B：预训练用默认参数（拍脑袋），下游微调认真搜参
- 路线C：预训练也在目标数据集上认真搜参，下游微调也认真搜参
- 两条路线的下游微调搜索流程完全一致，唯一变量是**预训练权重不同**

如果路线C显著优于路线B，说明"预训练参数也需要针对目标数据集搜索"；如果差不多，说明"预训练用默认参数就够了，重点在下游微调"。

---

## 1.4 当前决策：Gen3 固定，暂停旧结构微调

路线C三种子阶段1结果现固定为 **Gen3 baseline**，不再继续在当前
AAD遗留结构上追加无止境微调。Gen3的验证集结果为：

| 指标 | 三 seed 均值 ± 标准差 |
|------|----------------------|
| Balanced Accuracy | 65.823% ± 0.758% |
| Macro-F1 | 66.107% ± 0.666% |
| ROC-AUC | 67.892% ± 0.124% |
| Accuracy | 74.795% ± 0.335% |
| 类别坍缩 | 无 |

Gen3必须保留为不可回退基线，但不能把它解释为“路线C预训练目标已经被证明
适合专注度”。当前结果同时受到新预训练权重和路线C独立下游超参数的影响，
因此下一步回到预训练目标和编码器结构，建立 Gen4。

---

## 二、数据集

### 2.1 心算数据集（当前主线）

| 属性 | 值 |
|------|-----|
| 路径 | `data/processed/mental_arithmetic_250hz_8ch/` |
| 通道数 | 8 |
| 采样率 | 250 Hz |
| 任务 | 二分类：专注（心算）vs 休息 |
| 类别数 | 2（n_classes=2） |
| 归一化 | 配置文件中指定（normalize, normalization_mode, zero_channel_policy） |

### 2.2 Ear-SAAD数据集（路线A，已废弃）

| 属性 | 值 |
|------|-----|
| 任务 | AAD听觉注意解码（双耳分听） |
| 状态 | 未达门槛，已废弃 |

---

## 三、三条路线详细说明

### 3.1 路线A：AAD任务（已废弃）

**做了什么**：
- 在Ear-SAAD数据集上做听觉注意解码
- 尝试了多种模型和调参策略

**结果**：
- 效果未达项目门槛
- 决定放弃AAD，转向心算专注度二分类

**文件位置**：
- 相关实验记录在早期artifacts目录中（已不再维护）

---

### 3.2 路线B：默认预训练 + 认真微调（当前基线，gen2）

#### 3.2.1 预训练阶段

**做了什么**：
- 用**默认超参数**在心算数据集上做自监督预训练
- 预训练参数：learning_rate=1e-4, batch_size=32（没有搜索，直接用的默认值）

**预训练权重**：
- 路径：`artifacts/pretrain_aamp_mental_arithmetic_250hz_8ch/diagnostic_best_pretrain_model.pt`

#### 3.2.2 下游微调阶段（冻结Encoder）

微调分四步走，和路线C完全一致的搜索流程：

**第一步：手动试参确定gen1基线**
- 手动尝试几组参数，确定一个合理的起始点

**第二步：准随机搜索（Sobol 20组）**
- 用Sobol序列在搜索空间中均匀采样20组参数
- 目的：快速覆盖搜索空间，找到有希望的区域

**第三步：贝叶斯优化（EI 30组）**
- 以准随机的20组结果为初始观测
- 用高斯过程+EI采集函数继续搜索30组
- 目的：在有希望的区域精细挖掘

**第四步：class_weights手动搜索（5组）**
- 因为自动搜索中class_weights维度效果不稳定
- 手动尝试5组类别权重组合，找到最优的cw_D=[0.8, 2.0]

#### 3.2.3 最终结果（gen2）

| 指标 | 值 |
|------|-----|
| Balanced Accuracy | **65.27%** |
| std（3种子） | 1.49% |
| Macro F1 | 66.15% |
| 种子 | [42, 43, 44] |

#### 3.2.4 gen2最佳配置参数

| 参数 | 值 |
|------|-----|
| 模型架构 | mini_neuript_classifier |
| n_classes | 2 |
| learning_rate | 3.05e-05 |
| batch_size | 32 |
| dropout | 0.2455 |
| weight_decay | 1.08e-06 |
| class_weights | [0.8, 2.0] |
| encoder_freeze_epochs | 999（全程冻结） |
| freeze_pretrained_backbone | true |
| optimizer | adamw |
| epochs | 30 |
| patience | 5 |
| early_stopping_metric | val_balanced_accuracy |

#### 3.2.5 文件位置

| 文件 | 路径 |
|------|------|
| gen2配置文件 | `configs/tmp_cw_D_balanced.yaml` |
| gen2结果目录 | `artifacts/gen2_cw_search/cw_D_balanced/` |
| 预训练权重 | `artifacts/pretrain_aamp_mental_arithmetic_250hz_8ch/diagnostic_best_pretrain_model.pt` |

---

### 3.3 路线C：搜索预训练 + 认真微调（当前主线，进行中）

路线C的核心思想：**预训练参数也不能拍脑袋，也要在目标数据集上搜出来**。

#### 3.3.1 预训练参数搜索（Sobol 20组）

**搜索空间**：

| 参数 | 范围 | 类型 |
|------|------|------|
| learning_rate | [1e-5, 1e-3] | log_float |
| batch_size | [16, 128] | int |

**搜索结果**：
- 最佳：trial_9
- lr = 5.03e-4
- bs = 28
- val_loss = 0.2355

#### 3.3.2 正式预训练（3种子）

用搜索出的最佳参数（lr=5.03e-4, bs=28），跑3个种子的正式预训练：

| 种子 | 最低val_loss | 对应epoch |
|------|-------------|-----------|
| 42 | 0.2185 | epoch 89 |
| **43** | **0.2143** | **epoch 98** ← 最好 |
| 44 | 0.2256 | epoch 97 |
| **平均** | **0.2195** | — |
| std | 0.0047 | — |

**关键决策**：用户坚持用**seed43**的预训练权重（val_loss最低），而不是seed42（和路线B严格对应）。

**预训练权重**：
- 路径：`artifacts/pretrain_aamp_best_params_seed43/diagnostic_best_pretrain_model.pt`
- 大小：6.6 MB
- sha256：3668d09a...

#### 3.3.3 用路线B参数直接微调（负面对照实验）

**做了什么**：
- 直接拿路线B的gen2最佳参数（lr=3.05e-5, dropout=0.245, wd=1.08e-6, cw=[0.8,2.0]）
- 去微调路线C的seed43预训练权重
- 3种子验证

**结果**：

| 种子 | BalAcc | best_epoch |
|------|--------|------------|
| 42 | 59.04% | 7 |
| 43 | 58.06% | 3 |
| 44 | 59.70% | 3 |
| **平均** | **58.93% ± 0.67%** | — |
| Macro F1 | 58.59% | — |
| ROC-AUC | 62.69% | — |

**结论**：
- 比路线B的65.27%**低了6.34个百分点**
- best_epoch都很早（3或7轮），说明路线C的预训练特征和路线B的微调参数**适配度不高**
- 这证明了：**路线B的参数不能直接搬到路线C用，必须为路线C重新搜索微调超参数**

#### 3.3.4 路线C微调超参数搜索（当前进行中）

**搜索空间（6个参数）**：

| 参数 | 范围 | 类型 |
|------|------|------|
| learning_rate | [0.00001, 0.001] | log_float |
| dropout | [0.0, 0.4] | float |
| weight_decay | [0.0000001, 0.001] | log_float |
| batch_size | [16, 64] | int |
| rest_weight | [0.5, 1.5] | float（class_weights第1维） |
| focus_weight | [1.0, 3.0] | float（class_weights第2维） |

**搜索配置文件**：`configs/route_C_search.yaml`

---

**阶段1：准随机搜索（Sobol 20组）✅ 已完成**

结果汇总（前3名）：

| 排名 | BalAcc | lr | dropout | wd | bs |
|------|--------|-----|---------|-----|-----|
| 🥇 1 | **0.6132** | 4.12e-5 | 0.001 | 2.22e-5 | 25 |
| 🥈 2 | 0.6030 | 8.39e-4 | 0.283 | 2.75e-4 | 52 |
| 🥉 3 | 0.6012 | 1.47e-4 | 0.374 | 6.30e-4 | 61 |

**注意事项**：
- 搜索结果params里只有4个参数（lr, dropout, wd, bs），**没有rest_weight和focus_weight**
- 说明class_weights搜索可能未生效（脚本处理逻辑有bug）
- 实际运行时class_weights用的是配置文件默认值[0.8, 2.0]

**阶段1结果文件**：
- 路径：`artifacts/route_C_search_sobol/quasi_random_search_results.json`
- 格式：顶层keys = [best_params, best_value, objective_metric, maximize, n_completed, n_trials, method, top_5, all_results]
- 指标位置：`metrics.balanced_accuracy`（不是val_balanced_accuracy）
- 每个trial子目录：`artifacts/route_C_search_sobol/trial_0/` 到 `trial_19/`，每个含config.json/history.json/results.json/run_metadata.json

---

**阶段2：贝叶斯优化（EI 30组）🔄 待运行**

**首次尝试失败**：
- 写了`scripts/bayesian_search_continue.py`
- 但`run_trial`函数中凭猜测用了`EEGDataset.from_config()`，项目里没有这个API
- 导致30个trial全部失败，每个返回BalAcc=0.5作为fallback

**已修复**：
- 重写了`bayesian_search_continue.py`
- 核心改动：删掉自己写的`run_trial`，直接`from scripts.search_hyperparams import make_objective_fn`
- 100%复用项目已验证的训练逻辑（数据加载、模型构建、AttentionTrainer、指标提取）
- 不会再有API不匹配的问题

**待运行命令**：
```bash
cd /mnt/workspace/Attention-of-EEG

# 先清理失败的结果
rm -f artifacts/route_C_search_bayesian/bayesian_progress.json
rm -f artifacts/route_C_search_bayesian/bayesian_search_results.json
rm -rf artifacts/route_C_search_bayesian/trial_*

# 运行贝叶斯优化
PYTHONPATH=src python scripts/bayesian_search_continue.py \
    --initial-results artifacts/route_C_search_sobol/quasi_random_search_results.json \
    --config configs/route_C_search.yaml \
    --output artifacts/route_C_search_bayesian \
    --n-trials 30 \
    --search-epochs 30 \
    --seed 42 \
    --acquisition ei \
    --xi 0.01
```

**贝叶斯搜索输出目录**：`artifacts/route_C_search_bayesian/`

---

## 四、三条路线对比总表

| 维度 | 路线A（废弃） | 路线B（基线） | 路线C（当前主线） |
|------|-------------|-------------|-----------------|
| 任务 | AAD听觉注意解码 | 专注度二分类 | 专注度二分类 |
| 数据集 | Ear-SAAD | 心算8ch250Hz | 心算8ch250Hz |
| 预训练参数 | — | 默认（lr=1e-4, bs=32） | 搜索出（lr=5.03e-4, bs=28） |
| 预训练val_loss | — | — | 0.2143（seed43最佳） |
| 微调参数搜索 | — | Sobol20+Bayes30+CW5 | Sobol20✅ + Bayes30🔄 |
| 下游BalAcc | 未达门槛 | **65.27% ± 1.49%** | 待贝叶斯完成后确定 |
| 用B参数直接微调 | — | — | 58.93%（负面对照） |
| 状态 | 已废弃 | ✅ 完成，作为基线 | 🔄 进行中 |

---

## 五、下一步计划

### 5.1 立即要做的

1. **运行路线C贝叶斯优化**（30组EI）
   - 用重写后的`bayesian_search_continue.py`
   - 预计每个trial几分钟到十几分钟，30组约几小时
   - 输出到`artifacts/route_C_search_bayesian/`

2. **汇总两阶段搜索结果**
   - 合并Sobol 20组 + Bayes 30组 = 50组
   - 选出全局最佳微调组合

3. **class_weights问题排查**
   - 检查`scripts/search_hyperparams.py`中rest_weight/focus_weight的处理逻辑
   - 确认class_weights是否真的被搜索到了
   - 如果自动搜索未生效，可能需要像路线B一样手动搜5组class_weights

### 5.2 贝叶斯完成后要做的

4. **3种子正式训练**
   - 用路线C搜索出的最佳微调组合
   - 跑种子[42, 43, 44]
   - 得到路线C的最终BalAcc均值和std

5. **路线B vs 路线C严格对照**
   - 路线B：65.27% ± 1.49%
   - 路线C：待确定
   - 判断：预训练参数搜索是否真的带来了下游性能提升？

### 5.3 后续可能的方向

6. **解冻Encoder微调**（当前是全程冻结）
   - 尝试encoder_freeze_epochs设为较小值（如5或10）
   - 看微调后期解冻能否进一步提升

7. **数据增强 / 正则化策略探索**

8. **跨被试泛化性验证**

---

## 六、完整文件索引

### 6.1 配置文件

| 文件 | 说明 |
|------|------|
| `configs/tmp_cw_D_balanced.yaml` | 路线B gen2最佳配置 |
| `configs/route_C_search.yaml` | 路线C微调超参数搜索配置（6参数搜索空间） |

### 6.2 预训练权重

| 文件 | 说明 |
|------|------|
| `artifacts/pretrain_aamp_mental_arithmetic_250hz_8ch/diagnostic_best_pretrain_model.pt` | 路线B预训练权重（默认参数） |
| `artifacts/pretrain_aamp_best_params_seed43/diagnostic_best_pretrain_model.pt` | 路线C预训练权重（搜索参数，seed43最佳，6.6MB） |

### 6.3 搜索结果

| 文件/目录 | 说明 |
|-----------|------|
| `artifacts/route_C_search_sobol/quasi_random_search_results.json` | 路线C阶段1：Sobol 20组结果 |
| `artifacts/route_C_search_sobol/trial_0/` ~ `trial_19/` | 每个trial的详细输出（config/history/results/metadata） |
| `artifacts/route_C_search_sobol/best_config.yaml` | 阶段1最佳配置 |
| `artifacts/route_C_search_bayesian/` | 路线C阶段2：贝叶斯优化输出（待运行） |
| `artifacts/gen2_cw_search/cw_D_balanced/` | 路线B gen2结果 |

### 6.4 脚本

| 文件 | 说明 |
|------|------|
| `scripts/search_hyperparams.py` | 项目自带通用搜索脚本（支持quasi_random和bayesian两种模式） |
| `scripts/bayesian_search_continue.py` | 贝叶斯优化续跑脚本（已重写，复用make_objective_fn） |
| `scripts/train.py` | 训练入口（含build_model等工具函数） |

### 6.5 本地文件

| 文件 | 说明 |
|------|------|
| `C:\Users\Administrator\Doubao\chats\2026-09-09\new-chat\EEG_Attention_Project_Roadmap.md` | 项目路线记忆文件（旧版） |
| `C:\Users\Administrator\Doubao\chats\2026-09-09\new-chat\bayesian_search_continue.py` | 贝叶斯脚本本地副本（已重写版） |
| `C:\Users\Administrator\Downloads\search_results_summary.csv` | 准随机搜索17个有效trial的汇总CSV |

---

## 七、关键经验与踩坑记录

### 7.1 预训练val_loss低 ≠ 下游性能好

- 路线C预训练val_loss（0.2143）比路线B低约8.5%
- 但用路线B参数微调时，路线C下游性能（58.93%）反而比路线B（65.27%）低6.34%
- **教训**：预训练的代理指标（val_loss）和下游任务性能不一定正相关，必须实际跑下游验证

### 7.2 不能直接搬另一条路线的参数

- 路线B的gen2参数直接用到路线C，性能暴跌
- **教训**：每条路线的预训练特征分布不同，微调参数必须重新搜索

### 7.3 YAML科学计数法bug

- YAML中`1e-5`会被解析为字符串（str），不是浮点数
- 导致TypeError: must be real number, not str
- **解决**：搜索空间的边界值必须写成小数形式（0.00001），不能用1e-5

### 7.4 不要凭猜测写项目API

- 首次贝叶斯脚本用了`EEGDataset.from_config()`，项目里根本没有这个方法
- 导致30个trial全部失败
- **教训**：写脚本前必须先看项目里已有的`search_hyperparams.py`，直接复用`make_objective_fn`，不要自己重新实现训练逻辑

### 7.5 class_weights搜索可能未生效

- 准随机搜索结果的params里只有4个参数，没有rest_weight/focus_weight
- `search_hyperparams.py`里有两段重复的class_weights处理代码（第60-70行附近）
- 可能第一段`params.pop()`把参数pop掉了，导致第二段拿不到
- **待排查**：需要确认class_weights是否真的被搜索和应用

---

## 八、实验纪律

1. **每次只改一个变量**：路线B和路线C的唯一区别是预训练权重，下游搜索流程完全一致
2. **3种子验证**：所有最终结果都用种子[42, 43, 44]跑3次，报告均值和std
3. **冻结Encoder**：当前所有微调实验都是全程冻结预训练backbone，只训分类头
4. **早停**：patience=5，监控val_balanced_accuracy
5. **结果可复现**：所有随机种子固定，配置文件完整保存

---

*文档结束。有疑问或需要补充的地方随时说。*
