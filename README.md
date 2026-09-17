# EEG 专注度实时监测与 BBNE 闭环增强模型

> 基于 NeurIPS 2025 NeurIPT 设计思想的轻量级 EEG 注意力解码框架
>
> 阶段一：双分支轻量模型（直接训练，~50K 参数）
> 阶段二：迷你版 NeurIPT 自监督预训练 → 微调 → 蒸馏

---

## 目录

- [项目简介](#项目简介)
- [核心设计](#核心设计)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [模型架构](#模型架构)
- [超参数搜索](#超参数搜索)
- [两阶段路线图](#两阶段路线图)
- [参考文献](#参考文献)

---

## 项目简介

本项目旨在搭建一个**专注度实时监测与 BBNE（双耳拍频神经夹带）闭环增强系统**，通过入耳式 4 通道 EEG 信号实时判断用户的注意力状态，并在注意力下降时触发 BBNE 音频刺激进行闭环增强。

### 设计依据

本项目的核心设计思想来自 **NeurIPS 2025 NeurIPT** 论文（Foundation Model for Neural Interfaces），提取了四个关键创新并做轻量化适配：

1. **3D 电极嵌入**（3D Electrode Embedding）：用电极三维物理坐标做 sin/cos 空间编码，支持跨数据集、跨通道配置的泛化
2. **AAMP 振幅感知掩码**（Amplitude-Aware Masked Pretraining）：按信号振幅而非随机间隔掩码，逼模型学脑电节律而非局部插值
3. **IILP 左右耳分组池化**（Intra-Inter Lobar Pooling）：左耳内、右耳内独立池化，左右间拼接融合，利用脑区功能特异性
4. **分层注意力 + 渐进式混合专家**：时间注意力 + 空间注意力，浅层少专家、深层多专家

---

## 核心设计

### 1. 双分支模型架构（阶段一）

```
输入 EEG [batch, 4, 500]
    │
    ├── 左耳分支: 通道 0,1 (L04, L05)
    │   └── 3D嵌入 → Conv1d(32,k=5,s=2) → GELU → Conv1d(64,k=3,s=2) → GELU
    │       └── 多层特征拼接: [第1层GAP(32), 第2层mean(64), 第2层std(64)] = 160维
    │
    ├── 右耳分支: 通道 2,3 (R04, R05)  (参数独立，不共享)
    │   └── 同上结构
    │
    └── 融合层:
        拼接 [左耳特征(160), 右耳特征(160), |左耳-右耳|(160), 左耳*右耳(160)] = 640维
            │
            ▼
        MLP: 640 → 64 → 3 (注意力3分类)
```

**参数量：约 50K**（NPU 部署友好，和 AAD Gen2.0 的 35682 同量级）

### 2. 3D 电极嵌入

耳内 4 通道的默认三维坐标（相对坐标）：

| 通道 | x (左右) | y (前后) | z (上下) |
|------|----------|----------|----------|
| L04  | -1.0     | 0.5      | 0.5      |
| L05  | -1.0     | 0.5      | -0.5     |
| R04  | 1.0      | 0.5      | 0.5      |
| R05  | 1.0      | 0.5      | -0.5     |

每个坐标用 `d_model/3` 维 sin/cos 编码，三个坐标拼接。

### 3. AAMP 振幅感知掩码

- 动态掩码比例：`[0.2, 0.35, 0.5]`（NeurIPT 原配置）
- 按振幅降序排序，随机百分位中心掩码
- BERT 式 80/10/10 策略：80% [mask]、10% 随机、10% 不变

### 4. IILP 左右耳分组池化

- **脑叶内**：左耳 L04+L05 独立处理，右耳 R04+R05 独立处理（双分支结构体现）
- **脑叶间**：左右特征拼接 + 差值 + 乘积融合

---

## 快速开始

### 环境要求

```bash
Python 3.10+
PyTorch 2.0+
NumPy, SciPy, scikit-learn
```

### 安装

```bash
# 克隆仓库
git clone <repo-url>
cd Attention

# 安装依赖
pip install torch numpy scipy scikit-learn pyyaml tqdm
```

### 冒烟测试（用合成数据验证代码能跑通）

```bash
# 训练（合成数据）
python scripts/train.py --synthetic --output artifacts/smoke_test --epochs 3

# 预训练（合成数据）
python scripts/pretrain_aamp.py --synthetic --output artifacts/pretrain_smoke --epochs 3

# 超参数搜索（合成数据）
python scripts/search_hyperparams.py --synthetic --n-trials 5 --search-epochs 2
```

### 用真实数据训练

```bash
# 1. 准备数据（和 AAD 项目格式一致）
# data/processed/
# ├── train_waveforms.npy   [N_train, 4, 500]
# ├── train_labels.npy      [N_train]
# ├── train_subjects.npy    [N_train]
# ├── val_waveforms.npy
# ├── val_labels.npy
# ├── test_waveforms.npy
# └── test_labels.npy

# 2. 训练
python scripts/train.py --data data/processed --output artifacts/baseline --epochs 100

# 3. 多种子训练
python scripts/train.py --data data/processed --output artifacts/multi_seed --multi-seed --seeds 42 43 44

# 4. 评估
python scripts/evaluate.py --model-path artifacts/baseline/best_model.pt --data data/processed
```

### 超参数搜索

```bash
# 准随机搜索（Sobol序列，推荐先用这个）
python scripts/search_hyperparams.py \
    --method quasi_random \
    --quasi-method sobol \
    --n-trials 50 \
    --data data/processed \
    --output artifacts/search_sobol

# 贝叶斯优化（在准随机找到好的区间后用这个精调）
python scripts/search_hyperparams.py \
    --method bayesian \
    --acquisition ei \
    --n-trials 30 \
    --n-initial 10 \
    --data data/processed \
    --output artifacts/search_bayesian
```

### 运行单元测试

```bash
pip install pytest
pytest tests/ -v
```

---

## 项目结构

```
Attention/
├── README.md                          # 本文件
├── requirements.txt                   # 依赖
├── configs/
│   ├── baseline.yaml                  # 基线配置
│   └── search_space.yaml              # 搜索空间配置
├── src/
│   └── attention_model/
│       ├── __init__.py
│       ├── config.py                  # 全局配置类（借鉴 ms-swift 配置驱动设计）
│       ├── models/
│       │   ├── __init__.py
│       │   ├── embedding_3d.py        # 3D 电极嵌入（NeurIPT 创新①）
│       │   ├── dual_branch_attention.py  # 双分支注意力模型（阶段一核心）
│       │   └── mini_neuript.py        # 迷你版 NeurIPT（阶段二预训练用）
│       ├── data/
│       │   ├── __init__.py
│       │   ├── dataset.py              # 数据集和数据加载
│       │   └── aamp_masking.py         # AAMP 振幅感知掩码（NeurIPT 创新②）
│       ├── training/
│       │   ├── __init__.py
│       │   ├── trainer.py              # 训练器（多种子、早停、AMP、梯度裁剪）
│       │   └── losses.py               # 损失函数（交叉熵、一致性正则化、AAMP重建）
│       ├── evaluation/
│       │   ├── __init__.py
│       │   └── metrics.py              # 评估指标（窗口级+被试级）
│       └── search/
│           ├── __init__.py
│           ├── quasi_random.py          # 准随机搜索（Sobol/Halton/Latin Hypercube）
│           └── bayesian_optimization.py # 贝叶斯优化（GP + EI/UCB/POI）
├── scripts/
│   ├── train.py                        # 训练入口
│   ├── pretrain_aamp.py                # AAMP 自监督预训练入口
│   ├── search_hyperparams.py           # 超参数搜索入口
│   └── evaluate.py                     # 评估入口
├── tests/
│   ├── test_embedding_3d.py
│   ├── test_aamp_masking.py
│   ├── test_dual_branch.py
│   └── test_search.py
├── artifacts/                           # 实验产物（模型、结果、日志）
└── data/
    ├── raw/                             # 原始数据
    └── processed/                       # 预处理后数据
```

---

## 模型架构

### 阶段一：双分支轻量模型（当前主路线）

| 组件 | 配置 |
|------|------|
| 嵌入 | 3D 电极嵌入，d_model=32 |
| 分支1（左耳） | Conv1d(32→32,k=5,s=2) → GELU → Conv1d(32→64,k=3,s=2) → GELU |
| 分支2（右耳） | 同上，参数独立 |
| 多层特征拼接 | 第1层GAP(32) + 第2层mean(64) + 第2层std(64) = 160维/分支 |
| 融合 | [左耳, 右耳, \|差\|, 乘积] = 640维 |
| 分类头 | 640 → 64 → 3 |
| 总参数量 | ~50K |

### 阶段二：迷你版 NeurIPT（预训练用）

| 组件 | 配置 |
|------|------|
| 嵌入 | 3D 电极嵌入，d_model=128 |
| Encoder 层数 | 4层 |
| 注意力头 | 8头 |
| FFN 维度 | 512 |
| 专家配置 | [0, 2, 2, 4]（渐进式，浅层少专家） |
| 预训练任务 | AAMP 振幅感知掩码重建 |
| 总参数量 | ~1.5M |

---

## 超参数搜索

### 默认搜索空间

| 参数 | 范围 | 类型 |
|------|------|------|
| learning_rate | [1e-4, 5e-3] | float |
| batch_size | [16, 64] | int |
| d_model | [24, 48, 72, 96] | categorical |
| branch_conv1_out | [16, 64] | int |
| branch_conv2_out | [32, 128] | int |
| fusion_hidden | [32, 128] | int |
| dropout | [0.0, 0.3] | float |

### 搜索策略建议

1. **先用准随机搜索（Sobol，50次）**：全面覆盖搜索空间，找到好的参数区间
2. **再用贝叶斯优化（30次）**：在好的区间内精调
3. **目标指标**：`val_balanced_accuracy`（最大化）

---

## 两阶段路线图

### 阶段一：双分支模型直接训练（当前）

- [x] 项目搭建
- [x] 3D 电极嵌入
- [x] AAMP 掩码模块
- [x] 双分支模型（含 IILP 左右耳分组池化）
- [x] 训练器（多种子、早停、AMP）
- [x] 评估指标（窗口级+被试级）
- [x] 准随机搜索 + 贝叶斯优化
- [ ] 寻找合适的公开数据集
- [ ] 基线实验
- [ ] 超参数搜索
- [ ] 消融实验

### 阶段二：自监督预训练 → 微调 → 蒸馏

- [x] 迷你版 NeurIPT 模型
- [x] AAMP 预训练入口
- [ ] 收集无标签 EEG 数据（多个公开数据集混合）
- [ ] 自监督预训练
- [ ] 微调下游注意力分类任务
- [ ] 知识蒸馏到双分支小模型（部署用）

---

## 参考文献

1. **NeurIPS 2025 NeurIPT**: Foundation Model for Neural Interfaces
   - 核心创新：3D Electrode Embedding, AAMP, PMoE, IILP
   - 73.5M 参数，2000+ 小时 EEG 预训练，8个下游数据集6个SOTA

2. **AAD 项目经验**（本团队前期工作）
   - Gen2.0 基线：mean_std_pool，被试级 94.44%，窗口级 68.28%
   - 已验证有害：标签平滑、SE注意力、深度可分离卷积、EMA、BatchNorm、warmup+cosine

---

## 许可证

MIT License
