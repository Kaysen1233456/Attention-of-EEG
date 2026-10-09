# EEG Attention Decoding

> ⚠️ **状态：本 README 仍在整理中，下面混杂了两条路线，请先看清属于哪一条。**
> - **当前主线**：心算（Mental Arithmetic）数据集 —— 见下面的「术语澄清」
> - **已放弃**：Ear-SAAD 耳周听觉注意解码 —— 见「历史归档」，仅作记录，不要再执行

## 术语澄清：Gen5、Teacher B、「学生」、蒸馏

**当前主线（心算数据集）只有两个模型，它们是两个各自独立训练的分类模型：**

| | Gen5 | Teacher B |
|---|---|---|
| 时序分支 | temporal CNN（2 层 Conv1d） | **多尺度** CNN（3 条并行分支，卷积核 7/15/31）+ 1 层 Transformer |
| 频谱分支 | 相对频带功率 → MLP | 相对频带功率 → MLP |
| 参数量 | **8,418**（32.9 KB fp32） | **193,202**（754.7 KB fp32），**23 倍** |
| 训练方式 | 标签直接监督（class-balanced cross entropy） | 同左 |
| 同一评估口径 BA | **0.7232** | 0.7130 |

**四点关键澄清：**

1. **Gen5 与 Teacher B 之间目前没有任何关系。** 两者各自用标签做直接监督训练，
   Teacher B **从未**监督过 Gen5，也从未指导过它的训练。
2. **「Gen5 是学生」只是将来蒸馏阶段的角色预设，不是现状。**
   这个称呼来自**已放弃的 Ear-SAAD 路线** —— 那条路线真的有
   「Transformer 教师 → 轻量双分支 CNN 学生 → 耳戴设备」的蒸馏计划。
   它被沿用到了心算路线，但**心算路线的蒸馏从未启动过**。
3. **「直接监督训练」= 只用标签算交叉熵**；「蒸馏」= 用教师的输出
   （软标签 / 中间特征）当监督信号。这是两种不同的训练方式，
   **不要混用**。
4. **当前能代表模型能力的是 Gen5，不是 Teacher B。** Teacher B 尚未通过验收门槛
   （要求 3 seed 平均 BA ≥ 0.72，实测 0.6991），且在同一评估口径下低于 Gen5。
   **教师这一步目前没有产出可用的增益。**

Gen5 的固定模型、改动原因、交叉验证结果和最终测试协议记录在
[GEN5_PROTOCOL.md](GEN5_PROTOCOL.md)；教师路线的门槛与阶段见
[TEACHER_PROTOCOL.md](TEACHER_PROTOCOL.md)。

> 相关证据与作废记录：`artifacts/probes/_superseded/README.md`、
> `experiments_log/INVALID_METHODS_REGISTRY.md`、`PROJECT_ISSUES_PLAIN.md`。

---

## 历史归档：Ear-SAAD 路线（已放弃）

> ⚠️ **以下内容全部属于已放弃的 Ear-SAAD 路线**（19 个耳周通道、20 Hz、
> 5 秒窗口、二分类）。与当前心算主线无关，**不应再执行、训练或蒸馏**。

基于 Ear-SAAD 数据集的跨被试听觉注意解码项目。项目借鉴 NeurIPS 2025
NeurIPT 的 3D 电极嵌入、IILP 和 PMoE 思路，但所有组件都必须通过监督式、
被试独立的开发实验验证后才能保留。

该路线当时的设想是：

```text
监督式 Transformer 教师
        -> 左右耳双边特征与 logits
        -> 轻量级双分支 CNN 学生
        -> 耳戴式设备部署
```

### 已确定的原则

- 数据集是 Ear-SAAD：19 个 around-ear 通道、20 Hz、5 秒窗口、二分类。
- 训练、验证、测试按被试划分，不能把窗口视为独立被试。
- 当前开发实验只使用原始 `train` 和 `val` 被试，固定三折 8/4 被试交叉验证。
- 测试被试在架构和超参数选择阶段完全不读取。
- 当前监督教师从随机初始化开始训练。
- 之前的 AAMP 重建 checkpoint 已否决，不得用于新的下游实验。
- 不能把 Transformer encoder 权重直接加载到 CNN 卷积层。
- `subject_trial_*` 指标只作诊断；不能用验证标签优化出的阈值选择 checkpoint。

### 阶段一：监督教师验证

教师模型使用 `MiniNeurIPTClassifier`：

```text
19 通道 EEG
  -> 3D 电极嵌入
  -> temporal pooling
  -> Transformer encoder
  -> 左耳 IILP + 右耳 IILP
  -> [L, R, |L-R|, L*R]
  -> MLP 或 SwiGLU 分类器
```

架构消融按 B0-B5 执行，每次只增加一个组件：

| 变体 | 增量 |
| --- | --- |
| B0 | 现有双分支 CNN |
| B1 | 随机初始化 Transformer，无 PMoE，均值池化，普通 MLP |
| B2 | B1 + 独立左右 IILP 池化 |
| B3 | B2 + `|L-R|` 和 `L*R` 双边融合 |
| B4 | B3 + PMoE |
| B5 | B4 + SwiGLU 分类头 |

当前工程筛选门槛是开发三折平均 balanced accuracy 至少提升 `0.01`，
至少两折提升，且不能出现类别坍缩。教师只有在多折、多 seed 稳定达到
开发 BA `>= 0.60` 并优于同协议 CNN 后，才进入蒸馏阶段。

### 阶段二：教师到学生蒸馏

教师通过 logits、左右双边特征监督轻量双分支 CNN 学生：

```text
CE(labels) + KD(teacher logits, temperature)
             + bilateral feature loss
```

蒸馏头只在训练期间使用，部署模型不包含投影头。

## 数据

预处理后的 Ear-SAAD 数据应具有以下结构：

```text
data/processed/ear_saad/
├── train_waveforms.npy
├── train_labels.npy
├── train_subjects.npy
├── train_trials.npy
├── val_waveforms.npy
├── val_labels.npy
├── val_subjects.npy
├── val_trials.npy
├── test_waveforms.npy
├── test_labels.npy
├── test_subjects.npy
└── test_trials.npy
```

数组约定：

```text
waveforms: [N, 19, 100]
labels:    [N], 0/1
subjects:  [N]
trials:    [N]
```

19 个 around-ear 通道按左耳 9 通道、右耳 10 通道排列。坐标来自实验手册
通道布局的归一化模板，不是每个被试的实测 3D 电极坐标。论文和报告中必须
使用准确表述：

```text
source: cEEGrid layout-order approximate template
frame: ear_local_normalized_xyz
```

数据归一化只使用当前训练折被试的统计量，并按 8 个标准差裁剪。窗口局部
归一化已被开发实验否决，不能作为默认策略。

## 环境安装

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
pip install pytest
```

需要 Python 3.10+、PyTorch、NumPy、SciPy、scikit-learn、PyYAML 和 tqdm。

## 验证代码

在项目根目录运行：

```bash
python -m pytest -q
python -m compileall -q src scripts tests
```

训练前先审计开发数据并生成固定折：

```bash
python scripts/audit_development.py \
  --data data/processed/ear_saad \
  --output artifacts/development_audit_v1
```

该命令会生成：

```text
artifacts/development_audit_v1/audit.json
artifacts/development_audit_v1/folds.json
```

审计必须确认开发被试来自原始 train/val，测试被试未进入折叠清单；同时记录
重复窗口、非有限值、标签一致性、被试统计量和缺失通道代理信息。

## 可学习性诊断

该诊断只检查模型能否记住训练集固定小样本，不是泛化实验，也不产生正式
模型结论：

```bash
python scripts/diagnose_learnability.py \
  --config configs/teacher_ear_saad.yaml \
  --output artifacts/learnability_v1 \
  --device cuda
```

该诊断固定每类 16 个训练窗口，关闭 dropout、AMP 和权重衰减，并记录 CE、
accuracy、每类 recall、概率标准差、梯度范数和学习率。默认最多 300 步，
只有达到 accuracy `>= 0.98` 且 CE `< 0.1` 才标记为 `passed`。

失败只说明当前诊断预算或优化设置不足，不能直接推出模型不可学。

## 教师训练

默认教师配置位于 `configs/teacher_ear_saad.yaml`，其中 AAMP checkpoint 路径
为空，模型从随机初始化开始：

```bash
python scripts/train.py \
  --config configs/teacher_ear_saad.yaml \
  --epochs 25 \
  --seeds 42 \
  --device cuda \
  --output artifacts/teacher_seed42
```

固定三折开发实验：

```bash
python scripts/train.py \
  --config configs/teacher_ear_saad.yaml \
  --ablation B1 \
  --fold-manifest artifacts/development_audit_v1/folds.json \
  --fold-index 0 \
  --epochs 25 \
  --seeds 42 \
  --device cuda \
  --output artifacts/ablation/B1/fold0_seed42
```

每次运行会保存 `config.yaml`、`config.json`、`run_metadata.json`、
`history.json`、`results.json` 和 `best_model.pt`。`results.json` 中的
`best_checkpoint_train_metrics` 与 `best_val_metrics` 来自同一个最佳
checkpoint，可用于计算真实的训练/验证差距。在线训练阶段的 batch accuracy
不能代替这个诊断。

多 seed 使用独立子进程，每个 seed 都重新构造模型、数据加载器、优化器、
调度器和 AMP scaler：

```bash
python scripts/train.py \
  --config configs/teacher_ear_saad.yaml \
  --fold-manifest artifacts/development_audit_v1/folds.json \
  --fold-index 0 \
  --seeds 42 43 44 \
  --multi-seed \
  --epochs 25 \
  --device cuda \
  --output artifacts/teacher_fold0_seeds
```

开发实验默认不加载测试集。只有路线文档明确允许的最终评估阶段，才显式
添加 `--evaluate-test`。当前测试被试属于 exposed holdout，历史测试结果
不能当作全新的盲测结果。

B0/B1 三折三 seed 的固定执行协议、完整训练命令、结果目录和教师保留判定，
统一见 `STAGE1_FIXED_PROTOCOL.md`。该协议不会自动启动训练。

## 消融和搜索

按 B0-B5 顺序运行，保持相同 seed、折、预算、精度和归一化策略。不要在
没有通过审计和固定折验证前启动大规模搜索。

```bash
python scripts/train.py --config configs/teacher_ear_saad.yaml \
  --ablation B0 --fold-manifest artifacts/development_audit_v1/folds.json \
  --fold-index 0 --epochs 25 --seeds 42 --device cuda \
  --output artifacts/ablation/B0/fold0_seed42
```

只有选出最强变体后，才进行最多 16 次的验证集超参数搜索：

```bash
python scripts/search_hyperparams.py \
  --method quasi_random \
  --config configs/teacher_ear_saad.yaml \
  --ablation B1 \
  --n-trials 16 \
  --search-epochs 25 \
  --output artifacts/search_teacher_from_scratch \
  --seed 42
```

搜索参数中学习率和 weight decay 使用 log space。搜索和架构选择不能读取
测试集。

## 项目结构

```text
Attention/
├── configs/
├── scripts/
│   ├── audit_development.py
│   ├── diagnose_learnability.py
│   ├── preprocess_ear_saad.py
│   ├── train.py
│   ├── search_hyperparams.py
│   └── evaluate.py
├── src/attention_model/
├── tests/
├── IMPLEMENTATION_PLAN.md
├── IMPLEMENTATION_STATUS.md
├── PROJECT_STATE.md
└── TEACHER_NEXT_STEPS.md
```

## 路线文档

以下文件是当前实验执行的依据，其中计划和状态优先级最高：

1. `IMPLEMENTATION_PLAN.md`
2. `IMPLEMENTATION_STATUS.md`
3. `PROJECT_STATE.md`
4. `TEACHER_NEXT_STEPS.md`

任何性能结论都必须同时记录配置、seed、折、训练/验证被试、指标和限制。

## 参考

- NeurIPS 2025 NeurIPT: Foundation Model for Neural Interfaces
- Scientific Reports 2025: A Direct Comparison of Simultaneously Recorded
  Scalp, Around-Ear and In-Ear EEG for AAD
- Ear-SAAD dataset: <https://zenodo.org/records/16536441>

## 许可证

MIT License
