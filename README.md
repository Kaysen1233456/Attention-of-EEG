# EEG Attention Decoding

基于 Ear-SAAD 数据集的跨被试听觉注意解码项目。项目借鉴 NeurIPS 2025
NeurIPT 的 3D 电极嵌入、IILP 和 PMoE 思路，但所有组件都必须通过监督式、
被试独立的开发实验验证后才能保留。

当前目标不是在随机窗口划分上追求高分，而是建立可复现的：

```text
监督式 Transformer 教师
        -> 左右耳双边特征与 logits
        -> 轻量级双分支 CNN 学生
        -> 耳戴式设备部署
```

## 当前研究路线

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
