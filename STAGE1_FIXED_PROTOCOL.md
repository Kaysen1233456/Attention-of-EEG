# Ear-SAAD Stage 1 固定执行协议

本文档固定 B0/B1 教师路线的实验协议。训练不会由代码自动启动，所有
训练命令由用户手工执行。当前协议不读取 test，不覆盖旧产物。

## 固定协议

- 数据目录：`data/processed/ear_saad`
- 开发折：`configs/development_folds.json`
- 模型：B0 现有双分支 CNN；B1 随机初始化 Transformer、无 PMoE、均值池化、普通 MLP
- epoch budget：25
- seed：42、43、44
- precision：CUDA 上 `highest`；AMP 按配置启用
- normalization：`train_subjects_only`
- zero-channel policy：`retain`
- checkpoint 选择：validation balanced accuracy 最佳 epoch
- test：所有 Stage 1 运行禁止读取
- 输出根目录：`artifacts/stage1_b0_b1_cv_v1`

每个运行保存 `config.yaml`、`config.json`、`run_metadata.json`、
`history.json`、`results.json` 和 `best_model.pt`。`results.json` 同时包含
最佳 checkpoint 的 train/validation 指标、三项 gap 和每个 validation
subject 的 BA、Macro-F1、ROC-AUC、类别 recall 与混淆矩阵。多 seed 根目录
额外保存 `summary.json`，其中包含每个 seed 的验证指标、训练指标、gap 和
checkpoint 路径。

## 执行顺序

### Step 1：B0 三折 seed 42

```bash
for fold in 0 1 2; do
  PYTHONPATH=src python scripts/train.py \
    --config configs/teacher_ear_saad.yaml \
    --ablation B0 \
    --fold-manifest configs/development_folds.json \
    --fold-index "$fold" \
    --epochs 25 \
    --seeds 42 \
    --device cuda \
    --output "artifacts/stage1_b0_b1_cv_v1/B0/fold_${fold}/seed_42"
done
```

### Step 2：B1 三折 seed 42

```bash
for fold in 0 1 2; do
  PYTHONPATH=src python scripts/train.py \
    --config configs/teacher_ear_saad.yaml \
    --ablation B1 \
    --fold-manifest configs/development_folds.json \
    --fold-index "$fold" \
    --epochs 25 \
    --seeds 42 \
    --device cuda \
    --output "artifacts/stage1_b0_b1_cv_v1/B1/fold_${fold}/seed_42"
done
```

完成后检查 B1 平均 BA 是否接近既有参考值 `0.5411`。该参考值来自
`artifacts/B1_corrected_folds_v1/summary.json`。

### Step 3：B0/B1 三折 seed 43、44

```bash
for variant in B0 B1; do
  for seed in 43 44; do
    for fold in 0 1 2; do
      PYTHONPATH=src python scripts/train.py \
        --config configs/teacher_ear_saad.yaml \
        --ablation "$variant" \
        --fold-manifest configs/development_folds.json \
        --fold-index "$fold" \
        --epochs 25 \
        --seeds "$seed" \
        --device cuda \
        --output "artifacts/stage1_b0_b1_cv_v1/${variant}/fold_${fold}/seed_${seed}"
    done
  done
done
```

### Step 4：生成统一报告

```bash
PYTHONPATH=src python scripts/summarize_stage1.py \
  --root artifacts/stage1_b0_b1_cv_v1
```

输出：

- `artifacts/stage1_b0_b1_cv_v1/reports/stage1_summary.json`
- `artifacts/stage1_b0_b1_cv_v1/reports/subject_level_summary.json`

### Step 5：教师保留决策

自动判定要求：

- B1 平均 validation BA 相对 B0 至少提升 `0.01`；
- 至少 2/3 fold 提升；
- B1 Macro-F1 和 ROC-AUC 不显著下降，当前阈值为不低于 B0 `0.01`；
- validation 混淆矩阵中不能有类别 recall 低于 `0.10` 的 run。

```bash
python -m json.tool \
  artifacts/stage1_b0_b1_cv_v1/reports/stage1_summary.json
```

只有 `teacher_retain_decision` 为 `PASS` 才进入 Step 6。

### Step 6：条件性测试 temporal_pool=2

该步骤只在 Step 5 通过后执行。它是 B1 的单因素变化，仍使用三折、
三个 seed 和 25 epoch。

```bash
for fold in 0 1 2; do
  PYTHONPATH=src python scripts/train.py \
    --config configs/teacher_ear_saad.yaml \
    --ablation B1 \
    --temporal-pool 2 \
    --fold-manifest configs/development_folds.json \
    --fold-index "$fold" \
    --epochs 25 \
    --seeds 42 43 44 \
    --multi-seed \
    --device cuda \
    --output "artifacts/stage1_b1_pool2_cv_v1/fold_${fold}"
done
```

如果 pool 2 没有稳定提升，则保留原 B1，不测试 pool 1、卷积前端、PMoE
或更复杂分类头。

### Step 7：锁定教师并进行 CNN 蒸馏

只有在 B1 或 pool 2 通过后进入。当前仓库已有蒸馏 loss/projector 接口，
但尚未有完整、经过测试的蒸馏训练入口，因此本步骤暂不提供虚假的可执行
训练命令。进入本步骤前必须补齐并测试 frozen teacher 加载、CNN supervised
only 对照、logits KD、bilateral feature loss，以及三折三 seed 报告。
