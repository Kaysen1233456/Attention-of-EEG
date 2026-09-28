# 心算数据频域掩码预训练计划

## 目标与边界

当前数据是 250 Hz、8 通道、每窗 500 点的心算 EEG，标签为静息态/专注态二分类。预训练阶段不读取类别标签，学习目标是遮挡 alpha 频带后恢复被遮挡频点；下游三条分类路线共享预训练 encoder 表征。频域重建 loss 用于比较预训练超参数，不等同于二分类性能指标。

本计划将 validation 用作超参数搜索的固定比较集，搜索训练只使用 train split；test split 不进入搜索、早停或权重选择。完成参数选择后，正式预训练使用 train+validation 的无标签波形、固定轮数训练，不按 test 指标选择 checkpoint。

## 目标函数

每个窗口做 rFFT，按配置清零 8–13 Hz alpha 频点，再 irFFT 成输入信号。Encoder 编码遮挡信号，decoder 预测完整 rFFT 的实部和虚部。损失只在被遮挡的 alpha 频点计算复数幅值误差。固定 validation mask seed 保证 trial 间评价可比较。

每个 trial 以最低 validation loss 对应 epoch 计算：

`objective = best_val_loss + 0.25 * max(0, best_val_loss - train_loss_at_best_val)`

目标越低越好。惩罚项用于降低明显过拟合配置被选中的机会。最终下游仍须通过被试隔离的二分类评估比较三条路线，不能仅凭重建 loss 宣称分类最优。

## 搜索参数

准随机阶段用 scrambled Sobol 均匀覆盖参数空间；贝叶斯阶段把成功的 Sobol 试验作为观测，用 Matérn 高斯过程和 Expected Improvement 选下一组参数。

| 参数 | 搜索范围 |
|---|---|
| learning rate | `5e-5` 到 `1e-3`，对数尺度 |
| batch size | 8、16、24、32 |
| weight decay | `1e-6` 到 `1e-2`，对数尺度 |
| dropout | 0.0 到 0.3 |
| d_model | 48、72、96 |
| encoder layers | 2、3、4 |
| temporal pool | 2、4、8 |

每组参数都沿用同一心算数据配置、alpha mask、train-only 训练池、validation split、随机种子和每 trial epoch 上限。建议先做 12 个 Sobol trial，再做 12 个贝叶斯 trial；资源有限时可先用 6+6 验证流程。

## 实验步骤

1. 检查 CUDA、数据路径和配置；跑短 epoch 搜索，确认没有 OOM 或路径错误。
2. 运行 Sobol 探索并启动贝叶斯优化。每次 trial 的配置、history、stdout 日志和结果 JSON 独立存放。
3. 检查成功/失败 trial、验证 loss 曲线与最优参数。必要时增加搜索预算，避免单个 seed 的偶然最优。
4. 用最优参数在 train+validation 无标签数据上固定 epoch 正式预训练，生成 `final_pretrain_model.pt`。
5. 校验 checkpoint SHA-256、预训练元数据和频域目标配置。然后把 encoder 权重接入三条下游路线，以被试隔离验证集做专注/静息二分类比较，最终 test 只做一次最终评估。

## 产物

- 搜索：`search_protocol.json`、`search_results.json`、`search_summary.json`、`best_pretrain_params.json`。
- 每个 trial：`pretrain_config.yaml`、`pretrain_history.json`、`pretrain_metadata.json`、`run.log`、诊断/验证 checkpoint。
- 正式预训练：`final_pretrain_model.pt`、`pretrain_config.yaml`、`pretrain_metadata.json`、`pretrain_history.json`。

## 执行命令

在仓库根目录运行。先做 1 Sobol + 1 Bayesian 的管线 smoke test时，脚本要求至少 4 个 Sobol 点；可设 4+1、2 epoch：

```bash
cd /mnt/workspace/Attention-of-EEG
PYTHONPATH=src python scripts/search_mental_arithmetic_pretrain.py \
  --config configs/pretrain_aamp_mental_arithmetic.yaml \
  --output artifacts/mental_arithmetic_search_smoke \
  --quasi-trials 4 --bayes-trials 1 --epochs 2 --patience 2 \
  --device cuda --num-workers 4
```

正式搜索建议：

```bash
PYTHONPATH=src python scripts/search_mental_arithmetic_pretrain.py \
  --config configs/pretrain_aamp_mental_arithmetic.yaml \
  --output artifacts/mental_arithmetic_frequency_search_v1 \
  --quasi-trials 12 --bayes-trials 12 --epochs 25 --patience 6 \
  --device cuda --num-workers 4
```

搜索完成后按最优参数正式预训练：

```bash
PYTHONPATH=src python scripts/run_best_mental_arithmetic_pretrain.py \
  --config configs/pretrain_aamp_mental_arithmetic.yaml \
  --search-dir artifacts/mental_arithmetic_frequency_search_v1 \
  --output artifacts/pretrain_mental_arithmetic_frequency_best \
  --epochs 100 --device cuda --num-workers 4
```

查看搜索最优结果：

```bash
python -m json.tool artifacts/mental_arithmetic_frequency_search_v1/search_summary.json
python -m json.tool artifacts/mental_arithmetic_frequency_search_v1/best_pretrain_params.json
```

查看每次试验排序和状态：

```bash
python -c 'import json; p="artifacts/mental_arithmetic_frequency_search_v1/search_results.json"; d=json.load(open(p)); rows=[r for r in d["trials"] if r["status"]=="success"]; rows.sort(key=lambda r:r["metrics"]["objective"]); [print(r["trial_idx"], r["metrics"]["objective"], r["params"]) for r in rows]'
```

查看特定 trial 日志和曲线数据：

```bash
less artifacts/mental_arithmetic_frequency_search_v1/trials/trial_000/run.log
python -m json.tool artifacts/mental_arithmetic_frequency_search_v1/trials/trial_000/pretrain_history.json
```

检查最终权重和元数据：

```bash
ls -lh artifacts/pretrain_mental_arithmetic_frequency_best/final_pretrain_model.pt
python -m json.tool artifacts/pretrain_mental_arithmetic_frequency_best/pretrain_metadata.json
python -m json.tool artifacts/pretrain_mental_arithmetic_frequency_best/pretrain_history.json
```

## 解释结果时的限制

搜索目标是遮挡 alpha 频带重建，不是专注分类准确率。它能筛出适合当前自监督目标的结构与优化超参数；分类收益必须由后续三路线的独立、被试隔离二分类评估确认。当前输出头用时间 token 聚合后回归整个频谱，因此搜索出的 temporal pooling 仍需结合下游 encoder 表现复核。
