# Ear-SAAD AAMP 预训练协议

## 目的

AAMP 预训练用于学习 EEG 重建特征，不直接证明下游注意解码有效。
预训练 checkpoint 只有在后续固定三折、三 seed 的迁移实验中优于随机初始化
对照时，才可以作为有效的下游初始化。

## 主协议

主协议使用 `train` 和 `val` 被试的无标签波形：

- `test` 被试完全不读取、不参与预训练；
- 预训练数据不提供标签给训练循环；
- 归一化统计量只使用原始 `train` 被试；
- `val` 被试虽参与无标签预训练，但只记录重建诊断，不用于早停或 checkpoint 选择；
- 使用固定 epoch，`final_pretrain_model.pt` 是主协议权重；
- `best_pretrain_model.pt` 仅表示验证重建损失最低点，不能视为独立验证选择结果。

启动命令中的 `--unlabeled-splits train_val` 即为主协议。

## 超参数搜索协议

在正式预训练前，先运行 `scripts/search_aamp_lr_bs.py`。搜索空间只包含学习率和
批大小，weight decay、模型结构、掩码策略和随机种子保持固定，避免把多个变化
混在一起：

- 第一阶段使用 8 个 Sobol 准随机点，学习率在 `[1e-5, 5e-4]` 的 log 空间采样，批大小从 `[16, 32, 48, 64]` 选择；
- 第二阶段使用前一阶段全部结果拟合 Matern 高斯过程，使用 EI 采集函数追加 8 个点；
- 每个点最多训练 25 epoch，validation 连续 5 epoch 无改善即早停；
- 目标为 `best_val_loss + 0.25 * max(0, val_loss - train_loss)`，越低越好；
- 搜索阶段使用 `train_only`，validation 不参与参数更新，test 完全不读取；
- `search_summary.json` 和 `recommended_params.json` 保存搜索轨迹及推荐配置。

搜索目标仅用于改善自监督重建的泛化，不能单独定义教师合格。推荐配置正式复训
后，还必须通过下游分类迁移验证；教师仍以三折、seed 42/43/44 平均
window-level balanced accuracy 不低于 0.70 且每折无类别坍缩为硬门槛。

搜索命令：

```bash
PYTHONPATH=src python scripts/search_aamp_lr_bs.py \
  --config configs/pretrain_aamp_ear_saad.yaml \
  --output artifacts/aamp_lr_bs_search_v2 \
  --quasi-trials 8 \
  --bayes-trials 8 \
  --epochs 25 \
  --patience 5 \
  --device cuda
```

## 当前搜索结论

`artifacts/aamp_lr_bs_search_v2` 已完成完整搜索，共 16 个 trial：

- 8 个 Sobol 准随机 trial；
- 8 个 Matern GP + EI 贝叶斯优化 trial；
- 搜索阶段使用 `train_only`，test 未暴露；
- 当前最佳组合为 `learning_rate=0.00044348092176995735`、`batch_size=16`；
- 最佳 trial 为 `artifacts/aamp_lr_bs_search_v2/trials/trial_015`；
- 最佳 epoch 为 24/25；
- `best_val_loss=0.46251571589886253`；
- `train_loss_at_best_val=0.5047316401455533`；
- `generalization_gap=-0.04221592424669075`；
- `objective=0.46251571589886253`。

该结果只说明当前搜索空间内的自监督重建目标最优，不能直接证明下游分类有效。
推荐参数必须进入主协议固定 epoch 复训，再进行迁移验证。

## 推荐配置正式复训

使用当前最佳学习率和批大小，切换到主协议 `train_val`，固定 100 epoch：

```bash
PYTHONPATH=src python scripts/pretrain_aamp.py \
  --config configs/pretrain_aamp_ear_saad.yaml \
  --output artifacts/pretrain_aamp_ear_saad_search_v2_lr4p43e-4_bs16 \
  --epochs 100 \
  --batch-size 16 \
  --lr 0.00044348092176995735 \
  --unlabeled-splits train_val \
  --checkpoint-selection fixed_epochs \
  --device cuda
```

本轮训练完成后，以
`artifacts/pretrain_aamp_ear_saad_search_v2_lr4p43e-4_bs16/final_pretrain_model.pt`
作为主协议预训练权重。`best_pretrain_model.pt` 仍只作为重建诊断点，不作为独立
验证选择结果。

复训完成后检查：

- `pretrain_metadata.json` 中 `test_exposed_to_pretraining=false`；
- `unlabeled_splits=train_val`；
- `checkpoint_selection=fixed_epochs`；
- `pretrain_history.json` 中 train/val loss 是否稳定下降；
- `final_pretrain_model.pt` 是否生成并记录 SHA-256。

## Transductive 对照

`--unlabeled-splits all` 会把 test 波形也加入无标签预训练。这是
transductive/self-supervised adaptation，不再是完全盲测协议。必须单独保存、
单独迁移评估，不能和主协议 checkpoint 混用，也不能用其 test 结果声称跨被试
独立泛化。

## 输出审计文件

每次运行输出：

- `pretrain_config.yaml`
- `pretrain_metadata.json`
- `pretrain_history.json`
- `best_pretrain_model.pt`
- `final_pretrain_model.pt`

元数据记录数据协议、subject 列表、归一化、AAMP 参数、模型参数量、固定验证
mask seed 和 checkpoint SHA-256。

## 后续有效性验证

完成预训练后，必须在不读取 test 标签的前提下进行：

1. frozen encoder + 分类头；
2. full fine-tuning；
3. 与同架构 random-init 对照；
4. 固定三折、seed 42/43/44；
5. 报告 balanced accuracy、macro-F1、ROC-AUC、subject-level 指标和类别坍缩。

重建 loss 下降本身不是教师保留或蒸馏的依据。
