# 有害配置记录（避免再踩坑）

## 1. encoder_freeze_epochs 默认值=0 的bug
- **配置**：不设置 encoder_freeze_epochs（默认0），同时设置 freeze_pretrained_backbone=true
- **危害**：trainer初始化时调用 _set_encoder_trainable(0==0)=True，会把Encoder解冻，覆盖模型构建时的冻结设置
- **解决方案**：阶段1必须设置 encoder_freeze_epochs=999（大于epochs数，确保全程冻结）
- **发现时间**：2026-09-22

## 2. 阶段1 lr=5e-4 + encoder_freeze_epochs=0（默认）
- **配置**：lr=0.0005, encoder_freeze_epochs=0（默认，Encoder被意外解冻）
- **结果**：BalAcc mean=60.0%, std=7.4%，seed44完全类别坍缩（BalAcc=0.5），训练极不稳定
- **原因**：学习率偏大 + Encoder被意外解冻，导致训练剧烈震荡
- **结论**：阶段1学习率不应超过2e-4，且必须设置encoder_freeze_epochs=999

## 3. class_weights=[0.674, 2.5]（专注态权重过高）
- **配置**：lr=2e-4, class_weights=[0.674, 2.5], epochs=10
- **结果**：BalAcc mean=60.69%（-0.86%），Macro F1 mean=51.59%（**-8.85%暴跌**）
- **混淆矩阵**：静息召回40-46%，专注召回75-81%（从偏向静息变成偏向专注，矫枉过正）
- **原因**：专注态权重2.5过大，模型过度倾向预测专注类
- **结论**：class_weights最优值在1.937和2.5之间，可能在2.0-2.2左右。2.5绝对不要用。

## 4. 方案D seed42 和 seed44（epoch1就收敛，类别不均衡）
- **配置**：lr=1e-4, epochs=30
- **seed42**：最佳epoch=1，BalAcc=59.6%，静息召回69%，专注召回50%
- **seed44**：最佳epoch=1，BalAcc=64.3%，但静息召回94%，专注召回35%（严重偏向静息）
- **原因**：分类头很快收敛，预训练特征区分度有限，更多训练也没用
- **结论**：这两个seed的模型不是好模型，不要作为阶段2的起点。只有seed43是好的。

## 5. YAML中 weight_decay: 1e-5 被解析为字符串
- **配置**：weight_decay: 1e-5
- **危害**：PyYAML将1e-5解析为字符串，导致AdamW初始化时报错 TypeError: '<=' not supported between instances of 'float' and 'str'
- **解决方案**：写成 weight_decay: 0.00001（小数形式）
- **发现时间**：2026-09-22
