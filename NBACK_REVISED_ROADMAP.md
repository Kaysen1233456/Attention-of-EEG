# ds007169 n-back 修订路线（v3）

更新日期：2026-09-20

## 结论先行

当前不应直接启动 AAMP 预训练，也不应把现有 `data/processed/nback` 当成最终训练集。
最优先的问题是数据契约和实验定义，而不是把 Transformer 做得更大。

已确认的事实：

- `ds007169` 的正式任务是 18 个被试、19 个 EEG 通道、250 Hz、每个难度 100 个主任务 trial。
- 每个被试另外有 81 个 tutorial trial；现有预处理把 tutorial 与正式任务一起纳入了 8665 个候选事件。
- 现有脚本按事件起点截取 2 秒，但 trial 间隔约 1.7 秒，因此默认窗口会覆盖下一 trial 的前约 0.3 秒。
- 原始数据包含 `dropped_samples` 标记；现有预处理没有按丢样本事件剔除受影响 epoch。
- 现有 n-back 脚本没有记录 trial、tutorial、block、dropout 质量字段，训练结果无法追溯到单个事件。
- 当前实例是 CPU 环境：`torch 2.13.0+cpu`，`torch.cuda.is_available()` 为 `False`，没有 `nvidia-smi`。本轮不启动正式训练。

## 与原路线的关系

项目里存在两条不同路线，必须分开：

1. Ear-SAAD：19 个耳周通道、二分类听觉注意解码，当前主线是监督式教师到 CNN 学生。
2. ds007169：19 个头皮通道、四分类 n-back 认知负荷，适合作为认知负荷任务的独立教师/预训练来源。

不能把 Ear-SAAD 的 `B0-B5` 指标当作 ds007169 指标，也不能把 ds007169 的四分类模型直接解释成耳周听觉注意模型。ds007169 的价值是提供跨被试认知负荷表征和教师知识，最终迁移仍需在 Ear-SAAD 上单独验证。

## 第一阶段：数据重建

### 1. 原始事件筛选

- 仅保留 `marker_stream == n-backMarkers`。
- 仅保留 `trial_type in {1-back, 2-back, 3-back, 4-back}`。
- 仅保留 `istutorial == False` 的正式任务 trial。
- 标签使用 `nback_level`，映射为 `0..3`。
- 排除 epoch 与 `dropped_samples` 区间重叠的 trial，并记录剔除原因。

### 2. 时间窗口

第一版统一使用 `[event_onset, event_onset + 1.7 s)`，即 425 个采样点。
原因是该任务每个刺激 trial 的设计间隔约为 1.7 秒；直接使用 2 秒会泄漏下一 trial。

后续再做一个严格对照：`0.0-1.0 s` 刺激窗口与 `0.0-1.7 s` 完整 trial 窗口。窗口长度不是调参项，必须在数据协议中固定后再比较。

### 3. 信号处理

- 保持原始 250 Hz，不做无必要的重采样。
- 参考现有 1-45 Hz 过滤，但增加 50 Hz notch；具体频带通过验证集确定，不在测试集上挑选。
- 首版使用每通道训练被试统计量 z-score，统计量只从训练被试计算。
- 保留每个 epoch 的 subject、trial ordinal、n-back level、tutorial flag、dropout overlap、block index 和原始 onset。
- 训练前生成 `manifest.jsonl`，而不是只保存无法回溯的三个 `.npy` 数组。

### 4. 被试划分

- 固定 subject-disjoint train/validation/test；不要随机打散窗口。
- 主任务四个 level 每个被试数量一致，因此优先固定 12/3/3 被试划分，并额外做 3-fold development CV。
- 测试被试在架构、预处理、窗口长度、阈值和超参数选择中完全不可见。

## 第二阶段：可学习性与数据质量门

在 GPU 训练前必须通过以下门槛：

1. 审计报告显示每个被试正式任务为 400 个事件，tutorial 不进入训练集。
2. 四类数量接近一致，subject 划分无交集。
3. `dropped_samples` 重叠 epoch 已剔除并有计数。
4. 小样本记忆测试达到训练准确率接近 100%，证明标签、形状、loss 和优化器链路正确。
5. 频带功率、RMS、峰峰值等简单特征 baseline 必须报告 train/validation/test，作为神经网络上限和数据泄漏探针。
6. 训练集与验证集的每通道均值、标准差、频带功率分布必须输出；若跨被试偏移远大于 level 差异，先做稳健归一化与参考方案比较。

## 第三阶段：监督教师，而非先做大规模 AAMP

推荐顺序：

### T0：统计特征和线性基线

每个 epoch 计算 delta/theta/alpha/beta/gamma 相对功率、谱熵、RMS、Hjorth 参数和左右/前后区域差异。使用 logistic regression、LDA、XGBoost 或浅层 MLP。

目的不是追求最终分数，而是回答：n-back level 是否在跨被试条件下存在稳定信号。

### T1：小型时频 CNN

输入为原始 epoch 与 log-PSD 的双分支，模型保持 50K-300K 参数，使用 GroupNorm/LayerNorm，避免在 12 个训练被试上依赖 BatchNorm 的被试统计量。

### T2：轻量 Transformer 教师

仅当 T0/T1 通过数据质量门后使用：

- 4-8 个 temporal patch/token，而不是把每个通道的短窗口过度切碎。
- `d_model=64/96`、2-4 层、4-8 heads。
- 通道位置编码只作为候选，不假设近似 3D 坐标一定有帮助。
- 先使用 mean pooling；IILP、双边融合、PMoE、SwiGLU 逐项消融。
- 选择指标使用 macro-F1、balanced accuracy、one-vs-rest ROC-AUC 和跨折标准差。

## 第四阶段：自监督预训练的正确定位

AAMP/MAE 不再是默认起点。只有当监督 T0/T1 证明数据有可学习跨被试信号，且小模型已经正确工作时，才做受控自监督对照：

1. `masked reconstruction` 只用正式任务的无标签 EEG，tutorial 不能悄悄混入。
2. 预训练不能用验证/测试标签；建议从训练被试开始，另做全数据无标签预训练作为明确标注的 transductive 对照，不混为同一结果。
3. 必须与 zero-output、copy-input、band-power encoder baseline 比较。
4. 预训练是否有效只能由下游跨被试验证提升决定，不能由 reconstruction loss 单独决定。
5. 首轮固定小模型、短预算，避免在数据问题未解决前花 GPU 时间搜索 mask ratio 和大模型。

## 第五阶段：迁移到 Ear-SAAD

ds007169 与 Ear-SAAD 的任务、采样率、通道布局和标签语义不同，不能直接复制分类头。可行迁移路径：

- 共享 temporal encoder 的初始化，只迁移相同采样率/patch 定义下的时间建模层。
- 通道 embedding 和左右耳分支必须重新初始化或通过显式通道映射适配。
- 先 linear probe，再全量 fine-tune。
- 最终仍以 Ear-SAAD 固定被试折的 balanced accuracy、macro-F1、ROC-AUC 和 seed 均值/标准差为准。
- 教师没有稳定超过 Ear-SAAD CNN 基线前，不启动蒸馏。

## 训练前 GPU 清单

在远端 GPU 环境执行前先确认：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
nvidia-smi
python scripts/audit_nback_raw.py
python -m pytest -q
python -m compileall -q src scripts
```

审计和数据重建通过后，第一轮只跑 T0/T1 的单 seed smoke test，再跑固定 3-fold；不要直接启动 100 epoch AAMP。
