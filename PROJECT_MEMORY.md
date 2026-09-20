# Attention-of-EEG 项目记忆

更新时间：2026-09-20

## 当前总目标

Ear-SAAD 跨被试注意力解码：先验证 Transformer 教师，再将合格教师蒸馏到轻量
CNN 学生。最终部署对象是 CNN 学生，不是 Transformer 教师。

## 用户确认的硬门槛

教师必须在固定三折、seed 42/43/44 上达到平均 window-level validation
balanced accuracy `>= 0.70`，并且每个 fold/seed 不发生类别坍缩。还必须查看
macro-F1、ROC-AUC、subject-level 结果和 train-validation gap。未通过不得蒸馏。

## 既定七步路线

1. AAMP 自监督预训练完成并确认开发协议完整。
2. 使用正式 AAMP checkpoint 初始化 Transformer 分类教师。
3. 三折 × seed 42/43/44 训练分类教师。
4. 汇总 BA、macro-F1、ROC-AUC、类别稳定性和 subject-level 指标。
5. 平均 BA ≥ 0.70 且无类别坍缩才保留教师。
6. 教师通过后训练 CNN 蒸馏学生。
7. 未通过时记录为“重建有效但分类无效”，回到 AAMP/迁移策略诊断。

完整细节见 `TEACHER_AAMP_DISTILLATION_ROADMAP.md`。

## 今天的工程改动

- `scripts/pretrain_aamp.py`：增加 `train_only` 协议、显式 device 检查、可配置
  workers、固定 mask 验证、metadata/history/checkpoint hash 输出和验证早停。
- `scripts/search_aamp_lr_bs.py`：实现 Sobol 准随机后接 Matern-GP/EI 贝叶斯搜索，
  搜索学习率和 batch size，并记录 gap 惩罚目标。
- `configs/pretrain_aamp_ear_saad.yaml`：Ear-SAAD AAMP 配置。
- `configs/aamp_search_ear_saad.yaml`：搜索协议配置说明。
- `src/attention_model/config.py`、`scripts/train.py`：补充数据零通道策略和
  temporal pool 等配置兼容。
- `src/attention_model/data/aamp_masking.py`：向量化 masking，减少 CPU/NumPy
  循环开销。
- `AAMP_PRETRAIN_PROTOCOL.md`：记录 AAMP 数据隔离、搜索与迁移约束。
- `STAGE1_FIXED_PROTOCOL.md`、`scripts/summarize_stage1.py`：固定三折、多 seed
  汇总及教师决策支持。
- `.gitignore`：排除原始数据、下载目录、训练 artifacts 和 checkpoint。

## 已知实验事实

- 先前 random-init B0/B1 严格验证：B0 平均 BA 约 `0.5219`，B1 平均 BA 约
  `0.5113`，B1 delta 约 `-0.0105`，并且 B1 有类别坍缩，因此不能直接保留
  random-init Transformer 教师。
- 早期 AAMP 单次预训练：约 1.65M 参数，validation reconstruction loss 从
  `0.5717` 改善到约 `0.4718`。这不是分类指标。
- 当前 AAMP 搜索目录以 `artifacts/aamp_lr_bs_search_v2` 为准。搜索协议是
  8 Sobol + 8 Bayesian，每个 trial 最多 25 epoch、patience 5。搜索运行时
  不应把中间结果视为最终推荐参数。
- AAMP 模型：19 通道、20 Hz、5 秒窗口、100 samples、d_model=96、4 层、8 heads、
  d_ff=384、temporal_pool=4，约 1.65M 参数。

## 数据与可复现约束

- `data/processed/ear_saad` 是本地数据，不提交到 Git。
- `ds007169-download` 是 N-back 原始下载目录，不提交到 Git。
- `artifacts/`、`.pt`、`.npy`、`.npz` 等训练产物不提交到 Git。
- test subjects 在方案和超参固定前保持完全隔离。
- 所有新实验输出必须包含配置、seed、fold、checkpoint 和指标汇总。

## 下一次工作入口

1. 等 AAMP 搜索完整结束，检查 `search_summary.json` 是否包含 8+8 个完整 trial。
2. 依据推荐参数进行正式 train+val AAMP 复训，保存 checkpoint SHA。
3. 配置 AAMP 初始化的 Transformer 分类教师，先做 dry-run/load 验证。
4. 手动启动三折 × 三 seed 分类训练，不自动读取 test。
5. 汇总后按 BA ≥ 0.70 和类别稳定性决定是否蒸馏。
