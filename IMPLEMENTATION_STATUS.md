# 第一批实施状态

已完成实施方案 IMPLEMENTATION_PLAN.md，保留原有未提交改动，不修改数据或旧训练产物。

## 新增和修改

- config.py：temporal_frontend（默认none）、temporal_kernel（默认5）。
- mini_neuript.py：可选逐电极共享的深度可分离时间卷积残差前端，位于embedding之后、平均池化之前；默认Identity不新增state_dict参数。
- scripts/train.py：向教师构造器传递前端配置。
- scripts/diagnose_learnability.py：仅加载train数组，按类别固定抽样；禁用dropout/AMP/预训练，记录同checkpoint小样本eval、梯度、学习率及概率分散度。输出目录必须不存在，避免覆盖。该结果不是泛化分数。
- tests/test_temporal_frontend.py：默认路径strict加载、前端形状/梯度、非法参数测试。

## 验证与限制

初始实现阶段仅完成AST语法检查；本轮已在当前Python/PyTorch环境运行完整单元测试和CPU冒烟。A10上的真实训练仍需在远端环境执行。  
前端作用在含位置编码的embedding上，不是原始波形卷积；该选择是一个独立假设，不能声称已经提升性能。  
训练/搜索precision统一和主要指标异常处理已补齐；简单特征基线、三折自动实验仍待实施。测试集没有被本轮读取。

## 在项目训练环境执行

从D:\Attention目录运行（PYTHONPATH设为src）：

```text
python -m pytest tests/test_temporal_frontend.py -q
python scripts/diagnose_learnability.py --output D:/Attention/artifacts/learnability_v1 --device cuda
```

卷积对照的独立配置覆盖项：model.temporal_frontend=conv，model.temporal_kernel=5，model.temporal_pool=2。先与none/pool2比较；不得直接作为正式最佳配置。

## 本轮执行记录（2026-09-18）

- 完成开发数据审计：12 个 train/val 被试、无重复窗口、每个 subject/trial 标签一致。
- 固定三折 8/4 被试清单已写入 `configs/development_folds.json`。
- 多 seed 入口已改为每个 seed 独立子进程，重新构造模型、DataLoader、优化器、调度器和 AMP scaler。
- 训练运行保存 `run_metadata.json`，并在同一最佳 checkpoint 上额外计算训练集指标。
- 非有限 ROC-AUC 和评估指标现在直接抛错，不再静默写入 0。
- `EEGDataset(include_test=False).get_info()` 已修复，可用于纯开发折。
- 验证结果：`57 passed`，`compileall` 通过。

尚未完成：

- 在 A10 上运行修复后的 B1 三折 seed 42 对照；
- 在确认 seed 42 复现后再运行 seed 43/44；
- temporal_pool=2/1 和卷积前端的严格单因素比较；
- 教师达到门槛前不得启动蒸馏。
