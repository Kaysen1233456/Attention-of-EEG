# 改动日志（CHANGELOG）

> **append-only**，人写。只记录**改变项目行为 / 结论 / 协议**的改动。
> 每条必须引用对应的运行日志目录（见 `README.md` §4）。

---

## 2026-10-09

### 15:52 · RV-1 执行完成：折安全校准模块 + 四个探针重跑 + 登记册新增 C12

- **类型**：**代码修复（结论级）+ 探针重跑 + 登记册更新**
- **涉及文件**：`scripts/probes/_subject_calibration.py`（新）、
  `scripts/probes/{domain_probe,cnn_domain_probe,teacherB_probe,matrix_probe}.py`、
  `artifacts/probes/*.json`、`artifacts/probes/_superseded/README.md`、
  `experiments_log/INVALID_METHODS_REGISTRY.md`
- **运行日志**：`runs/20261009-154628_probe_rv1_domain_adaptation_v2`、
  `runs/20261009-154822_probe_rv1_cnn_domain`、`runs/20261009-155017_probe_rv1_teacherB_domain`、
  `runs/20261009-155207_probe_rv1_matrix_2x2`
  （另有两次中间运行：`154413` 被 v2 取代、`154705` 因 dtype 报错失败）

#### 修复内容

| 编号 | 改动 |
|---|---|
| **C10（修复）** | 新增 `scripts/probes/_subject_calibration.py`，把「选校准窗 → 只在校准窗拟合 μ/σ → 应用 → 返回校准窗掩码」封成一个强制折安全的接口。四个探针全部改为 `calibrate_by_subject()` + `matched_eval_indices()`，校准窗**从训练与评估集中剔除**。旧 `subject_standardize()` 改为**调用即抛 RuntimeError**，防止回归 |
| **C12（新增）** | 发现 `fit_subject_stats` 的 `block` 参数是**无效参数** —— `sub[:, b].mean(axis=0)` 本就逐列求均值，分块只影响赋值，对结果毫无影响；且分块是**连续切片**而特征列是**频带优先**布局，连分组都错。结果「被试+通道级标准化」**从来等于「被试级标准化」**（原报 N1 = N2 = 0.6976 即为铁证）。已改为 `groups=channel_groups(40, 8)` 真池化 |

#### 重测结果（协议 v1.3 方案 A：静息段前 30 窗，折安全）

| 探针 | 条件 | 旧（作废） | **新** |
|---|---|---|---|
| 2 线性 | N0 / N1 / N2 | 0.6255 / 0.6976 / 0.6976 | **0.6348 / 0.6967 / 0.6733** |
| 3 Gen5 | S0 / S1 / S2 | 0.6905 / 0.7440 / 0.7344 | **0.6956 / 0.7263 / 0.7175** |
| 4 TeacherB | T0 / T1 / T2 | 0.7071 / 0.6613 / 0.7308 | **0.7130 / 0.6899 / 0.7180** |
| 5 2×2 | Gen5 ±std / TB ±std | 0.6943/0.7359 / 0.7071/0.7405 | **0.7232/0.7275 / 0.7130/0.7314** |

#### 结论级变化（三条）

1. **校准收益被高估 2~10 倍**：原报 +3.3~+7.2 分 → 实测 **+0.4~+3.1 分**。
   2×2 口径下 Gen5 仅 **+0.4 分**，落在折间波动（±0.05）内。
2. **教师优势消失**：同一评估集 Teacher B **0.7130** vs Gen5 **0.7232**，**学生反高 1.0 分**
   （参数 23 倍）。结合 `teacher_b_seed_confirmation` 的「教师真实约 0.67」，
   **「做强教师再蒸馏」的前提不成立**。
3. **N2 曾为静默重复行**，修正后 N2 = 0.6733 ≠ N1 = 0.6967。

#### 对后续计划的影响

- ❌ 不再默认「做强教师 → 蒸馏」；改为**先把 Gen5 学生本身做上去**。
- ⚠️ D1（被试级标准化）收益仍为正但已缩小，**引用前必须先做多 seed 重复**。
- ✅ `_superseded/` 保留全部作废旧值（未删除），对照表见 `artifacts/probes/_superseded/README.md`。

---

### 15:35 · 协议收尾（RV-2/3/4/6/7）+ 编号勘误 + 登记册新增 C10/C11

- **类型**：协议修订 + 文档纠错 + 新工具
- **涉及文件**：`CALIBRATION_PROTOCOL.md`、`experiments_log/INVALID_METHODS_REGISTRY.md`、
  `experiments_log/README.md`、`experiments_log/runs/20261009-111900_teacher_b_data_audit/README.md`、
  **新增** `scripts/restamp_run_logs.py`、**新增** `PROJECT_ISSUES_PLAIN.md`

#### ⚠️ 编号勘误（重要）

审查清单原先使用的 `B1`–`B8` **与登记册 B 组（配置层事故）的 `B1`–`B5` 冲突**。
例：「B4」曾同时指「方案D seed42/seed44 训练配置问题」和「Subject31 静息窗不足」。
**清单已全部改名为 `RV-1`–`RV-8`**（RV = review）。登记册 B 组含义不变。
已在 `PROJECT_ISSUES_PLAIN.md` 与登记册顶部加编号空间说明。

#### 协议修订：`CALIBRATION_PROTOCOL.md` v1.1 → **v1.2**

| 编号 | 改动 |
|---|---|
| **RV-2** | §1.2 定死校准作用阶段为「**窗口级 z-score（含 clip ±8）之后**」，并给出 6 步完整定义。原措辞「原始窗口上的 40 维向量」有歧义：实测相对频带功率对窗口 z-score **不是不变的**（最大差 4.52、均值 0.154，log 域），两种实现会算出不同统计量 |
| **RV-3** | 新增 §1.3：**禁止白化**（30 窗估计 40 维协方差，`matrix_rank=29 < 40`、条件数 `3.8e18`，数学上不可行）；只允许对角标准化；最小样本数 30 |
| **RV-4** | 新增 **R5**：校准后剩余静息窗 < 15 → 标记 `rest_after_cal_low`，结果表必须**含/不含该被试各报一行**。当前唯一触发者 **Subject31**（40 窗 − 30 = 10） |
| **R1** | §1.1 加「⛔ 已知阻塞」区块，记录 R1 在 PhysioNet eegmat 上**不可满足**的实测证据与三个候选方向（**待 owner 拍板**） |
| §3 | 第 6 条补上 `rest_after_cal_low` 被试 |

#### 文档纠错

- **RV-6**：`runs/20261009-111900_teacher_b_data_audit/README.md` 原写「fold5（被试 1,10,19,26）」——
  `1/10/19/26` 是 `train_subjects + val_subjects` 的**位置索引**，真实为
  **Subject13 / Subject09 / Subject01 / Subject27**。已加勘误表。
  同时在 `experiments_log/README.md` 新增 **§6 被试命名约定**（含完整位置→名字对照表）。
- 同文件补注「32 × 2 = 64」为 train 28 + val 4，**不含 test 4 人**（避免与 metadata 的 36 人混淆）。

#### 新工具：`scripts/restamp_run_logs.py`（RV-7）

回填日志时 git 提交号会变，直接覆盖 `meta.json` 会丢失原始记录。
本工具把每次提交号**追加**进 `sha_history`（**只增不减**），顶层 `git` 更新为当前状态。
已验证**幂等**（重复运行不产生重复条目）且保留原有字段。用法见 `experiments_log/README.md` §7。

#### 登记册新增与降级

- **新增 C10**：在 `GroupKFold` 划分**之前**做被试级标准化 → 测试折信息泄漏 → 校准收益系统性高估。🔴 禁用
- **新增 C11**：用 30 个校准窗口做全协方差白化 → 数学上不可行。🔴 禁用
- **降级 D1**：「频域分支被试级标准化」原列为已验证正面做法（Gen5 +4.2 / Teacher B +3.3），
  但**证据来自含 C10 泄漏的探针** → 标记「**证据已作废，待重测**」。

#### 仍未完成

- **RV-1**：探针代码修复（把统计量拟合挪进折内）+ 旧结果作废留痕 —— 待 owner 确认
- **RV-5**：协议阻断规则的解封条件 —— 取决于 R1 定案
- **R1**：待 owner 拍板 A / B / C

---

### 15:07 · 频谱频带审计：README 滤波说法被证伪 + gamma/EMG 风险假设

- **类型**：审计（只读）
- **日志**：`runs/20261009-150732_spectral_band_audit/`
- **新增脚本**：`scripts/audit_spectral_bands.py`

**起因**：核对官方 README 与真实信号，检查 40 维频谱特征是否存在结构性缺陷。

**两个事实性结论**：

1. ✅ **官方 README 的「high-pass 30 Hz」说法是错的。**
   实测 EDF 头 prefiltering 字段（权威）：`HP:0.5Hz LP:45Hz RF:50Hz`。
   → `preprocess_mental_arithmetic.py` 的 1–45 Hz 带通**正确**，特征构造无系统性错误。
   （若 README 说法属实，40 维里 32 维将是垃圾 —— 属灾难级问题，已排除。）
2. ✅ **无死维。** gamma(30–45Hz) 仅占 1–4% 功率，但 log(相对功率) 使其 8 个维度
   std 达到 `0.47–0.77`，与其他频带同量级。近似常数维 = 0/40。

**新增 3 条待验证假设（n=2，非结论）**：

| # | 假设 | 依据 |
|---|---|---|
| H1 | theta 信息量最低 | 效应量两被试均 ≈0.26/0.32（其他频带 0.47–0.69） |
| H2 | beta/gamma 跨被试不稳定 | 效应量跨被试极差 0.35 / 0.17 |
| **H3** | **gamma 判别力可能为 EMG 污染** | **gamma 仅占 1–4% 功率却是单频带最强判别器（BA 0.756/0.761），且判别力集中于 Pz/P3/P4/T4 而非额叶** |

**H3 为最高优先级** —— 若成立会导致「测试集虚高、部署失效」。
初步观察 45–100 Hz 能量比 1–30 Hz 低 2–5 个数量级，**暂不支持强 EMG**，需全量验证。
**验证完成前不得声称 gamma 捕捉到了认知负荷。**

**过程教训（已记入运行日志）**：第一版脚本用 `accuracy` 评估频带可分性，
得出「delta 0.77 / alpha 0.71」等看似可用的数字，实际**几乎完全由多数类（91/31 → 0.746）决定**，
属无效结果。已改为 BA 重跑。**再次印证本项目坚持用 BA 的必要性。**

**未改变计划**：3 条假设均需全量 36 被试嵌套 CV 验证，且需先解决 R1 口径。

---

### 15:02 · 数据来源核查：确认数据集完整 + R1 不可满足

- **类型**：审计（只读，无项目写入）
- **日志**：`runs/20261009-150224_eegmat_provenance_audit/`

**触发**：owner 发现本地 `data/raw/eegmat/` 只有 4 个 EDF（2 被试），
怀疑「数据集不完整」导致「无法声称有独立 calibration session」。

**核查结论**：

| 问题 | 结论 |
|---|---|
| 本地原始数据是否残缺？ | ✅ **是** —— 本地 4/72 文件（2/36 被试），可重下修复 |
| 数据集本身是否缺一段独立记录？ | ❌ **不是缺，是设计上就没有** |
| 处理产物来源是否可信？ | ✅ **已验证为真**（见下对账） |

**决定性对账**（本地独立预测 vs 云端独立实测，完全一致）：

```
25 × 91 + (40 + 85 + 94) = 2494  = metadata rest          ✅
28 × 31                  =  868  = metadata focus         ✅
2494 + 868               = 3362  = metadata train_samples ✅
4 × 91 / 4 × 31          = 364 / 124 = val & test         ✅
```

→ **`data/processed/mental_arithmetic_250hz_8ch` 是真实 36 人数据，非伪造。**

**数据集身份**：PhysioNet `eegmat` v1.0.0（36 人 × 2 段 = 72 文件，500 Hz，19 EEG 通道）。
**无 `_3.edf`，无 session 子目录** → 不存在第三段独立记录。

**影响**：

1. ❌ `CALIBRATION_PROTOCOL.md` **R1（独立无标签 calibration recording）在本数据集上不可满足**。
   **这不是下载不全，无法通过重新下载或改变窗口切法解决。**
2. ❌ 此前探针报的 calibration **+3.3~+7.2 分收益双重不可信**：
   (a) 折前拟合泄漏（`subject_standardize` 在 `GroupKFold` **之前**调用，已登记为 **C10**）；
   (b) 校准段本就非独立 session。
3. ✅ 训练 / 评估数据有效性**不受影响**，既有结论可继续引用。
4. ⚠️ `data/` 与 `*.npy` 均在 `.gitignore` 中 → **数据无法从 GitHub 恢复**。
5. 📌 次要发现：数组 dtype 为 `float64`（非 float32），内存占用翻倍。

**待 owner 拍板**：R1 三个出路 —— A 同段留出前缀+强制披露 / B 放弃校准只报 zero-shot / C 新采数据。

**未启动**：校准训练、蒸馏训练（数据来源层面已澄清，但 R1 定案前不启动）。

---

### 14:45 · 修复 RV-8（原记作 B8）：`test_audit_fold2.py` 断言路径错误

- **类型**：测试修复（1 行）
- **文件**：`tests/test_audit_fold2.py:19`

```diff
-    assert report["trial_label_conflicts"] == 1
+    assert report["trial_report"]["trial_label_conflicts"] == 1
```

**结果**：`2 failed, 71 passed`（共 73 项），此前 `3 failed, 70 passed`。
剩余 2 个失败均为 `tests/test_aamp_masking.py`（已放弃的 Ear-SAAD/AAMP 路线，文件未改动）。
实现 `scripts/audit_fold2_subjects.py` **未改动** —— 改测试不改实现，避免动返回结构。

**注**：计划中曾误写「73 passed / 2 failed」；73 是测试**总数**，正确预期是 **71 passed / 2 failed**。

---

### 13:47 · Gen5 协议冻结 batch_size=256

- **文件**：`GEN5_PROTOCOL.md`、`scripts/final_spectral_cnn.py`
- `batch_size = 256` 显式写入最终训练协议（来源：准随机 + 贝叶斯搜索的最优点）
- `bs128` / `bs512` 标记为「未采用的扫描点」
- `final_spectral_cnn.py` 默认值 64 → **256**（与协议对齐，避免 BatchNorm/优化器行为漂移）
- 新增 `--allow-exposed-test` 守卫：Gen5 test holdout 已暴露，禁止用于模型选择

---

### 13:34 · 建立实验日志规范

- **新增**：`experiments_log/README.md`、`scripts/run_with_log.py`
- 硬规则：任何改变模型行为 / 数据划分 / 评估口径 / 协议的操作都必须留下可回溯日志
- 回填 4 个历史运行日志到 `experiments_log/runs/`
- 注：`logs/` 与 `*.log` 已被 gitignore，`experiments_log/` 是唯一可提交的回溯入口
