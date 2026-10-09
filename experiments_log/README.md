# 实验日志规范（Experiment Logging Convention）

> **硬规则**：任何改变模型行为、数据划分、评估口径、协议或产物的操作，都必须留下可回溯的日志。
> 没有日志的改动视为未发生 —— 不得作为结论引用。

本目录（`experiments_log/`）**不被 `.gitignore` 忽略**，因此这里的内容会提交到 GitHub，
是唯一的「长期回溯入口」。

> ⚠️ 不要用 `logs/` 或 `*.log` 存日志 —— 它们已被 `.gitignore` 忽略，提交不上去，回溯时找不到。

---

## 1. 目录结构

```text
experiments_log/
├── README.md                     # 本文件：规范
├── CHANGELOG.md                  # 改动日志（append-only，人写）
├── harmful_configs.md            # 已知有害配置档案（原有）
├── INVALID_METHODS_REGISTRY.md   # 无效方法登记册
└── runs/                         # 每次运行的完整日志
    └── <YYYYMMDD-HHMMSS>_<name>/
        ├── meta.json             # 机器可读：命令 / git SHA / 环境 / 依赖版本 / 耗时 / 退出码
        ├── stdout.log            # 完整标准输出
        ├── stderr.log            # 完整标准错误
        ├── combined.log          # 合并时间线（人工回溯主要看这个）
        └── README.md             # 人读摘要 + 关键结论（运行后必须补写）
```

---

## 2. 怎么跑（唯一推荐方式）

**不要**直接 `python scripts/xxx.py > out.log`。**用包装器**：

```bash
python scripts/run_with_log.py --name <短名> --note "<备注>" -- \
    python scripts/<你的脚本>.py <参数...>
```

例：

```bash
# 三 seed 教师确认
python scripts/run_with_log.py --name teacher_b_seed_confirmation --note "门槛 0.72" -- \
    python scripts/train_teacher_b_seeds.py --output artifacts/teacher_b_seed_confirmation

# 嵌套阈值校准
python scripts/run_with_log.py --name teacher_b_threshold_calibration --note "固定0.5 vs 校准" -- \
    python scripts/calibrate_teacher_b_threshold.py --output artifacts/teacher_b_calibration

# Gen5 嵌套 CV（被试级校准对照）
python scripts/run_with_log.py --name gen5_nested_cv_calibrated -- \
    python scripts/cv_spectral_cnn.py --folds 8 --calibration subject
```

包装器会自动捕获：

- 完整 stdout / stderr / 合并时间线
- 命令原文与工作目录
- **git 提交 SHA、分支、是否有未提交改动**（回溯时能精确定位代码版本）
- Python 版本、平台、numpy / scipy / torch / sklearn / mne 版本
- 起止时间、耗时、退出码
- 自动脱敏 `ghp_*` / `github_pat_*` / `sk-*` 等 token

运行结束后**必须**打开生成的 `README.md`，在「关键结论」区块补写：

1. 跑出了什么（关键指标数值）
2. 结论是什么
3. 是否改变了后续计划

---

## 3. 日志命名规范

| 前缀 | 用途 |
|---|---|
| `train_` | 训练类运行 |
| `cv_` | 交叉验证 / 模型选择 |
| `eval_` | 测试集评估 |
| `probe_` | 探针 / 诊断实验（不改模型，只回答一个问题） |
| `audit_` | 数据 / 产物审计 |
| `search_` | 超参搜索 |

短名用下划线连接，全小写，能自解释。例：`teacher_b_seed_confirmation`。

---

## 4. 与 git 的关系

一次有意义的改动应该对应一个 commit，commit message 里**引用对应的运行日志目录名**：

```text
Freeze Gen5 batch_size=256 and register scan points

- GEN5_PROTOCOL.md: batch_size=256 written into the final training protocol
- scripts/final_spectral_cnn.py: default aligned to 256 (was 64)
- experiments_log/runs/20261009-134000_teacher_b_seed_confirmation/
- experiments_log/CHANGELOG.md

Refs: TEACHER_PROTOCOL.md acceptance gate
```

---

## 5. 已有运行日志索引

| 运行 | 日志目录（实际路径） | 结论 |
|---|---|---|
| Teacher B 三 seed 确认 | `runs/20261009-112100_teacher_b_seed_confirmation/` | ❌ 不合格（BA 0.6991 < 0.72） |
| Teacher B 阈值校准 | `runs/20261009-112900_teacher_b_threshold_calibration/` | ❌ 校准无收益（0.6681 vs 0.6693） |
| Teacher B 困难折数据审计 | `runs/20261009-111900_teacher_b_data_audit/` | ✅ 0 结构错误；fold5 是形态差异 |
| 四个探针合并运行（含混淆 / 被试级标准化 / 教师-学生 2×2 / CNN 域） | `runs/20261009-110000_probe_series/` | ⛔ **探针 2/3/4/5 的「被试级标准化」收益已作废**（C10 折前拟合泄漏）；✅ 探针 1 混淆探针不受影响 |
| **心算数据集来源核查** | `runs/20261009-150224_eegmat_provenance_audit/` | ✅ 上游完整（72 文件 / 36 人）；处理产物来源**已验证为真**；❌ **无第三段独立记录 → R1 不可满足** |
| **频谱频带审计** | `runs/20261009-150732_spectral_band_audit/` | ✅ 官方 README「30Hz 高通」说法被证伪（EDF 头为 0.5–45Hz）；无死维；⚠️ theta 信息量最低、beta/gamma 跨被试不稳定；⚠️ **gamma 判别力疑为 EMG 污染（n=2 假设）** |
| **RV-1 重跑 · 域适应（线性）** | `runs/20261009-154628_probe_rv1_domain_adaptation_v2/` | ✅ N0 **0.6348** / N1 **0.6967**(+6.2) / N2 **0.6733**(+3.9)；N2 修正后**不再等于 N1**（C12） |
| **RV-1 重跑 · Gen5 CNN 域** | `runs/20261009-154822_probe_rv1_cnn_domain/` | ✅ S0 **0.6956** / S1 **0.7263**(**+3.1**，旧报 +5.4) / S2 **0.7175**(−0.9，C2 结论不变) |
| **RV-1 重跑 · Teacher B 域** | `runs/20261009-155017_probe_rv1_teacherB_domain/` | ✅ T0 **0.7130** / T1 **0.6899** / T2 **0.7180**(**+2.8**，旧报 +7.0) |
| **RV-1 重跑 · 教师 vs 学生 2×2（决定性）** | `runs/20261009-155207_probe_rv1_matrix_2x2/` | ⚠️ **教师优势消失**：TeacherB **0.7130** vs Gen5 **0.7232**（**学生反高 1.0**，参数 23×）；校准增益 Gen5 仅 **+0.4**、TeacherB **+1.8** |
| ~~RV-1 重跑（中间版）~~ | `runs/20261009-154413_probe_rv1_domain_adaptation/`、`runs/20261009-154705_probe_rv1_cnn_domain/` | 🗑 已被取代（前者 N2 无效、后者 dtype 报错），保留供回溯 |

> 📌 **RV-1 重跑的口径**：校准 = 静息段前 30 窗、**只在校准窗拟合 μ/σ**、校准窗从训练与评估集剔除。
> 作废旧值与重测新值的完整对照表见 **`artifacts/probes/_superseded/README.md`**。

> ⚠️ **索引修正记录（2026-10-09）**：本表此前把三个探针写成独立的 `runs/20261009-*_probe_*/` 目录，
> 但实际它们全部合并在 `runs/20261009-110000_probe_series/` 一个目录里。
> 已按实际路径更正 —— 这正是本规范要防的「回溯时找不到」。

---

## 6. 被试命名约定（重要，防止回溯时认错人）

> **硬规则：日志和报告中提到被试，一律写 `SubjectNN` 全名（如 `Subject13`）。禁止只写数字。**

### 为什么

`metadata.json` 里被试的排列顺序是**打乱的**，不是 `Subject00, Subject01, Subject02, ...`。
因此「被试 13」这种说法是**歧义的** —— 它可能指 `Subject13`，也可能指列表里
第 13 个位置上的被试（= `Subject04`）。**两者完全不是一个人。**

| 说法 | 是否可用 | 含义 |
|---|---|---|
| `Subject13` | ✅ | 唯一确定，被试编号 13 |
| 「位置索引 13」 | ⚠️ 仅在与 `FOLD_5` 等内部常量对照时使用 | `train_subjects + val_subjects` 列表的第 13 项 = `Subject04` |
| 「被试 13」 | ❌ **禁止** | 歧义 |

### 当前列表顺序（`train_subjects` 28 项 + `val_subjects` 4 项 = 32 项）

```text
位置 0-27 (train):
  0=Subject35   1=Subject13   2=Subject26   3=Subject30   4=Subject16
  5=Subject31   6=Subject21   7=Subject12   8=Subject08   9=Subject17
 10=Subject09  11=Subject34  12=Subject00  13=Subject04  14=Subject29
 15=Subject15  16=Subject19  17=Subject05  18=Subject11  19=Subject01
 20=Subject24  21=Subject02  22=Subject33  23=Subject03  24=Subject32
 25=Subject23  26=Subject27  27=Subject10
位置 28-31 (val):
 28=Subject22  29=Subject18  30=Subject25  31=Subject06
```

（test 的 4 人被审计脚本有意排除，不在上表内：Subject20 / Subject07 / Subject14 / Subject28。）

### 已知踩坑记录

`runs/20261009-111900_teacher_b_data_audit/README.md` 曾把 `FOLD_5 = {1,10,19,26}`
这个**位置索引集合**当成被试编号，写成「被试 1,10,19,26」。
真实对应为 **Subject13 / Subject09 / Subject01 / Subject27**。已在该文件内勘误。

---

## 7. 日志补写工具：`scripts/restamp_run_logs.py`

### 为什么需要它

`runs/` 里的日志有一部分是**事后回填**的。回填时记录的 git 提交号，
和后来实际改动后的提交号不一样。如果直接覆盖 `meta.json`，
**原本记录的那个提交号就永久丢失了**，回溯时无法定位「当时跑的到底是哪一版代码」。

### 它怎么解决

把每一次记录到的提交号**追加**进 `meta.json` 的 `sha_history` 数组，
只更新顶层的 `git` 字段（表示"当前代码版本"）。
**`sha_history` 是只增数组，既有条目永不被删改。**

```text
首次运行：meta.git.sha          ->  sha_history[0]
之后运行：sha_history[1], [2], ...   逐次追加
```

### 用法

```bash
# 看所有日志当前的盖戳情况（只读，不改任何文件）
python scripts/restamp_run_logs.py --list

# 预演：只显示会改什么，不落盘
python scripts/restamp_run_logs.py --all --note "补上修复后的提交号" --dry-run

# 实际执行
python scripts/restamp_run_logs.py --all --note "补上修复后的提交号"

# 只处理某一个
python scripts/restamp_run_logs.py --run 20261009-150732_spectral_band_audit
```

**推荐节奏**：每次提交后跑一次 `--all --note "<本次提交做了什么>"`，
让日志始终能对应到正确的代码版本。

### 特性

- **幂等**：重复运行不会产生重复条目（已验证）
- **只增不减**：`sha_history` 里的历史条目不会被删除或改写
- **保字段**：`meta.json` 里的其他字段（`conclusions` / `reconciliation` / `dataset` 等）原样保留
- **`--dry-run`**：先看清楚再落盘
