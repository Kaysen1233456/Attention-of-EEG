# Subject-Level Calibration Protocol (被试级校准协议)

版本：v1.3（2026-10-09）
状态：**生效**。R1 已定案为**方案 A（任务前静息前缀校准）**，见 §1.1 与 §2 R1。
适用范围：心算 EEG 的 Gen5 student baseline、Teacher B candidate 和后续蒸馏模型

> ⚠️ 这里的「student」**只是角色命名**。Gen5 与 Teacher B 目前是两个各自独立、
> 只用标签直接监督训练的模型；**蒸馏从未启动**，Teacher B 从未监督过 Gen5。
> 见 `README.md` 的「术语澄清」一节。

> **变更记录**
> - v1.3.1（2026-10-09）：**补入方案 A 下的实测收益**（RV-1 折安全重跑，见 §2 R1）。
>   旧报 +3.3~+7.2 分系 C10 折前拟合泄漏所致，已作废；**实测为 +0.4 ~ +3.1 分**。
> - v1.3（2026-10-09）：**R1 定案为方案 A**；§1.1 从「已知阻塞」改为「定案记录 + 升级路径」；
>   §3 新增**解封条件**（原「阻断无解封路径」）；§2 R1 重写。
> - v1.2（2026-10-09）：§1.1 记录「R1 不可满足」的实测证据；§1.2 定死校准作用阶段
>   （窗口级 z-score 之后）、新增禁止白化与最小样本数规则；新增 §2 R5（低样本被试双行报告）。
> - v1.1（2026-10-09）：修正 60 s / 30 窗口；补 R1–R4。

## 0. 目的与边界

开发集审计显示，被试间频谱差异大于被试内任务效应。此前的信噪比和预期提升
是特征预实验，不是模型泛化结果。阈值校准在嵌套 CV 中没有收益，因此下一项
实验验证被试级无标签校准，而不是继续搜索阈值。

本协议定义的是：

```text
跨被试模型 + 目标被试 60 秒无标签静息校准
```

（「该静息段是否独立于心算段」已定案为**方案 A：任务前静息前缀**，见 §1.1 与 §2 R1。）

这不再是 zero-shot cross-subject。校准期需要在推理前采集，且校准窗口不参与
任务评分。

## 1. 固定定义

### 1.1 校准数据

- 校准时长固定为 `60 s`。
- 当前数据是 2 秒不重叠窗口，因此固定为 `30` 个窗口。
- **校准数据 = 目标被试静息段（`SubjectNN_1.edf`）时间序最靠前的 30 个窗口。**
- 校准窗口由**位置**确定，不由标签确定（本方案根本不读 `label`）。
- 校准窗口必须在评估集中**完全剔除**（见 R2）。
- 当前数据中 Subject31 只有 40 个 rest 窗口，扣除 30 窗后仅剩 **10** 窗 →
  触发 R5 双行报告。120 秒协议只能作为另一个明确命名的探索性协议。

> ### ✅ 定案：采用方案 A（任务前静息前缀校准）
>
> **决策记录**：2026-10-09，项目 owner 拍板采用**方案 A**。
> 方案 C（独立第三次记录）列为**后续升级路径**，待取得数据后另行评审。
>
> **背景证据**（云端核查 + 官方文档，见
> `experiments_log/runs/20261009-150224_eegmat_provenance_audit/`）
>
> 数据集为 PhysioNet `eegmat` v1.0.0，云端 `eegmat/` 共 **72 个 EDF = 36 被试 × 2 段**，
> **无第三段、无隐藏子目录**。官方 README.txt 原文：
> *"Each folder contains two recording files per subject: with `_1` postfix -- the recording of
> the background EEG of a subject (before mental arithmetic task); with `_2` postfix -- the
> recording of EEG during the mental arithmetic task."*
>
> 即：**每个被试只有「心算前的静息段」和「心算段」两段，二者属于同一次 session。**
> 因此原 R1 要求的「独立静息 recording」在数据来源层面**不存在** ——
> 这不是下载不全，无法通过重新下载或改变窗口切法解决。
>
> **定案内容见 §2 R1。**

### 1.2 校准特征与作用阶段

**校准作用在「窗口级 z-score（含 clip ±8）之后」产生的 40 维频谱向量上。**

这是本协议唯一允许的校准阶段。定义如下（顺序不可调换）：

```text
① 波形：2 s / 500 采样点原始窗口
② 窗口级 z-score：(x - mean_window) / (std_window + eps)，再 clip 到 ±8
③ 频谱：对 ② 之后的波形做 rfft，算各频带绝对功率
④ 相对功率：band_power / total_power_1_to_45Hz
⑤ log：log(relative_power + 1e-8)          <- 得到 40 维向量
⑥ 被试级校准：f_calibrated = (f - subject_mean) / (subject_std + epsilon)
```

```text
每通道：log(band_power / total_power_1_to_45Hz)
频带：delta, theta, alpha, beta, gamma
布局：ch0_delta ... ch0_gamma, ..., ch7_delta ... ch7_gamma
```

> **为什么必须写死「第 ② 步之后」**
>
> 早期措辞写成「原始窗口上的 40 维向量」，但模型实际是在**窗口 z-score 之后**才算频谱。
> 两者**不等价**：相对频带功率对窗口 z-score **不是不变的**，实测差异最大 **4.52**、
> 平均 **0.154**（log 域），成因是 z-score 里的**减均值**（clip 未触发，0% 采样被裁）。
> 若实现阶段不一致，同一份数据会算出不同的统计量，且难以察觉。
> **训练、验证、推理三处必须完全一致。**

### 1.3 校准变换的限制

- **禁止白化（whitening）。** 30 个校准窗口无法可靠估计 40 维协方差矩阵：
  实测 `matrix_rank(cov) = 29 < 40`，条件数 `3.8e18`，全协方差白化在数学上不可行，
  强行求逆只会放大噪声。
- **只允许对角标准化**，即 `(f - subject_mean) / (subject_std + epsilon)`，
  其中 `subject_std` 逐维独立估计。实测对角 std 范围 `0.24 – 0.61`，量级安全。
- **最小样本数**：校准窗口数必须 ≥ `30`（即固定 60 s）。不足 30 的被试标记为
  `calibration_insufficient`，不得用缩减窗口数蒙混过关（见 R5）。

## 2. 硬规则

### R1 - 无标签的静息校准段（**已定案：方案 A**）

- 校准数据 = 目标被试**静息段时间序最靠前的 30 个窗口**（60 s），见 §1.1。
- 只允许使用目标被试自己的记录；**禁止**使用其他被试的数据。
- 禁止使用 processed `label` 值挑选校准窗口。本方案由**位置**选窗，不读 label。
- 校准窗口必须在评估集中完全剔除（R2）。
- 结果必须显式标注 `calibration_scope: "within_session"`。
- **禁止**声称跨 session 泛化、跨被试泛化或域泛化。

**为什么这是合法且贴近部署的**

官方说明：`_1` 是「心算任务**之前**的背景 EEG」，`_2` 是「心算任务**期间**」，
两段属于同一次坐下来的记录。而边缘部署的真实流程正是：
**戴上设备 → 静默 60 s 采集基线 → 开始任务**。
所以「用任务前的静息段做校准」不是取巧，它是对部署流程的忠实建模。

**必须承担的代价（不可省略）**

1. 只能声称 `within-session calibration`。
2. **必须分开报告，不得只报合并值**：
   - **心算窗 BA**（`focus` 窗）← **唯一有部署意义、可对外引用的数字**
   - 留出静息窗 BA（校准窗之外的静息窗）← 仅供参考；因与校准段同源，**会系统性偏高**
3. 任何「提升 X 分」的结论，必须同时给出 zero-shot 对照（R4）。

**实测收益（方案 A，2026-10-09 RV-1 折安全重跑）**

> ⚠️ 引用校准收益时**只能**用下表数值。此前的「+3.3 ~ +7.2 分」来自含 **C10 泄漏**
> （统计量在 `GroupKFold` **之前**用全部窗口拟合）的实现，**已作废**。
> 重跑日志：`experiments_log/runs/20261009-154628_probe_rv1_domain_adaptation_v2`、
> `..._154822_probe_rv1_cnn_domain`、`..._155017_probe_rv1_teacherB_domain`、
> `..._155207_probe_rv1_matrix_2x2`。对照表见 `artifacts/probes/_superseded/README.md`。

| 模型 / 评估集 | zero-shot | + 校准 | 增益 |
|---|---|---|---|
| 线性（log 相对频带功率），时长匹配 1922 窗 | 0.6348 | 0.6967 | **+6.2** |
| Gen5，时长匹配 1922 窗 | 0.6956 | 0.7263 | **+3.1** |
| Teacher B，时长匹配 1922 窗 | 0.6899 | 0.7180 | **+2.8** |
| Gen5，原始集 2890 窗（校准窗已剔除） | 0.7232 | 0.7275 | **+0.4** |
| Teacher B，原始集 2890 窗（校准窗已剔除） | 0.7130 | 0.7314 | **+1.8** |

**怎么读这张表**

1. **增益真实存在但量级小**：+0.4 ~ +3.1 分（旧报 +3.3 ~ +7.2）。
2. 折间 std 为 ±0.045 ~ ±0.075，因此 **Gen5 在原始集上的 +0.4 分落在噪声内**，
   不得据此单独声称校准有效 —— 需多 seed 重复。
3. 校准对 **Teacher B 的帮助（+1.8）大于对 Gen5（+0.4）**。
4. ⚠️ 同口径下 **Teacher B 并不优于 Gen5**（0.7130 vs 0.7232，学生反高 1.0 分，参数少 23 倍）
   —— 这直接影响「先做强教师再蒸馏」的路线前提，见 `PROJECT_ISSUES_PLAIN.md`。

**升级路径（方案 C，暂缓）**

Owner 计划后续尝试获取**独立第三次记录**（新数据集或在公司环境重新采集），届时按以下步骤升级：

1. 落盘 `recording_id` / `session_id` provenance（manifest）
2. 用独立记录作校准，评估集不含该校准段
3. 将 `calibration_scope` 由 `within_session` 升级为 `cross_session`
4. **必须同时保留方案 A 的结果，两行并列报告**

> ⚠️ 第 4 项不可跳过。否则无法区分「独立性带来的收益」与「换了数据集带来的差异」，
> 升级本身会变成一个混杂实验。

### R2 - 校准窗口不参与评分

计算统计量的 30 个窗口必须从 BA、Macro-F1、ROC-AUC、混淆矩阵和逐被试指标
中完全剔除。校准是部署前适应期，不是任务推理期。

### R3 - 统计量必须落盘并复用

```json
{
  "subject": "Subject07",
  "calibration_seconds": 60,
  "n_calibration_windows": 30,
  "calibration_window_indices": [0, 1, 2],
  "calibration_scope": "within_session",
  "source": "pretask_rest_prefix_unlabeled",
  "feature_layout": ["ch0_delta", "ch0_theta", "...", "ch7_gamma"],
  "mean": [],
  "std": [],
  "epsilon": 1e-6,
  "whitening": false,
  "transductive": true,
  "scored_calibration_windows": false
}
```

训练和评估脚本必须从该文件读取统计量，不得在评分循环中即时重算。所有
统计量文件必须包含输入数据 hash、recording id、窗口索引和代码版本。

> **`source` / `calibration_scope` 取值规则**
>
> | 情形 | `source` | `calibration_scope` |
> |---|---|---|
> | 静息段前 30 窗（**当前唯一允许**） | `pretask_rest_prefix_unlabeled` | `within_session` |
> | 独立静息记录（升级后） | `independent_rest_recording_unlabeled` | `cross_session` |
>
> **不得填写 `independent_rest_recording_unlabeled`** —— 当前数据集不存在该记录。
> `whitening` 必须为 `false`（见 §1.3）。
> `transductive` 为 `true`：统计量取自目标被试自己的无标签窗口，这是本方案的**预期行为**，
> 但必须如实标注，不得当作 zero-shot 结果呈现。

### R4 - 两种模型都必须有 zero-shot 对照

每次 calibrated 实验必须同时报告：

1. Gen5 zero-shot 与 Gen5 calibrated；
2. Teacher B zero-shot 与 Teacher B calibrated；
3. 同一 fold、seed、epoch 规则和评分窗口下的 calibrated minus zero-shot。

### R5 - 低样本被试必须双行报告

- **判定条件**：某被试在扣除校准窗口后，剩余**静息评估窗 < `15`**，则该被试标记
  `rest_after_cal_low`。
- **报告要求**：所有结果表必须**同时给出两行** ——
  (a) **含**该被试；(b) **不含**该被试。
  不得只报其中一行，也不得为了让他"够数"而减少校准窗口数（那会违反 §1.3 最小样本数）。
- **理由**：窗口数过少时，逐被试 BA 的随机性极大，会把总体均值带偏；双行报告让读者
  自行判断该被试的影响。
- **当前已知会触发者**：

  | 被试 | 静息段 | 静息窗 | 扣 30 窗校准后 | 是否触发 |
  |---|---|---|---|---|
  | **Subject31** | **80 s** | **40** | **10** | ✅ **触发** |
  | Subject04 | 170 s | 85 | 55 | ❌ |
  | Subject10 | 188 s | 94 | 64 | ❌ |
  | 其余 33 人 | 182 s | 91 | 61 | ❌ |

  （数据来源：`experiments_log/runs/20261009-150224_eegmat_provenance_audit/`，
  云端逐文件 EDF 头实测，与本地预测完全一致。）

## 3. 嵌套评估要求

1. train/validation/test subject 集合互不相交。
2. 至少使用 seed `42`, `43`, `44`，报告 mean/std。
3. 使用嵌套 GroupKFold；内层选择 epoch，外层只评估一次。
4. 每个外层被试使用**自己的静息段前 30 窗**作校准；校准窗口不计分。
5. 报告每个被试的 BA、AUC、混淆矩阵以及所有 fold 的最低 BA。
6. 报告 Subject14、fold5 被试、以及**所有被标记 `rest_after_cal_low` 的被试**
   （当前为 **Subject31**，见 R5）的结果。
7. 阈值固定为 `0.5`；不再进行阈值搜索。

训练被试也必须按同一规则使用自己的校准段（静息段前 30 窗），否则训练输入
和目标被试推理输入不一致。

### 解封条件（原「阻断无解封路径」）

实验在**以下任一条件**成立时可以继续；两条都不满足时，必须停止并标记
`calibration_not_available`，**不得**回退到带标签的 rest run。

| 条件 | 适用情形 | 必须的标注 |
|---|---|---|
| **条件 1** | 使用静息段前 30 窗作校准（**当前生效**） | `calibration_scope: "within_session"` |
| **条件 2** | 使用独立静息记录作校准 | `calibration_scope: "cross_session"`，且 manifest 含 `recording_id` / `session_id` |

两种条件下都必须同时满足 **R2**（校准窗不计分）与 **R5**（低样本被试双行报告）。
条件 2 额外要求按 R1「升级路径」第 4 项**并列报告条件 1 的结果**。

## 4. 待办

| # | 任务 | 产出 |
|---|---|---|
| 1 | 补充 recording/session/window provenance | manifest with explicit calibration windows |
| 2 | 实现 `calibrate_subject.py` 并落盘 40 维统计量 | `artifacts/calibration/<subject>.json` |
| 3 | 将 calibrated spectral branch 接入 Gen5 和 Teacher B | versioned feature pipeline |
| 4 | 嵌套 CV 重跑 Gen5，报告 zero-shot/calibrated | `artifacts/gen5_calibrated_nested/` |
| 5 | 同口径重跑 Teacher B | `artifacts/teacher_b_calibrated_nested/` |
| 6 | 更新对外文档和部署说明 | 60 s calibration requirement |

在任务 1 完成前，不得声称 calibrated nested CV 已完成，也不得启动蒸馏。

## 5. 不变项

Teacher B 的窗口级归一化和 BatchNorm 仍保持不变，直到完成独立单因素对照。
移除 BatchNorm、改变窗口归一化或改变频谱定义都必须成为单独实验，并同时报告
zero-shot 与 calibrated 两行，不能混入本协议的第一轮验证。
