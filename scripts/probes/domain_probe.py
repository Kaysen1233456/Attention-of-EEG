"""
探针 2：跨被试域适应 —— 被试级特征标准化是否有效？
--------------------------------------------------
背景：审计显示不同被试的相对频带功率差异极大（如 rest 段 alpha：Subject26=0.297 vs Subject13=0.132），
      说明"被试身份"本身是极强的 nuisance 变量。跨被试场景下，用被试自己的（无标签）窗口
      做标准化是合法的，且常能显著提升。

条件（全部在时长匹配设置下：每被试 31 rest + 31 focus）：
  N0 现状      : log 相对频带功率，不做被试级标准化
  N1 被试级标准化: 用该被试**校准窗**的均值/方差，对该被试的 40 维频带特征做 z-score（无标签）
  N2 被试+通道级: 同上，但**按通道池化** —— 每个通道的 5 个频带共用一个 μ/σ 标量

只用线性模型（相对频带功率 + 逻辑回归），确定性、秒级完成。
只读项目数据。

⚠️ 2026-10-09 修正（登记册 C10）
--------------------------------
原实现用 `subject_standardize(feats, grp)` 在 GroupKFold **之前**、用**全部**窗口
拟合被试级 μ/σ → 评估窗参与了统计量估计 → 收益被系统性高估。
现改为按 `CALIBRATION_PROTOCOL.md` v1.3 方案 A：
**只在静息段最靠前的 30 个窗口上拟合 μ/σ**，并把校准窗从评估集中剔除。
时长匹配采样改为在**非校准**静息窗中进行。
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from _subject_calibration import (  # noqa: E402
    DEFAULT_N_CALIB, calibrate_by_subject, channel_groups, describe,
    matched_eval_indices,
)

DATA = Path(r"C:\Users\famsun\Attention\data\processed\mental_arithmetic_250hz_8ch")
BANDS = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45))
SR, NT, NCH = 250, 500, 8


def load_development(root):
    x = np.concatenate([np.load(root / f"{s}_waveforms.npy") for s in ("train", "val")])
    y = np.concatenate([np.load(root / f"{s}_labels.npy") for s in ("train", "val")])
    starts = np.r_[0, np.flatnonzero(y[1:] != y[:-1]) + 1, len(y)]
    runs = list(zip(starts[:-1], starts[1:]))
    groups = np.empty(len(y), dtype=np.int64)
    for subject, i in enumerate(range(0, len(runs), 2)):
        groups[runs[i][0]:runs[i + 1][1]] = subject
    return x.astype(np.float32), y.astype(np.int64), groups, runs


def normalize_windows(x):
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True) + 1e-6
    return np.clip((x - mean) / std, -8.0, 8.0).astype(np.float32)


def band_power_feats(x):
    """(N, 40): 每通道 5 个相对频带功率，取 log"""
    power = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    freq = np.fft.rfftfreq(NT, d=1.0 / SR)
    total = power[..., (freq >= 1) & (freq <= 45)].mean(-1, keepdims=True) + 1e-8
    feats = [power[..., (freq >= lo) & (freq < hi)].mean(-1) / total.squeeze(-1)
             for lo, hi in BANDS]                      # list of (N, 8)
    return np.log(np.concatenate(feats, axis=1) + 1e-8)  # (N, 40)


def subject_standardize(*args, **kwargs):
    """【已移除】旧接口 —— 折前拟合，见登记册 C10。保留仅为给出明确报错。"""
    from _subject_calibration import subject_standardize as _raise
    return _raise(*args, **kwargs)


def subject_channel_standardize(*args, **kwargs):
    """【已移除】旧接口 —— 折前拟合，见登记册 C10。

    等价替代：`calibrate_by_subject(feats, y, grp, groups=channel_groups(n_dim, n_channels))`。
    """
    raise RuntimeError(
        "subject_channel_standardize() 已被移除（折前拟合，登记册 C10）。"
        "请改用 calibrate_by_subject(feats, y, grp, groups=channel_groups(n_dim, n_channels))。"
    )


def evaluate(feats, y, grp, folds=8, seed=42):
    pred = np.zeros(len(y), dtype=int)
    prob = np.zeros(len(y))
    for tr, va in GroupKFold(folds).split(feats, y, grp):
        sc = StandardScaler().fit(feats[tr])
        clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(sc.transform(feats[tr]), y[tr])
        pred[va] = clf.predict(sc.transform(feats[va]))
        prob[va] = clf.predict_proba(sc.transform(feats[va]))[:, 1]
    subs = {}
    for s in np.unique(grp):
        m = grp == s
        subs[int(s)] = float(balanced_accuracy_score(y[m], pred[m]))
    return {
        "window_ba": float(balanced_accuracy_score(y, pred)),
        "window_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "window_auc": float(roc_auc_score(y, prob)),
        "subject_ba_mean": float(np.mean(list(subs.values()))),
        "subject_ba_std": float(np.std(list(subs.values()))),
        "n_subjects_above_0.60": int(sum(v > 0.60 for v in subs.values())),
        "per_subject": {str(k): round(v, 4) for k, v in sorted(subs.items())},
    }


def main():
    x_raw, y_all, groups_all, runs = load_development(DATA)
    x_all = normalize_windows(x_raw)
    feats_all = band_power_feats(x_all)

    # ── 折安全校准：只在静息段最靠前的 30 窗上拟合 μ/σ（协议 v1.3 方案 A）──
    feats_cal, calib_mask, calib_info = calibrate_by_subject(
        feats_all, y_all, groups_all, n_calib=DEFAULT_N_CALIB)
    # N2：按【通道】分组池化 —— 每个通道的 5 个频带共享一个 μ/σ 标量。
    # ⚠️ 原 `block=len(BANDS)` 是无效参数（逐列均值与分块无关），
    #    导致 N2 与 N1 数学上完全相同；现用 channel_groups 给出真正的通道级变体。
    n_ch = feats_all.shape[1] // len(BANDS)
    feats_cal_ch, _, calib_info_ch = calibrate_by_subject(
        feats_all, y_all, groups_all, n_calib=DEFAULT_N_CALIB,
        groups=channel_groups(feats_all.shape[1], n_ch))

    print("=== 校准窗口预算（协议 §1.1 / R5）===")
    print(describe(groups_all, y_all, calib_mask))
    print()

    # ── 时长匹配：在【非校准】静息窗中采样（校准窗不参与评估，R2）──
    rng = np.random.default_rng(42)
    idx, skipped = matched_eval_indices(
        y_all, groups_all, calib_mask, rng, strict=False)
    y, grp = y_all[idx], groups_all[idx]
    n_subj = int(len(np.unique(grp)))
    print("时长匹配后窗口数 = %d，被试 = %d，跳过被试组 = %s"
          % (len(idx), n_subj, [s["subject_group"] for s in skipped]))
    print()

    conditions = {
        "N0_raw": feats_all[idx],
        "N1_subject_std": feats_cal[idx],
        "N2_subject_channel_std": feats_cal_ch[idx],
    }

    out = {
        "setting": "duration_matched (rest:focus = 1:1, 校准窗已从评估集剔除)",
        "protocol": "CALIBRATION_PROTOCOL.md v1.3 方案 A: 静息段前 30 窗校准, 折安全拟合",
        "n_windows": int(len(idx)),
        "n_subjects": n_subj,
        "calibration": calib_info,
        "calibration_channel_pooled": calib_info_ch,
        "skipped_subjects": skipped,
        "results": {},
    }
    for name, f in conditions.items():
        r = evaluate(f, y, grp)
        out["results"][name] = r
        print(f"{name:26s} 窗口BA={r['window_ba']:.4f}  被试级BA={r['subject_ba_mean']:.4f}"
              f"±{r['subject_ba_std']:.4f}  AUC={r['window_auc']:.4f}  "
              f">0.6被试={r['n_subjects_above_0.60']}/{n_subj}")

    Path("artifacts/probes/probe2_domain_adaptation.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n已写入 probe2_domain_adaptation.json")


if __name__ == "__main__":
    main()
