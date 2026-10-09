"""
探针 4：Teacher B 在「被试级标准化」下还剩多少优势？
----------------------------------------------------
问题：Gen5 学生 + 被试级标准化 已达窗口BA 0.7440，而 Teacher B 原协议只有 0.7166。
      那么教师还有存在价值吗？必须在同一评估下对比。

配置（全部 8 折 subject-disjoint GroupKFold，epochs=15，与项目一致）：
  T0 原始集(91+31) 无标准化   → 应复现项目报告的 ~0.7166
  T1 时长匹配集     无标准化   → 与 S0 可比
  T2 时长匹配集     +被试级标准化 → 与 S1 可比

只读项目数据。

⚠️ 2026-10-09 修正（登记册 C10）
--------------------------------
原实现用 `subject_standardize(feats, grp)` 在 GroupKFold **之前**、用**全部**窗口
拟合被试级 μ/σ → 评估窗参与统计量估计 → T2 的收益被系统性高估（原报 +7.0 分）。
现改为协议 v1.3 方案 A：**只在静息段最靠前的 30 窗拟合 μ/σ**，校准窗从评估集剔除。
T0 的评估集 = 全部非校准窗；T1/T2 的时长匹配改在**非校准**静息窗中采样。
"""
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent))
from _subject_calibration import (  # noqa: E402
    DEFAULT_N_CALIB, calibrate_by_subject, describe, matched_eval_indices,
)

DATA = Path(r"C:\Users\famsun\Attention\data\processed\mental_arithmetic_250hz_8ch")
BANDS = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45))
SR, NT, NCH = 250, 500, 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PARAMS = dict(learning_rate=0.00030265400272101283, weight_decay=4.084129623824164e-05,
              dropout=0.14111876080331268, d_model=128, n_heads=4, n_layers=1, batch_size=128)


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
    power = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    freq = np.fft.rfftfreq(NT, d=1.0 / SR)
    total = power[..., (freq >= 1) & (freq <= 45)].mean(-1, keepdims=True) + 1e-8
    feats = [power[..., (freq >= lo) & (freq < hi)].mean(-1) / total.squeeze(-1) for lo, hi in BANDS]
    return np.log(np.concatenate(feats, axis=1) + 1e-8).astype(np.float32)


def subject_standardize(*args, **kwargs):
    """【已移除】旧接口 —— 折前拟合，见登记册 C10。等价替代见 _subject_calibration。"""
    from _subject_calibration import subject_standardize as _raise
    return _raise(*args, **kwargs)


class TeacherB(nn.Module):
    """与项目 scripts/teacher_b_common.py 的 TeacherB 完全一致，只是频域特征从外部传入"""

    def __init__(self, d_model=128, n_heads=4, n_layers=1, dropout=0.14, n_channels=8, n_time=500):
        super().__init__()
        self.temporal_branches = nn.ModuleList([
            nn.Sequential(nn.Conv1d(n_channels, 32, k, padding=k // 2), nn.BatchNorm1d(32), nn.GELU())
            for k in (7, 15, 31)])
        self.temporal_projection = nn.Conv1d(96, d_model, 1)
        self.pool = nn.AvgPool1d(kernel_size=10, stride=10)
        self.position = nn.Parameter(torch.zeros(1, n_time // 10, d_model))
        nn.init.trunc_normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
                                           dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.spectral = nn.Sequential(nn.LayerNorm(n_channels * len(BANDS)),
                                      nn.Linear(n_channels * len(BANDS), 64), nn.GELU(), nn.Dropout(dropout))
        self.classifier = nn.Sequential(nn.LayerNorm(d_model + 64), nn.Linear(d_model + 64, 128),
                                        nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 2))

    def forward(self, x, s):
        t = torch.cat([b(x) for b in self.temporal_branches], dim=1)
        t = self.temporal_projection(t)
        t = self.pool(t).transpose(1, 2)
        t = t + self.position[:, :t.size(1)]
        t = self.transformer(t).mean(dim=1)
        sp = self.spectral(s)
        return self.classifier(torch.cat([t, sp], dim=1))


def predict(model, xw, xs, batch=128):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(xw), batch):
            out.append(torch.softmax(model(torch.from_numpy(xw[i:i + batch]).to(DEVICE),
                                           torch.from_numpy(xs[i:i + batch]).to(DEVICE)), 1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def train(xw, xs, y, tr, va, seed, epochs=15):
    torch.manual_seed(seed); np.random.seed(seed)
    p = PARAMS
    model = TeacherB(d_model=p["d_model"], n_heads=p["n_heads"], n_layers=p["n_layers"],
                     dropout=p["dropout"]).to(DEVICE)
    loader = DataLoader(TensorDataset(torch.from_numpy(xw[tr]), torch.from_numpy(xs[tr]),
                                      torch.from_numpy(y[tr])), batch_size=p["batch_size"], shuffle=True)
    cnt = np.bincount(y[tr], minlength=2)
    w = torch.tensor(cnt.sum() / (2 * cnt), dtype=torch.float32, device=DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=p["learning_rate"], weight_decay=p["weight_decay"])
    crit = nn.CrossEntropyLoss(weight=w)
    best, best_state, stale = -np.inf, None, 0
    for _ in range(epochs):
        model.train()
        for xb, sb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb.to(DEVICE), sb.to(DEVICE)), yb.to(DEVICE))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
        pr = predict(model, xw[va], xs[va])
        ba = balanced_accuracy_score(y[va], pr >= 0.5)
        if ba > best:
            best, best_state, stale = ba, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= 4: break
    model.load_state_dict(best_state)
    return model


def run(name, xw, xs, y, grp, folds=8):
    n = len(y); prob = np.zeros(n); rows = []
    for f, (tr, va) in enumerate(GroupKFold(folds).split(xw, y, grp)):
        t0 = time.time()
        m = train(xw, xs, y, tr, va, 42 + f)
        p = predict(m, xw[va], xs[va])
        prob[va] = p
        rows.append({"fold": f, "balanced_accuracy": float(balanced_accuracy_score(y[va], p >= 0.5)),
                     "roc_auc": float(roc_auc_score(y[va], p)), "sec": round(time.time() - t0, 1)})
    pred = (prob >= 0.5).astype(int)
    subs = {int(s): float(balanced_accuracy_score(y[grp == s], pred[grp == s])) for s in np.unique(grp)}
    r = {"setting": name, "n_windows": int(n),
         "window_ba": float(balanced_accuracy_score(y, pred)),
         "window_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
         "window_auc": float(roc_auc_score(y, prob)),
         "subject_ba_mean": float(np.mean(list(subs.values()))),
         "fold_ba_mean": float(np.mean([x["balanced_accuracy"] for x in rows])),
         "fold_ba_std": float(np.std([x["balanced_accuracy"] for x in rows])),
         "fold_ba_min": float(np.min([x["balanced_accuracy"] for x in rows])),
         "n_subjects_above_0.60": int(sum(v > 0.60 for v in subs.values())),
         "per_subject": {str(k): round(v, 4) for k, v in sorted(subs.items())}, "folds": rows}
    print(f"{name:38s} 窗口BA={r['window_ba']:.4f} AUC={r['window_auc']:.4f} "
          f"折间{r['fold_ba_mean']:.4f}±{r['fold_ba_std']:.4f} min折={r['fold_ba_min']:.3f} "
          f">0.6被试={r['n_subjects_above_0.60']}/{len(np.unique(grp))}", flush=True)
    return r


def main():
    x_raw, y_all, groups_all, runs = load_development(DATA)
    xw_all = normalize_windows(x_raw)
    feats_all = band_power_feats(xw_all)

    # ── 折安全校准：只在静息段最靠前的 30 窗上拟合 μ/σ（协议 v1.3 方案 A）──
    feats_cal, calib_mask, calib_info = calibrate_by_subject(
        feats_all, y_all, groups_all, n_calib=DEFAULT_N_CALIB)

    print("=== 校准窗口预算（协议 §1.1 / R5）===")
    print(describe(groups_all, y_all, calib_mask))
    print()

    out = {
        "teacher_b_params": PARAMS,
        "device": str(DEVICE),
        "protocol": "CALIBRATION_PROTOCOL.md v1.3 方案 A: 静息段前 30 窗校准, 折安全拟合",
        "calibration": calib_info,
        "results": [],
    }

    # ── T0 原始集：评估 = 全部非校准窗（校准窗剔除，R2）──
    k0 = ~calib_mask
    out["results"].append(run("T0_original_no_std(calib_excluded)",
                              xw_all[k0], feats_all[k0], y_all[k0], groups_all[k0]))

    # ── T1/T2 时长匹配集：在【非校准】静息窗中采样 ──
    rng = np.random.default_rng(42)
    idx, skipped = matched_eval_indices(
        y_all, groups_all, calib_mask, rng, strict=False)
    xw, ys, grp = xw_all[idx], y_all[idx], groups_all[idx]
    out["skipped_subjects"] = skipped
    out["n_subjects_matched"] = int(len(np.unique(grp)))
    print("时长匹配：窗口 %d，被试 %d，跳过被试组 %s\n"
          % (len(idx), len(np.unique(grp)), [s["subject_group"] for s in skipped]))

    out["results"].append(run("T1_matched_no_std", xw, feats_all[idx], ys, grp))
    out["results"].append(run("T2_matched_+subject_std", xw, feats_cal[idx], ys, grp))

    Path("artifacts/probes/probe4_teacherB_domain.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n已写入 probe4_teacherB_domain.json")


if __name__ == "__main__":
    main()
