"""
探针 3：被试级标准化对 Gen5 CNN 是否同样有效？
------------------------------------------------
在 Gen5 模型（temporal CNN + 相对频带功率）上，只改一处：
  频域分支的输入 从「原始 log 相对频带功率」改为「被试级标准化的 log 相对频带功率」
时序分支完全不变。

对比：
  S0 = 现状 Gen5
  S1 = Gen5 + 频域分支被试级标准化（无标签，需目标被试约 60s 校准数据）
  S2 = 现状 Gen5 + 3 seed 软投票集成（不动模型，纯集成）

设置：时长匹配（每被试 31 rest + 31 focus），8 折 subject-disjoint GroupKFold。
只读项目数据。

⚠️ 2026-10-09 修正（登记册 C10）
--------------------------------
原实现用 `subject_standardize(feats, grp)` 在 GroupKFold **之前**、用**全部**窗口
拟合被试级 μ/σ → 评估窗参与统计量估计 → 收益被系统性高估（原报 +5.4 分）。
现改为协议 v1.3 方案 A：**只在静息段最靠前的 30 窗拟合 μ/σ**，校准窗从评估集剔除。
时长匹配采样改在**非校准**静息窗中进行。
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
    feats = [power[..., (freq >= lo) & (freq < hi)].mean(-1) / total.squeeze(-1)
             for lo, hi in BANDS]
    return np.log(np.concatenate(feats, axis=1) + 1e-8).astype(np.float32)


def subject_standardize(*args, **kwargs):
    """【已移除】旧接口 —— 折前拟合，见登记册 C10。等价替代见 _subject_calibration。"""
    from _subject_calibration import subject_standardize as _raise
    return _raise(*args, **kwargs)


class Gen5(nn.Module):
    """temporal 分支吃波形；spectral 分支吃外部传入的 40 维频带特征"""

    def __init__(self, n_channels=8):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(n_channels, 16, 9, padding=4), nn.BatchNorm1d(16), nn.GELU(),
            nn.Conv1d(16, 32, 7, padding=3), nn.BatchNorm1d(32), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.spectral = nn.Sequential(
            nn.LayerNorm(n_channels * len(BANDS)),
            nn.Linear(n_channels * len(BANDS), 32), nn.GELU(),
        )
        self.classifier = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Dropout(0.1), nn.Linear(32, 2))

    def forward(self, xw, xs):
        t = self.temporal(xw).squeeze(-1)
        s = self.spectral(xs)
        return self.classifier(torch.cat([t, s], dim=1))


def predict(model, xw, xs, batch=128):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(xw), batch):
            logits = model(torch.from_numpy(xw[i:i + batch]).to(DEVICE),
                           torch.from_numpy(xs[i:i + batch]).to(DEVICE))
            out.append(torch.softmax(logits, 1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def train(xw, xs, y, tr, va, seed, epochs=15):
    torch.manual_seed(seed); np.random.seed(seed)
    model = Gen5().to(DEVICE)
    loader = DataLoader(TensorDataset(torch.from_numpy(xw[tr]), torch.from_numpy(xs[tr]),
                                      torch.from_numpy(y[tr])), batch_size=64, shuffle=True)
    cnt = np.bincount(y[tr], minlength=2)
    w = torch.tensor(cnt.sum() / (2 * cnt), dtype=torch.float32, device=DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(weight=w)
    best, best_state, stale = -np.inf, None, 0
    for _ in range(epochs):
        model.train()
        for xb, sb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb.to(DEVICE), sb.to(DEVICE)), yb.to(DEVICE))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
        p = predict(model, xw[va], xs[va])
        ba = balanced_accuracy_score(y[va], p >= 0.5)
        if ba > best:
            best, best_state, stale = ba, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= 4: break
    model.load_state_dict(best_state)
    return model


def evaluate_setting(name, xw, xs, y, grp, seeds=(42,), folds=8):
    n = len(y)
    prob = np.zeros(n)
    fold_rows = []
    for f, (tr, va) in enumerate(GroupKFold(folds).split(xw, y, grp)):
        t0 = time.time()
        ps = []
        for sd in seeds:
            m = train(xw, xs, y, tr, va, sd + f)
            ps.append(predict(m, xw[va], xs[va]))
        p = np.mean(ps, axis=0)
        prob[va] = p
        fold_rows.append({"fold": f, "balanced_accuracy": float(balanced_accuracy_score(y[va], p >= 0.5)),
                          "roc_auc": float(roc_auc_score(y[va], p)), "sec": round(time.time() - t0, 1)})
    pred = (prob >= 0.5).astype(int)
    subs = {int(s): float(balanced_accuracy_score(y[grp == s], pred[grp == s])) for s in np.unique(grp)}
    r = {
        "setting": name, "n_seeds": len(seeds),
        "window_ba": float(balanced_accuracy_score(y, pred)),
        "window_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "window_auc": float(roc_auc_score(y, prob)),
        "subject_ba_mean": float(np.mean(list(subs.values()))),
        "subject_ba_std": float(np.std(list(subs.values()))),
        "fold_ba_mean": float(np.mean([f["balanced_accuracy"] for f in fold_rows])),
        "fold_ba_std": float(np.std([f["balanced_accuracy"] for f in fold_rows])),
        "fold_ba_min": float(np.min([f["balanced_accuracy"] for f in fold_rows])),
        "n_subjects_above_0.60": int(sum(v > 0.60 for v in subs.values())),
        "per_subject": {str(k): round(v, 4) for k, v in sorted(subs.items())},
        "folds": fold_rows,
    }
    print(f"{name:34s} 窗口BA={r['window_ba']:.4f} 被试级BA={r['subject_ba_mean']:.4f}"
          f"±{r['subject_ba_std']:.4f} AUC={r['window_auc']:.4f} "
          f"折间{r['fold_ba_mean']:.4f}±{r['fold_ba_std']:.4f} "
          f"min折={r['fold_ba_min']:.3f} >0.6被试={r['n_subjects_above_0.60']}/{len(np.unique(grp))}",
          flush=True)
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

    # ── 时长匹配：在【非校准】静息窗中采样（校准窗不参与评估，R2）──
    rng = np.random.default_rng(42)
    idx, skipped = matched_eval_indices(
        y_all, groups_all, calib_mask, rng, strict=False)
    xw, ys, grp = xw_all[idx], y_all[idx], groups_all[idx]
    n_subj = int(len(np.unique(grp)))

    print(f"device={DEVICE}  时长匹配集: {xw.shape}  被试={n_subj}  "
          f"跳过被试组={[s['subject_group'] for s in skipped]}\n", flush=True)

    out = {
        "setting": "duration_matched (rest:focus = 1:1, 校准窗已从评估集剔除)",
        "protocol": "CALIBRATION_PROTOCOL.md v1.3 方案 A: 静息段前 30 窗校准, 折安全拟合",
        "device": str(DEVICE),
        "n_windows": int(len(idx)),
        "n_subjects": n_subj,
        "calibration": calib_info,
        "skipped_subjects": skipped,
        "results": [],
    }
    f_raw, f_cal = feats_all[idx], feats_cal[idx]
    out["results"].append(evaluate_setting("S0_Gen5_baseline", xw, f_raw, ys, grp, seeds=(42,)))
    out["results"].append(evaluate_setting("S1_Gen5_+subject_std_spectral", xw, f_cal, ys, grp, seeds=(42,)))
    out["results"].append(evaluate_setting("S2_Gen5_+subject_std_3seed_ens", xw, f_cal, ys, grp, seeds=(42, 43, 44)))

    Path("artifacts/probes/probe3_cnn_domain.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n已写入 probe3_cnn_domain.json")


if __name__ == "__main__":
    main()
