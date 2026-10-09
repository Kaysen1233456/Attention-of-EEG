"""
心算数据集：类别不平衡 / 时长混淆 探针实验
------------------------------------------------
目的：检验 Gen5 报告的 8 折 CV BA=0.6734 是"真实认知信号"，还是部分来自
      "rest 录制时长是 focus 的 3 倍（91 vs 31 窗口）"这一采集设计 artifact。

四个条件（均在 32 个开发被试上做 subject-disjoint 8 折 GroupKFold）：
  A 原样        : 91 rest + 31 focus（复现项目协议）
  B 时长匹配    : 每被试随机抽 31 rest + 31 focus（消除不平衡与时长差）
  C 前段匹配    : 每被试取前 31 rest + 31 focus（检验段内漂移/位置效应）
  D 标签打乱    : 在 B 的窗口上、被试内打乱标签（零假设对照）

指标：窗口级 BA/F1/AUC + 被试级 BA（先算每个被试的 BA 再平均）+ 逐被试明细。

只读项目数据，不修改项目任何文件。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

DATA = Path(r"C:\Users\famsun\Attention\data\processed\mental_arithmetic_250hz_8ch")
BANDS = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45))
SR, NT = 250, 500


# ---------- 数据 ----------
def load_development(root):
    """与项目 cv_spectral_cnn.load_development 完全一致：按 rest/focus 连续段两两配对推断被试"""
    x = np.concatenate([np.load(root / f"{s}_waveforms.npy") for s in ("train", "val")])
    y = np.concatenate([np.load(root / f"{s}_labels.npy") for s in ("train", "val")])
    starts = np.r_[0, np.flatnonzero(y[1:] != y[:-1]) + 1, len(y)]
    runs = list(zip(starts[:-1], starts[1:]))
    assert len(runs) % 2 == 0, "Expected paired rest/focus runs per subject"
    groups = np.empty(len(y), dtype=np.int64)
    for subject, i in enumerate(range(0, len(runs), 2)):
        groups[runs[i][0]:runs[i + 1][1]] = subject
    return x.astype(np.float32), y.astype(np.int64), groups, runs


def normalize_windows(x):
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True) + 1e-6
    return np.clip((x - mean) / std, -8.0, 8.0).astype(np.float32)


def relative_band_power(x):
    power = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    freq = np.fft.rfftfreq(NT, d=1.0 / SR)
    total = power[..., (freq >= 1) & (freq <= 45)].mean(-1, keepdims=True) + 1e-8
    feats = [power[..., (freq >= lo) & (freq < hi)].mean(-1) / total.squeeze(-1)
             for lo, hi in BANDS]
    return np.log(np.concatenate(feats, axis=1) + 1e-8)


# ---------- 条件构造 ----------
def build_condition(runs, y, groups, mode, rng):
    """返回 (idx, labels, groups_out)"""
    n_subj = len(runs) // 2
    keep = []
    for s in range(n_subj):
        rest = np.arange(runs[2 * s][0], runs[2 * s][1])
        focus = np.arange(runs[2 * s + 1][0], runs[2 * s + 1][1])
        if mode == "A":
            sel = np.r_[rest, focus]
        elif mode == "B":
            sel = np.r_[rng.choice(rest, size=len(focus), replace=False), focus]
        elif mode == "C":
            sel = np.r_[rest[:len(focus)], focus]
        elif mode == "D":
            sel = np.r_[rng.choice(rest, size=len(focus), replace=False), focus]
        else:
            raise ValueError(mode)
        keep.append(sel)
    idx = np.concatenate(keep)
    lab = y[idx].copy()
    if mode == "D":  # 被试内打乱标签
        for s in range(n_subj):
            pass
        # 按被试块打乱
        lab2 = lab.copy()
        pos = 0
        for s in range(n_subj):
            n = len(keep[s])
            blk = lab2[pos:pos + n].copy()
            rng.shuffle(blk)
            lab2[pos:pos + n] = blk
            pos += n
        lab = lab2
    return idx, lab, groups[idx]


# ---------- 模型 ----------
class SpectralCNN(nn.Module):
    """与项目 scripts/cv_spectral_cnn.py 中 Gen5 模型结构一致"""

    def __init__(self, n_channels=8):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(n_channels, 16, kernel_size=9, padding=4), nn.BatchNorm1d(16), nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=7, padding=3), nn.BatchNorm1d(32), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.spectral = nn.Sequential(
            nn.LayerNorm(n_channels * len(BANDS)),
            nn.Linear(n_channels * len(BANDS), 32), nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32), nn.GELU(), nn.Dropout(0.1), nn.Linear(32, 2)
        )
        freq = torch.fft.rfftfreq(NT, d=1.0 / SR)
        self.register_buffer("total_mask", (freq >= 1) & (freq <= 45))
        self.register_buffer("band_masks", torch.stack(
            [(freq >= lo) & (freq < hi) for lo, hi in BANDS]))

    def forward(self, x):
        temporal = self.temporal(x).squeeze(-1)
        spec = torch.fft.rfft(x, dim=-1).abs().square()
        total = spec[..., self.total_mask].mean(-1, keepdim=True) + 1e-8
        bands = [spec[..., m].mean(-1) / total.squeeze(-1) for m in self.band_masks]
        spectral = self.spectral(torch.log(torch.cat(bands, dim=1) + 1e-8))
        return self.classifier(torch.cat([temporal, spectral], dim=1))


def predict(model, x, batch=128, device="cpu"):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            out.append(torch.softmax(
                model(torch.from_numpy(x[i:i + batch]).to(device)), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def train_cnn(x, y, tr, va, seed, epochs, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SpectralCNN().to(device)
    tr_loader = DataLoader(TensorDataset(torch.from_numpy(x[tr]), torch.from_numpy(y[tr])),
                           batch_size=64, shuffle=True)
    cnt = np.bincount(y[tr], minlength=2)
    w = torch.tensor(cnt.sum() / (2 * cnt), dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(weight=w)
    va_loader = DataLoader(TensorDataset(torch.from_numpy(x[va]), torch.from_numpy(y[va])), batch_size=128)
    best, best_state, stale = -np.inf, None, 0
    for _ in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb.to(device)), yb.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        model.eval()
        p = predict(model, x[va], device=device)
        ba = balanced_accuracy_score(y[va], p >= 0.5)
        if ba > best:
            best, best_state, stale = ba, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= 4:
                break
    model.load_state_dict(best_state)
    return model


def subject_level(y, pred, grp):
    rows = {}
    for s in np.unique(grp):
        m = grp == s
        if len(np.unique(y[m])) < 2:
            continue
        rows[int(s)] = float(balanced_accuracy_score(y[m], pred[m]))
    return rows


# ---------- 主流程 ----------
def run_condition(mode, x, y, groups, runs, seed, epochs, device, folds=8):
    rng = np.random.default_rng(seed)
    idx, lab, grp = build_condition(runs, y, groups, mode, rng)
    xs, ys = x[idx], lab

    # --- 线性参考：相对频带功率 + 逻辑回归 ---
    feats = relative_band_power(xs)
    lin_pred = np.zeros(len(ys), dtype=int)
    lin_prob = np.zeros(len(ys))
    for tr, va in GroupKFold(folds).split(feats, ys, grp):
        sc = StandardScaler().fit(feats[tr])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(feats[tr]), ys[tr])
        lin_pred[va] = clf.predict(sc.transform(feats[va]))
        lin_prob[va] = clf.predict_proba(sc.transform(feats[va]))[:, 1]
    lin = {
        "window_ba": float(balanced_accuracy_score(ys, lin_pred)),
        "window_f1": float(f1_score(ys, lin_pred, average="macro", zero_division=0)),
        "window_auc": float(roc_auc_score(ys, lin_prob)),
    }
    lin_sub = subject_level(ys, lin_pred, grp)
    lin["subject_ba_mean"] = float(np.mean(list(lin_sub.values())))
    lin["subject_ba_std"] = float(np.std(list(lin_sub.values())))

    # --- Gen5 CNN ---
    cnn_pred = np.zeros(len(ys), dtype=int)
    cnn_prob = np.zeros(len(ys))
    fold_rows = []
    for f, (tr, va) in enumerate(GroupKFold(folds).split(xs, ys, grp)):
        t0 = time.time()
        model = train_cnn(xs, ys, tr, va, seed + f, epochs, device)
        p = predict(model, xs[va], device=device)
        cnn_prob[va] = p
        cnn_pred[va] = (p >= 0.5).astype(int)
        fold_rows.append({
            "fold": f,
            "subjects": sorted(np.unique(grp[va]).tolist()),
            "balanced_accuracy": float(balanced_accuracy_score(ys[va], cnn_pred[va])),
            "roc_auc": float(roc_auc_score(ys[va], p)),
            "sec": round(time.time() - t0, 1),
        })
    cnn = {
        "window_ba": float(balanced_accuracy_score(ys, cnn_pred)),
        "window_f1": float(f1_score(ys, cnn_pred, average="macro", zero_division=0)),
        "window_auc": float(roc_auc_score(ys, cnn_prob)),
        "fold_ba_mean": float(np.mean([r["balanced_accuracy"] for r in fold_rows])),
        "fold_ba_std": float(np.std([r["balanced_accuracy"] for r in fold_rows])),
        "fold_ba_min": float(np.min([r["balanced_accuracy"] for r in fold_rows])),
        "fold_ba_max": float(np.max([r["balanced_accuracy"] for r in fold_rows])),
    }
    cnn_sub = subject_level(ys, cnn_pred, grp)
    cnn["subject_ba_mean"] = float(np.mean(list(cnn_sub.values())))
    cnn["subject_ba_std"] = float(np.std(list(cnn_sub.values())))
    cnn["n_subjects_above_0.60"] = int(sum(v > 0.60 for v in cnn_sub.values()))
    cnn["n_subjects_above_0.50"] = int(sum(v > 0.50 for v in cnn_sub.values()))
    cnn["n_subjects"] = len(cnn_sub)

    return {
        "condition": mode,
        "n_windows": int(len(ys)),
        "class_balance": {"rest": int((ys == 0).sum()), "focus": int((ys == 1).sum())},
        "linear_relative_bandpower": lin,
        "gen5_cnn": cnn,
        "per_subject_ba_cnn": {str(k): round(v, 4) for k, v in sorted(cnn_sub.items())},
        "folds": fold_rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--conditions", nargs="+", default=["A", "B", "C", "D"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    root = Path(args.data)
    x_raw, y, groups, runs = load_development(root)
    x = normalize_windows(x_raw)
    print(f"开发集: {x.shape}, 被试数={len(runs)//2}", flush=True)
    lens = [int(b - a) for a, b in runs]
    print(f"rest 段长度 min/med/max = {min(lens[0::2])}/{int(np.median(lens[0::2]))}/{max(lens[0::2])}", flush=True)
    print(f"focus 段长度 min/med/max = {min(lens[1::2])}/{int(np.median(lens[1::2]))}/{max(lens[1::2])}", flush=True)

    out = {"data": str(root), "epochs": args.epochs, "seed": args.seed,
           "runs_per_subject": {"rest_median": int(np.median(lens[0::2])),
                                "focus_median": int(np.median(lens[1::2]))},
           "results": []}
    for mode in args.conditions:
        print(f"\n===== 条件 {mode} =====", flush=True)
        r = run_condition(mode, x, y, groups, runs, args.seed, args.epochs, device)
        out["results"].append(r)
        print(f"  [线性 相对频带功率] BA={r['linear_relative_bandpower']['window_ba']:.4f} "
              f"被试级BA={r['linear_relative_bandpower']['subject_ba_mean']:.4f}", flush=True)
        print(f"  [Gen5 CNN]  窗口BA={r['gen5_cnn']['window_ba']:.4f} "
              f"被试级BA={r['gen5_cnn']['subject_ba_mean']:.4f} "
              f"折间 {r['gen5_cnn']['fold_ba_mean']:.4f}±{r['gen5_cnn']['fold_ba_std']:.4f} "
              f"[{r['gen5_cnn']['fold_ba_min']:.3f},{r['gen5_cnn']['fold_ba_max']:.3f}] "
              f"AUC={r['gen5_cnn']['window_auc']:.4f} "
              f">0.6的被试={r['gen5_cnn']['n_subjects_above_0.60']}/{r['gen5_cnn']['n_subjects']}", flush=True)

    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已写入 {args.output}", flush=True)


if __name__ == "__main__":
    main()
