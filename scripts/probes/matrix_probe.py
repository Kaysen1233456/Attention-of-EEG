"""
探针 5（决定性）：教师 vs 学生 的 2x2 对照
------------------------------------------
在同一评估协议（原始集，8 折 subject-disjoint GroupKFold，seed=42+fold，epochs=15）
下，同时改变两个因素：
  A 因素：模型      {Gen5 学生, Teacher B}
  B 因素：频域输入   {原始 log 相对频带功率, 被试级标准化}

得到 2x2 表，回答：教师相对学生的优势，有多少来自"模型更强"，
有多少来自"被试级标准化"。

只读项目数据。

⚠️ 2026-10-09 修正（登记册 C10）
--------------------------------
原实现用 `subject_standardize(feats, grp)` 在 GroupKFold **之前**、用**全部**窗口
拟合被试级 μ/σ → 评估窗参与统计量估计 → "+subject_std" 两格的收益被系统性高估
（原报 gen5 +4.2 分 / teacherB +3.3 分）。
现改为协议 v1.3 方案 A：**只在静息段最靠前的 30 窗拟合 μ/σ**，校准窗从评估集剔除。
评估集因此从「91 rest + 31 focus」变为「61 rest + 31 focus」（与 probe4 的 T0 同口径）。
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
    DEFAULT_N_CALIB, calibrate_by_subject, describe,
)
from teacherB_probe import TeacherB, PARAMS  # noqa: E402
from cnn_domain_probe import Gen5  # noqa: E402

DATA = Path(r"C:\Users\famsun\Attention\data\processed\mental_arithmetic_250hz_8ch")
BANDS = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45))
SR, NT = 250, 500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_development(root):
    x = np.concatenate([np.load(root / f"{s}_waveforms.npy") for s in ("train", "val")])
    y = np.concatenate([np.load(root / f"{s}_labels.npy") for s in ("train", "val")])
    starts = np.r_[0, np.flatnonzero(y[1:] != y[:-1]) + 1, len(y)]
    runs = list(zip(starts[:-1], starts[1:]))
    groups = np.empty(len(y), dtype=np.int64)
    for subject, i in enumerate(range(0, len(runs), 2)):
        groups[runs[i][0]:runs[i + 1][1]] = subject
    return x.astype(np.float32), y.astype(np.int64), groups


def normalize_windows(x):
    return np.clip((x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-6),
                   -8.0, 8.0).astype(np.float32)


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


def build(model_name):
    if model_name == "gen5":
        return Gen5(), dict(lr=1e-3, wd=1e-4, bs=64)
    return TeacherB(d_model=PARAMS["d_model"], n_heads=PARAMS["n_heads"],
                    n_layers=PARAMS["n_layers"], dropout=PARAMS["dropout"]), \
        dict(lr=PARAMS["learning_rate"], wd=PARAMS["weight_decay"], bs=PARAMS["batch_size"])


def train_eval(model_name, xw, xs, y, tr, va, seed, epochs=15):
    torch.manual_seed(seed); np.random.seed(seed)
    model, hp = build(model_name)
    model = model.to(DEVICE)
    loader = DataLoader(TensorDataset(torch.from_numpy(xw[tr]), torch.from_numpy(xs[tr]),
                                      torch.from_numpy(y[tr])), batch_size=hp["bs"], shuffle=True)
    cnt = np.bincount(y[tr], minlength=2)
    w = torch.tensor(cnt.sum() / (2 * cnt), dtype=torch.float32, device=DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    crit = nn.CrossEntropyLoss(weight=w)
    best, best_state, stale = -np.inf, None, 0
    for _ in range(epochs):
        model.train()
        for xb, sb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb.to(DEVICE), sb.to(DEVICE)), yb.to(DEVICE))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
        model.eval()
        with torch.no_grad():
            p = torch.softmax(model(torch.from_numpy(xw[va]).to(DEVICE),
                                    torch.from_numpy(xs[va]).to(DEVICE)), 1)[:, 1].cpu().numpy()
        ba = balanced_accuracy_score(y[va], p >= 0.5)
        if ba > best:
            best, best_state, stale = ba, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= 4: break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p = torch.softmax(model(torch.from_numpy(xw[va]).to(DEVICE),
                                torch.from_numpy(xs[va]).to(DEVICE)), 1)[:, 1].cpu().numpy()
    return p


def run(model_name, std, xw, xs, y, grp, folds=8):
    """xs 为**已算好**的频域特征（原始或折安全校准后的），不再在函数内部做被试级标准化。"""
    prob = np.zeros(len(y)); rows = []
    for f, (tr, va) in enumerate(GroupKFold(folds).split(xw, y, grp)):
        t0 = time.time()
        prob[va] = train_eval(model_name, xw, xs, y, tr, va, 42 + f)
        rows.append({"fold": f, "balanced_accuracy": float(balanced_accuracy_score(y[va], prob[va] >= 0.5)),
                     "roc_auc": float(roc_auc_score(y[va], prob[va])), "sec": round(time.time() - t0, 1)})
    pred = (prob >= 0.5).astype(int)
    subs = {int(s): float(balanced_accuracy_score(y[grp == s], pred[grp == s])) for s in np.unique(grp)}
    name = f"{model_name}{'_+subject_std' if std else '_no_std'}"
    r = {"setting": name, "n_windows": int(len(y)),
         "window_ba": float(balanced_accuracy_score(y, pred)),
         "window_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
         "window_auc": float(roc_auc_score(y, prob)),
         "fold_ba_mean": float(np.mean([x["balanced_accuracy"] for x in rows])),
         "fold_ba_std": float(np.std([x["balanced_accuracy"] for x in rows])),
         "fold_ba_min": float(np.min([x["balanced_accuracy"] for x in rows])),
         "subject_ba_mean": float(np.mean(list(subs.values()))),
         "n_subjects_above_0.60": int(sum(v > 0.60 for v in subs.values())),
         "per_subject": {str(k): round(v, 4) for k, v in sorted(subs.items())}, "folds": rows}
    print(f"{name:28s} 窗口BA={r['window_ba']:.4f} AUC={r['window_auc']:.4f} "
          f"折间{r['fold_ba_mean']:.4f}±{r['fold_ba_std']:.4f} min折={r['fold_ba_min']:.3f} "
          f">0.6被试={r['n_subjects_above_0.60']}/{len(np.unique(grp))}", flush=True)
    return r


def main():
    x_raw, y_all, grp_all = load_development(DATA)
    xw_all = normalize_windows(x_raw)
    feats_all = band_power_feats(xw_all)

    # ── 折安全校准：只在静息段最靠前的 30 窗上拟合 μ/σ（协议 v1.3 方案 A）──
    feats_cal, calib_mask, calib_info = calibrate_by_subject(
        feats_all, y_all, grp_all, n_calib=DEFAULT_N_CALIB)

    print("=== 校准窗口预算（协议 §1.1 / R5）===")
    print(describe(grp_all, y_all, calib_mask))
    print()

    # ── 评估集 = 全部非校准窗（R2：校准窗不计分）──
    k = ~calib_mask
    xw, y, grp = xw_all[k], y_all[k], grp_all[k]
    f_raw, f_cal = feats_all[k], feats_cal[k]
    n_subj = int(len(np.unique(grp)))

    out = {"protocol": ("original set with calibration windows excluded "
                        "(91 rest → 61 rest + 31 focus per subject), "
                        "8-fold GroupKFold, epochs=15"),
           "calibration_protocol": "CALIBRATION_PROTOCOL.md v1.3 方案 A: 静息段前 30 窗校准, 折安全拟合",
           "device": str(DEVICE),
           "n_windows": int(len(y)),
           "n_subjects": n_subj,
           "calibration": calib_info,
           "results": []}
    print(f"device={DEVICE}  评估集: {xw.shape}  被试={n_subj}\n", flush=True)
    for model_name in ("gen5", "teacherB"):
        for std in (False, True):
            out["results"].append(run(model_name, std, xw, f_cal if std else f_raw, y, grp))
    Path("artifacts/probes/probe5_2x2.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n已写入 probe5_2x2.json")


if __name__ == "__main__":
    main()
