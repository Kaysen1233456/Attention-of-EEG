"""
频谱频带审计（只读，不产生任何项目写入）

回答的问题：
  1. EDF 头里记录的预滤波设置是什么？官方 README 说的「30 Hz 高通」是否属实？
  2. 40 维频谱特征（8 通道 x 5 频带）里，是否存在「死维」（近似常数、无信息）？
  3. 各频带对被试内「静息 vs 心算」的效应量分别是多少？跨被试是否一致？
  4. 各频带单独的可分性（用平衡准确率 BA，避免类别不平衡陷阱）是多少？

背景：开发集审计显示被试间频谱差异大于被试内任务效应，模型卡在 BA ~0.70。
      本脚本用于判断「40 维特征里各频带的相对信息量」，为特征裁剪/加权提供依据。

⚠️ 局限（必须随结果一起引用）：
  - 本地只有 Subject00 / Subject01 两个被试的原始 EDF，n=2，只能产生「假设」不是「结论」。
  - 第 4 节是【被试内】最近质心分类，rest/focus 来自同一被试的连续记录，
    没有跨被试域偏移，BA 会明显高于跨被试结果，不能与跨被试 BA 直接比较。
  - 类别不平衡：rest 91 窗 vs focus 31 窗，多数类准确率 0.746。
    因此本脚本一律使用 BA，绝不用 accuracy。

用法：
    python scripts/audit_spectral_bands.py
    python scripts/audit_spectral_bands.py --data-dir data/raw/eegmat
"""

import argparse
import pathlib

import numpy as np

# EDF 里的 8 个重点电极（与 preprocess_mental_arithmetic.py 的选择保持一致）
EEG8 = ["EEG Fz", "EEG F3", "EEG F4", "EEG T3", "EEG T4", "EEG Pz", "EEG P3", "EEG P4"]
# 与 preprocess_mental_arithmetic.py / cv_spectral_cnn.py 保持一致的 5 个频带
BANDS = [
    ("delta", 1, 4),
    ("theta", 4, 8),
    ("alpha", 8, 13),
    ("beta", 13, 30),
    ("gamma", 30, 45),
]
FS = 500.0          # 原始 EDF 采样率
TARGET_FS = 250.0   # 预处理后的采样率（仅记录，本审计在原始 500 Hz 上做）
WINDOW_S = 2.0


def read_edf(path):
    """最小 EDF 读取器（仅标准 EDF，非 EDF+），返回 (signals, fs, labels, prefiltering, n_records, record_dur)"""
    b = path.read_bytes()
    n_sig = int(b[252:256])
    n_rec = int(b[236:244])
    rec_dur = float(b[244:252])

    def field(offset, width, count):
        return [
            b[offset + i * width: offset + (i + 1) * width].decode("latin1").strip()
            for i in range(count)
        ]

    labels = field(256, 16, n_sig)
    p_min = [float(x) for x in field(256 + n_sig * 104, 8, n_sig)]
    p_max = [float(x) for x in field(256 + n_sig * 112, 8, n_sig)]
    d_min = [float(x) for x in field(256 + n_sig * 120, 8, n_sig)]
    d_max = [float(x) for x in field(256 + n_sig * 128, 8, n_sig)]
    prefilter = field(256 + n_sig * 136, 80, n_sig)
    samples_per_rec = [int(x) for x in field(256 + n_sig * 216, 8, n_sig)]

    header_bytes = 256 + n_sig * 256
    raw = np.frombuffer(b[header_bytes:], dtype="<i2")
    raw = raw[: sum(samples_per_rec) * n_rec].reshape(n_rec, sum(samples_per_rec))

    signals = {}
    cursor = 0
    for i in range(n_sig):
        block = raw[:, cursor:cursor + samples_per_rec[i]].ravel().astype(np.float64)
        cursor += samples_per_rec[i]
        scale = (p_max[i] - p_min[i]) / (d_max[i] - d_min[i])
        signals[labels[i]] = (block - d_min[i]) * scale + p_min[i]

    fs = samples_per_rec[0] / rec_dur
    return signals, fs, labels, prefilter, n_rec, rec_dur


def build_features(signals, fs, window_s=WINDOW_S):
    """返回 (n_windows, 40)，布局 ch0_delta..ch0_gamma, ch1_delta.., ..., ch7_gamma

    与模型内部一致：log(band_power / total_power_1_to_45Hz + 1e-8)
    """
    n = int(round(window_s * fs))
    freqs = np.fft.rfftfreq(n, 1 / fs)
    masks = [(freqs >= lo) & (freqs < hi) for _, lo, hi in BANDS]
    taper = np.hanning(n)

    n_win = min(len(signals[ch]) // n for ch in EEG8)
    feats = np.zeros((n_win, 40))
    for ci, ch in enumerate(EEG8):
        x = signals[ch] - signals[ch].mean()
        for k in range(n_win):
            seg = x[k * n:(k + 1) * n] * taper
            power = np.abs(np.fft.rfft(seg)) ** 2
            band_power = np.array([power[m].sum() for m in masks])
            rel = band_power / (band_power.sum() + 1e-12)
            feats[k, ci * 5:(ci + 1) * 5] = np.log(rel + 1e-8)
    return feats


def balanced_accuracy(score, y):
    """对 score 以 0 为阈值时的平衡准确率（正类 y==1）"""
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    return 0.5 * ((pos > 0).mean() + (neg <= 0).mean())


def band_power_share(signals, fs):
    """Welch 平均（4 s 段、50% 重叠），返回各频带占 1-45 Hz 的功率百分比"""
    n = int(4 * fs)
    step = n // 2
    taper = np.hanning(n)
    freqs = np.fft.rfftfreq(n, 1 / fs)
    masks = [(freqs >= lo) & (freqs < hi) for _, lo, hi in BANDS]

    share = np.zeros(len(BANDS))
    for ch in EEG8:
        x = signals[ch] - signals[ch].mean()
        n_seg = max(1, (len(x) - n) // step)
        psd = np.zeros(n // 2 + 1)
        for k in range(n_seg):
            seg = x[k * step:k * step + n] * taper
            psd += np.abs(np.fft.rfft(seg)) ** 2
        psd /= n_seg
        bp = np.array([psd[m].sum() for m in masks])
        share += bp / bp.sum() * 100
    return share / len(EEG8)


def main():
    ap = argparse.ArgumentParser(description="频谱频带审计（只读）")
    ap.add_argument("--data-dir", default="data/raw/eegmat",
                    help="eegmat 原始 EDF 目录")
    ap.add_argument("--subjects", nargs="*", default=None,
                    help="指定被试编号，如 00 01；默认自动发现全部")
    args = ap.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    if not data_dir.is_dir():
        raise SystemExit(f"目录不存在: {data_dir}")

    if args.subjects:
        subjects = list(args.subjects)
    else:
        subjects = sorted({
            p.name.split("_")[0].replace("Subject", "")
            for p in data_dir.glob("Subject*_1.edf")
        })

    print("=" * 78)
    print("频谱频带审计  (scripts/audit_spectral_bands.py)")
    print(f"数据目录: {data_dir.resolve()}")
    print(f"被试: {subjects}   n={len(subjects)}")
    print("=" * 78)
    print()
    print("⚠️ n 很小，以下结果只能作为【假设】，必须用全量嵌套 CV 验证。")
    print()

    # ---------- 1. EDF 头信息 ----------
    print("-" * 78)
    print("1. EDF 头信息（含预滤波设置）")
    print("-" * 78)
    for s in subjects:
        for suf in ("1", "2"):
            p = data_dir / f"Subject{s}_{suf}.edf"
            if not p.exists():
                print(f"  缺失: {p.name}")
                continue
            _, fs, _, prefilter, n_rec, rec_dur = read_edf(p)
            total = n_rec * rec_dur
            tag = "rest " if suf == "1" else "focus"
            uniq = sorted({x for x in prefilter if x})
            print(f"  Subject{s}_{suf}.edf  [{tag}]  {total:.0f}s  fs={fs:.0f}Hz  "
                  f"windows(2s)={int(total // 2)}")
            for u in uniq:
                print(f"      prefiltering: {u}")
            if not uniq:
                print("      prefiltering: (空)")

    # ---------- 2. 频带功率占比 ----------
    print()
    print("-" * 78)
    print("2. 各频带功率占比（占 1-45 Hz，Welch 4s/50%）")
    print("-" * 78)
    print(f"  {'被试':10s} {'段':6s} " + " ".join(f"{bn:>7s}" for bn, _, _ in BANDS))
    for s in subjects:
        for suf in ("1", "2"):
            p = data_dir / f"Subject{s}_{suf}.edf"
            if not p.exists():
                continue
            signals, fs = read_edf(p)[:2]
            share = band_power_share(signals, fs)
            tag = "rest" if suf == "1" else "focus"
            print(f"  Subject{s:4s} {tag:6s} " + " ".join(f"{v:6.1f}%" for v in share))
    print()
    print("  => gamma(30-45Hz) 仅占 ~1-4%，但 log(相对功率) 会把它放大到与其他频带同量级。")

    # ---------- 3. 特征矩阵与死维检查 ----------
    print()
    print("-" * 78)
    print("3. 40 维特征：逐维方差（检查死维）")
    print("-" * 78)
    feats = {}
    for s in subjects:
        rest = data_dir / f"Subject{s}_1.edf"
        focus = data_dir / f"Subject{s}_2.edf"
        if not (rest.exists() and focus.exists()):
            continue
        feats[s] = {
            "1": build_features(read_edf(rest)[0], read_edf(rest)[1]),
            "2": build_features(read_edf(focus)[0], read_edf(focus)[1]),
        }
        print(f"  Subject{s}: rest {feats[s]['1'].shape}   focus {feats[s]['2'].shape}")

    if feats:
        all_feats = np.vstack([feats[s][k] for s in feats for k in ("1", "2")])
        sd = all_feats.std(axis=0)
        print()
        print(f"  整体 std 范围: {sd.min():.4f} .. {sd.max():.4f}   min/max={sd.min()/sd.max():.3f}")
        for bi, (bn, _, _) in enumerate(BANDS):
            v = sd[bi::5]
            print(f"  {bn:6s} 8 通道 std: " + " ".join(f"{x:.2f}" for x in v)
                  + f"   mean={v.mean():.3f}")
        dead = sd < 0.01 * sd.max()
        print(f"  近似常数维 (std < 1% max): {int(dead.sum())} / 40  -> {list(np.where(dead)[0])}")
        print("  => 无死维。log-relative-power 让弱频带也保留了同等量级方差。")

    # ---------- 4. 各频带效应量 ----------
    print()
    print("-" * 78)
    print("4. 各频带效应量 = |E[focus]-E[rest]| / pooled_std  （逐通道，无量纲）")
    print("-" * 78)
    print(f"  通道顺序: {' '.join(c.replace('EEG ', '') for c in EEG8)}")
    effect = {}
    for s in feats:
        X1, X2 = feats[s]["1"], feats[s]["2"]
        diff = X2.mean(axis=0) - X1.mean(axis=0)
        pooled = np.sqrt((X1.var(axis=0) + X2.var(axis=0)) / 2) + 1e-12
        eff = np.abs(diff) / pooled
        effect[s] = eff
        print(f"  Subject{s}")
        for bi, (bn, _, _) in enumerate(BANDS):
            v = eff[bi::5]
            print(f"    {bn:6s} " + " ".join(f"{x:5.2f}" for x in v)
                  + f"   | mean={v.mean():.2f}")

    # ---------- 5. 跨被试一致性 ----------
    if len(effect) >= 2:
        print()
        print("-" * 78)
        print("5. 跨被试一致性（效应量是否稳定）")
        print("-" * 78)
        keys = list(effect)
        for bi, (bn, _, _) in enumerate(BANDS):
            vals = [effect[k][bi::5].mean() for k in keys]
            spread = max(vals) - min(vals)
            flag = "  <== 跨被试不稳定" if spread > 0.30 else ""
            print(f"  {bn:6s} " + "  ".join(f"{k}={v:.2f}" for k, v in zip(keys, vals))
                  + f"   极差={spread:.2f}{flag}")

    # ---------- 6. 各频带被试内可分性（BA） ----------
    print()
    print("-" * 78)
    print("6. 各频带【被试内】可分性（最近质心，BA）")
    print("-" * 78)
    print("  注：rest/focus 同属一个被试的连续记录，无跨被试域偏移，")
    print("      此列 BA 不能与跨被试 BA 直接比较，只用于比较频带间相对信息量。")
    print("  注：一律用 BA；本数据 rest:focus = 91:31，多数类准确率 0.746，accuracy 无意义。")
    print()
    for s in feats:
        X1, X2 = feats[s]["1"], feats[s]["2"]
        X = np.vstack([X1, X2])
        y = np.r_[np.zeros(len(X1)), np.ones(len(X2))]
        print(f"  Subject{s}   rest={len(X1)}  focus={len(X2)}")
        for bi, (bn, _, _) in enumerate(BANDS):
            cols = [bi + 5 * c for c in range(8)]
            Xb = X[:, cols]
            c0, c1 = Xb[y == 0].mean(axis=0), Xb[y == 1].mean(axis=0)
            w = c1 - c0
            mid = (c0 + c1) / 2
            ba8 = balanced_accuracy((Xb - mid) @ w, y)
            per_ch = []
            for ci in range(8):
                v = Xb[:, ci]
                a, b_ = v[y == 0].mean(), v[y == 1].mean()
                per_ch.append(balanced_accuracy(v - (a + b_) / 2, y))
            print(f"    {bn:6s} BA(8ch)={ba8:.3f}   逐通道: "
                  + " ".join(f"{p:.2f}" for p in per_ch)
                  + f"   | 均值={np.mean(per_ch):.3f}")
        c0, c1 = X[y == 0].mean(axis=0), X[y == 1].mean(axis=0)
        w = c1 - c0
        print(f"    {'ALL40':6s} BA(40d)={balanced_accuracy((X - (c0 + c1) / 2) @ w, y):.3f}"
              "   <== 被试内上限参考")
        print()

    print("=" * 78)
    print("结论使用须知：")
    print("  - 本审计 n 极小，所有频带结论都是【待验证假设】，不得直接写进协议。")
    print("  - gamma 的判别力可能来自 30-45Hz 的 EMG（肌肉紧张）污染，而非神经 gamma。")
    print("    心算任务伴随皱眉/咬紧牙关，EMG 会落在该频段。验证方法见 experiments_log。")
    print("=" * 78)


if __name__ == "__main__":
    main()
