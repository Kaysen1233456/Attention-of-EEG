"""
被试级校准的「折安全」工具 —— 按 CALIBRATION_PROTOCOL.md v1.3 方案 A 实现。

═══════════════════════════════════════════════════════════════════
为什么需要这个模块（登记册 C10）
═══════════════════════════════════════════════════════════════════
原探针的实现顺序是错的：

    feats_ss = subject_standardize(feats, grp)   # ← 用【全部】窗口拟合 μ/σ
    for tr, va in GroupKFold(...).split(...):    # ← 然后才划分
        train / eval

被评估的那个被试，它的 μ/σ 是用**它自己全部窗口**算出来的，
而评估也在同一批窗口上做 → **评估窗参与了归一化统计量的估计**。

等价于：用考卷答案算"个性化参数"，再用这份考卷打分。
→ 收益被系统性高估。这就是原报的「+3.3 ~ +7.2 分」不可引用的原因。

═══════════════════════════════════════════════════════════════════
正确的顺序（本模块强制）
═══════════════════════════════════════════════════════════════════
    ① 每个被试只用【静息段最靠前的 30 个窗口】拟合 μ/σ
    ② 把 μ/σ 应用到该被试的**所有**窗口
    ③ 训练与评估都**只使用非校准窗**（R2：校准窗不计分）

本模块把 ①②③ 封成一个函数 `calibrate_by_subject()`，
并把「哪些窗口是校准窗」返回给调用方，**强制**调用方把它们从评分里剔除。

另外提供 `matched_eval_indices()`：在**非校准**静息窗里做时长匹配采样，
使「时长匹配」与「30 窗校准」可以共存（原始 91 窗足够：30 校准 + 31 匹配 + 余量）。

═══════════════════════════════════════════════════════════════════
用法
═══════════════════════════════════════════════════════════════════
    from _subject_calibration import calibrate_by_subject, matched_eval_indices

    feats_cal, calib_mask, info = calibrate_by_subject(feats_raw, y, grp, n_calib=30)

    # 原始集：评估 = 全部非校准窗
    eval_mask = ~calib_mask

    # 时长匹配集：在非校准静息窗里采样，与 focus 对齐
    idx = matched_eval_indices(y, grp, calib_mask, rng)

    # 然后 GroupKFold 只在 idx 上做
"""

import numpy as np

DEFAULT_N_CALIB = 30
DEFAULT_EPSILON = 1e-8


def subject_standardize(*args, **kwargs):
    """【已移除】旧接口。调用即报错，防止有人再写出折前拟合的代码。"""
    raise RuntimeError(
        "subject_standardize() 已被移除：它在 GroupKFold 划分【之前】用全部窗口拟合 μ/σ，\n"
        "会让评估窗参与统计量估计（登记册 C10，原报 +3.3~+7.2 分因此不可引用）。\n"
        "请改用 calibrate_by_subject()，它只在校准窗上拟合并返回需剔除的校准窗掩码。"
    )


def _subject_list(grp):
    return np.unique(grp)


def rest_calibration_mask(y, grp, n_calib=DEFAULT_N_CALIB):
    """返回布尔掩码：每个被试【静息段（label==0）时间序最靠前的 n_calib 个窗口】。

    窗口按数组下标顺序即为时间顺序（数据按 被试→rest段→focus段 排列）。

    若某被试静息窗不足 n_calib，抛出 ValueError 并给出该被试名与可用窗数 ——
    不允许静默缩减（那会违反 CALIBRATION_PROTOCOL.md §1.3 最小样本数）。
    """
    y = np.asarray(y)
    grp = np.asarray(grp)
    mask = np.zeros(len(y), dtype=bool)
    insufficient = []
    for s in _subject_list(grp):
        idx = np.flatnonzero((grp == s) & (y == 0))
        if len(idx) < n_calib:
            insufficient.append((int(s), int(len(idx))))
            continue
        mask[idx[:n_calib]] = True
    if insufficient:
        detail = "；".join("被试组 %d 仅有 %d 个静息窗" % (s, n) for s, n in insufficient)
        raise ValueError(
            "静息窗不足以执行固定 %d 窗校准（见 CALIBRATION_PROTOCOL.md §1.3）：%s"
            % (n_calib, detail)
        )
    return mask


def channel_groups(n_dim, n_channels):
    """按**通道**分组特征列，供 `groups=` 使用（组内共享一个 μ/σ 标量）。

    约定特征列布局为**频带优先**：`np.concatenate([band0(N,8), band1(N,8), ...], axis=1)`。
    于是列 j 的通道 = j % n_channels，通道 c 的列 = [c, c+n, c+2n, ...]。

    ⚠️ 这正是原 `block=len(BANDS)` 想表达、但实际没做到的事：
    原来的 block 是**连续**切片（0-4 / 5-9 / ...），既不是通道，也没有池化效果，
    因为 `mean(axis=0)` 本来就是逐列计算 —— 分块只影响"赋值给哪一段"。
    结果就是 N2（被试+通道级）与 N1（被试级）**数学上完全相同**，是静默重复行。
    """
    if n_dim % n_channels:
        raise ValueError("特征维数 %d 不能被通道数 %d 整除" % (n_dim, n_channels))
    return [np.arange(c, n_dim, n_channels) for c in range(n_channels)]


def _normalize_groups(n_dim, groups):
    """把 groups 归一成 list[np.ndarray]；None → 每维自成一组（逐维统计，无池化）。"""
    if groups is None:
        return [np.array([j]) for j in range(n_dim)]
    out = []
    for g in groups:
        idx = np.asarray(g, dtype=int)
        if idx.ndim != 1 or idx.size == 0:
            raise ValueError("groups 的每个元素必须是非空一维索引数组")
        out.append(idx)
    seen = np.concatenate(out)
    if len(seen) != n_dim or len(np.unique(seen)) != n_dim:
        raise ValueError("groups 必须不重不漏地覆盖全部 %d 个特征维" % n_dim)
    return out


def fit_subject_stats(feats, grp, calib_mask, groups=None, epsilon=DEFAULT_EPSILON):
    """**只在校准窗上**为每个被试拟合 μ/σ。

    返回 {subject_id: (mean, std)}，mean/std 形状为 (n_dim,)。

    groups=None  → 逐维统计（每个特征维独立的 μ/σ）
    groups=[...] → **组内池化**：同一组的列共享一个 μ/σ 标量
                   （如 `channel_groups(40, 8)` = 每个通道的 5 个频带共用一个标量）
    """
    feats = np.asarray(feats, dtype=np.float64)
    grp = np.asarray(grp)
    calib_mask = np.asarray(calib_mask, dtype=bool)
    gidx = _normalize_groups(feats.shape[1], groups)

    stats = {}
    for s in _subject_list(grp):
        m = (grp == s) & calib_mask
        if m.sum() == 0:
            raise ValueError("被试组 %d 没有任何校准窗，无法拟合统计量" % int(s))
        sub = feats[m]
        mu = np.empty(feats.shape[1])
        sd = np.empty(feats.shape[1])
        for idx in gidx:
            pooled = sub[:, idx]
            mu[idx] = pooled.mean()
            sd[idx] = pooled.std()
        stats[int(s)] = (mu, sd + epsilon)
    return stats


def apply_subject_stats(feats, grp, stats, out_dtype=None):
    """把统计量应用到该被试的**所有**窗口（含校准窗本身）。

    内部按 float64 计算；若给出 `out_dtype` 则把结果转回该 dtype
    （torch 模型要求 float32，调用方传入 float64 会触发 Double/Float 不匹配）。
    """
    feats = np.asarray(feats)
    dtype = out_dtype if out_dtype is not None else feats.dtype
    work = feats.astype(np.float64, copy=False)
    grp = np.asarray(grp)
    out = np.empty_like(work)
    for s in _subject_list(grp):
        m = grp == s
        mu, sd = stats[int(s)]
        out[m] = (work[m] - mu) / sd
    return out.astype(dtype, copy=False)


def calibrate_by_subject(feats, y, grp, n_calib=DEFAULT_N_CALIB, groups=None,
                         epsilon=DEFAULT_EPSILON):
    """一步完成「选校准窗 → 只在校准窗拟合 → 应用」。

    返回 (feats_calibrated, calib_mask, info)

    ⚠️ 调用方**必须**把 `calib_mask` 为 True 的窗口从训练与评分中剔除（R2）。
    `info` 可直接写进结果 JSON 的 provenance 字段（对应协议 R3）。
    """
    y = np.asarray(y)
    grp = np.asarray(grp)
    feats_arr = np.asarray(feats)
    calib_mask = rest_calibration_mask(y, grp, n_calib)
    stats = fit_subject_stats(feats_arr, grp, calib_mask, groups=groups, epsilon=epsilon)
    # 保持输入 dtype：float32 输入 → float32 输出（torch 模型依赖这一点）
    feats_cal = apply_subject_stats(feats_arr, grp, stats, out_dtype=feats_arr.dtype)

    gidx = _normalize_groups(feats_arr.shape[1], groups)
    grouping = ("per_dimension (%d independent)" % len(gidx) if groups is None
                else "pooled over %d groups of size %d" % (len(gidx), len(gidx[0])))

    rest_counts = [int(((grp == s) & (y == 0)).sum()) for s in _subject_list(grp)]
    info = {
        "calibration_seconds": int(n_calib * 2),
        "n_calibration_windows": int(n_calib),
        "n_calibration_windows_total": int(calib_mask.sum()),
        "n_subjects": int(len(_subject_list(grp))),
        "min_rest_windows_per_subject": int(min(rest_counts)),
        "feature_grouping": grouping,
        "epsilon": epsilon,
        "calibration_scope": "within_session",
        "source": "pretask_rest_prefix_unlabeled",
        "whitening": False,
        "transductive": True,
        "scored_calibration_windows": False,
        "fitted_on": "calibration windows only (fold-safe)",
    }
    return feats_cal, calib_mask, info


def matched_eval_indices(y, grp, calib_mask, rng, n_rest=None, strict=True):
    """时长匹配采样：每个被试在**非校准**静息窗里随机取 n_rest 个，加上**全部** focus 窗。

    这样既保留「rest : focus = 1:1」的匹配设计（登记册 D2），
    又不让校准窗进入评估集（R2）。

    n_rest 默认 = 该被试的 focus 窗数。

    返回 (indices, skipped_subjects)

    ⚠️ Subject31 扣除 30 窗校准后仅剩 10 个静息窗，**无法**匹配 31 个 focus 窗。
       `strict=True` 时抛错；`strict=False` 时跳过该被试并在 `skipped_subjects` 中返回，
       由调用方按 R5 的要求**显式报告**（含/不含各一行）。
    """
    y = np.asarray(y)
    grp = np.asarray(grp)
    calib_mask = np.asarray(calib_mask, dtype=bool)
    keep, skipped = [], []
    for s in _subject_list(grp):
        rest_pool = np.flatnonzero((grp == s) & (y == 0) & ~calib_mask)
        focus = np.flatnonzero((grp == s) & (y == 1))
        k = len(focus) if n_rest is None else int(n_rest)
        if len(rest_pool) < k:
            msg = ("被试组 %d 扣除校准窗后仅剩 %d 个静息窗，不足以匹配 %d 个 focus 窗"
                   % (int(s), len(rest_pool), k))
            if strict:
                raise ValueError(msg)
            skipped.append({"subject_group": int(s), "rest_left": int(len(rest_pool)),
                            "focus": int(k), "reason": msg})
            continue
        keep.append(np.r_[rng.choice(rest_pool, size=k, replace=False), focus])
    return np.sort(np.concatenate(keep)), skipped


def describe(grp, y, calib_mask):
    """打印每个被试的窗口预算（校准 / 可评估静息 / focus），便于核对 R5。"""
    lines = []
    for s in _subject_list(grp):
        m = grp == s
        n_rest = int((m & (y == 0)).sum())
        n_focus = int((m & (y == 1)).sum())
        n_calib = int((m & calib_mask).sum())
        n_rest_left = n_rest - n_calib
        flag = "  <== 触发 R5 rest_after_cal_low" if n_rest_left < 15 else ""
        lines.append("  被试组 %-3d 静息 %3d → 校准 %2d → 可评估静息 %3d ｜ focus %3d%s"
                     % (int(s), n_rest, n_calib, n_rest_left, n_focus, flag))
    return "\n".join(lines)
