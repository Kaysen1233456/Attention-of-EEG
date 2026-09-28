"""Audit Fold-2 subject/domain separation in waveform and AAMP feature spaces."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "scripts"))

from attention_model.config import AttentionConfig
from attention_model.data import EEGDataset
from train import build_model


def _load_fold_arrays(root: Path, fold_index: int):
    manifest = json.loads((root / "configs/development_folds.json").read_text())
    fold = manifest["folds"][fold_index]
    data_dir = root / "data/processed/ear_saad"
    arrays = {}
    for split in ("train", "val"):
        arrays[f"{split}_waveforms"] = np.load(data_dir / f"{split}_waveforms.npy")
        arrays[f"{split}_labels"] = np.load(data_dir / f"{split}_labels.npy")
        arrays[f"{split}_subjects"] = np.load(data_dir / f"{split}_subjects.npy")
        arrays[f"{split}_trials"] = np.load(data_dir / f"{split}_trials.npy")
    combined = {
        key: np.concatenate([arrays[f"train_{key}"], arrays[f"val_{key}"]])
        for key in ("waveforms", "labels", "subjects", "trials")
    }
    train_mask = np.isin(combined["subjects"], fold["train_subjects"])
    val_mask = np.isin(combined["subjects"], fold["val_subjects"])
    return fold, {k: v[train_mask] for k, v in combined.items()}, {
        k: v[val_mask] for k, v in combined.items()
    }


def _normalize(waveforms, train_waveforms):
    mean = train_waveforms.mean(axis=(0, 2), keepdims=True)
    std = train_waveforms.std(axis=(0, 2), keepdims=True) + 1e-8
    return np.clip((waveforms - mean) / std, -8.0, 8.0)


def _subject_centroids(features, subjects):
    result = {}
    for subject in sorted(np.unique(subjects).tolist()):
        values = features[subjects == subject]
        result[str(int(subject))] = {
            "n_samples": int(len(values)),
            "centroid": values.mean(axis=0).tolist(),
            "within_subject_rms": float(
                np.sqrt(np.mean(np.square(values - values.mean(axis=0))))
            ),
        }
    return result


def _centroid_distances(centroids):
    ids = sorted(centroids, key=int)
    matrix = np.zeros((len(ids), len(ids)), dtype=np.float64)
    for i, left in enumerate(ids):
        for j, right in enumerate(ids):
            matrix[i, j] = np.linalg.norm(
                np.asarray(centroids[left]["centroid"])
                - np.asarray(centroids[right]["centroid"])
            )
    return ids, matrix


def _plot(points, subjects, train_subjects, val_subjects, title, path):
    fig, ax = plt.subplots(figsize=(10, 7))
    for subject in sorted(np.unique(subjects).tolist()):
        mask = subjects == subject
        split = "train" if int(subject) in train_subjects else "val"
        marker = "o" if split == "train" else "X"
        ax.scatter(
            points[mask, 0], points[mask, 1], s=9, alpha=0.35,
            marker=marker, label=f"S{int(subject)} ({split})",
        )
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_report(root: Path, output: Path) -> dict:
    fold, train, val = _load_fold_arrays(root, 2)
    train_w = _normalize(train["waveforms"], train["waveforms"]).astype(np.float32)
    val_w = _normalize(val["waveforms"], train["waveforms"]).astype(np.float32)
    waveforms = np.concatenate([train_w, val_w])
    subjects = np.concatenate([train["subjects"], val["subjects"]])
    split_subjects = {"train": fold["train_subjects"], "val": fold["val_subjects"]}

    raw_flat = waveforms.reshape(len(waveforms), -1)
    raw_pca = PCA(n_components=2, random_state=42).fit_transform(raw_flat)
    raw_model = PCA(n_components=16, random_state=42)
    raw_16 = raw_model.fit_transform(raw_flat)

    config = AttentionConfig.from_yaml(root / "configs/teacher_aamp_frozen_ear_saad.yaml")
    model = build_model(config).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    features = []
    with torch.no_grad():
        for start in range(0, len(waveforms), 128):
            batch = torch.from_numpy(waveforms[start:start + 128]).to(device)
            encoded = model.encoder.encode(batch)
            features.append(encoded.mean(dim=(1, 2)).cpu().numpy())
    features = np.concatenate(features)
    feature_pca_model = PCA(n_components=16, random_state=42)
    feature_16 = feature_pca_model.fit_transform(features)
    feature_pca = feature_16[:, :2]

    rng = np.random.default_rng(42)
    sample_size = min(1800, len(features))
    sample = np.sort(rng.choice(len(features), size=sample_size, replace=False))
    tsne = TSNE(
        n_components=2, perplexity=35, init="pca", learning_rate="auto",
        random_state=42,
    ).fit_transform(feature_16[sample])
    all_subjects = sorted(np.unique(subjects).tolist())
    output.mkdir(parents=True, exist_ok=True)
    _plot(
        raw_pca, subjects, fold["train_subjects"], fold["val_subjects"],
        "Fold 2 normalized waveform PCA", output / "fold2_waveform_pca.png",
    )
    _plot(
        feature_pca, subjects, fold["train_subjects"], fold["val_subjects"],
        "Fold 2 AAMP encoder feature PCA", output / "fold2_aamp_feature_pca.png",
    )
    _plot(
        tsne, subjects[sample], fold["train_subjects"], fold["val_subjects"],
        "Fold 2 AAMP encoder feature t-SNE", output / "fold2_aamp_feature_tsne.png",
    )
    _, raw_dist = _centroid_distances(
        _subject_centroids(raw_16, subjects)
    )
    _, feature_dist = _centroid_distances(
        _subject_centroids(feature_16, subjects)
    )
    report = {
        "fold": fold,
        "device": str(device),
        "test_used": False,
        "n_samples": int(len(subjects)),
        "subjects": all_subjects,
        "feature_definition": "mean pooled output of the pretrained AAMP encoder before the classifier",
        "waveform_pca_explained_variance_ratio": raw_model.explained_variance_ratio_.tolist(),
        "aamp_feature_pca_explained_variance_ratio": feature_pca_model.explained_variance_ratio_.tolist(),
        "waveform_subject_centroids": _subject_centroids(raw_16, subjects),
        "aamp_feature_subject_centroids": _subject_centroids(feature_16, subjects),
        "waveform_centroid_distance_subject_order": all_subjects,
        "waveform_centroid_distances": raw_dist.tolist(),
        "aamp_feature_centroid_distance_subject_order": all_subjects,
        "aamp_feature_centroid_distances": feature_dist.tolist(),
        "plots": [
            "fold2_waveform_pca.png",
            "fold2_aamp_feature_pca.png",
            "fold2_aamp_feature_tsne.png",
        ],
    }
    (output / "fold2_feature_space_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def render(report: dict) -> str:
    ids = report["aamp_feature_centroid_distance_subject_order"]
    matrix = np.asarray(report["aamp_feature_centroid_distances"])
    val_ids = set(report["fold"]["val_subjects"])
    rows = [
        "# Fold 2 特征空间审计",
        "",
        "本报告仅使用 development train/validation，未读取 test。特征为 AAMP 预训练 encoder 的 mean pooled 表征。",
        "",
        "## PCA 方差解释率",
        f"- 归一化波形 PCA 前两维：`{sum(report['waveform_pca_explained_variance_ratio'][:2]):.4f}`",
        f"- AAMP 特征 PCA 前两维：`{sum(report['aamp_feature_pca_explained_variance_ratio'][:2]):.4f}`",
        "",
        "## AAMP 特征空间被试质心距离",
        "",
        "| 被试 | 训练/验证 | 组内 RMS |",
        "|---:|---|---:|",
    ]
    for subject in ids:
        item = report["aamp_feature_subject_centroids"][str(subject)]
        rows.append(
            f"| {subject} | {'验证' if subject in val_ids else '训练'} | "
            f"{item['within_subject_rms']:.6f} |"
        )
    rows += [
        "",
        "验证被试到训练被试的 AAMP 质心距离：",
        "",
        "| 验证被试 | 最近训练被试 | 最近距离 |",
        "|---:|---:|---:|",
    ]
    for subject in sorted(val_ids):
        i = ids.index(subject)
        train_indices = [j for j, item in enumerate(ids) if item not in val_ids]
        j = min(train_indices, key=lambda k: matrix[i, k])
        rows.append(f"| {subject} | {ids[j]} | {matrix[i, j]:.6f} |")
    rows += [
        "",
        "## 文件",
        "- `fold2_waveform_pca.png`：归一化原始波形 PCA",
        "- `fold2_aamp_feature_pca.png`：AAMP encoder 特征 PCA",
        "- `fold2_aamp_feature_tsne.png`：AAMP encoder 特征 t-SNE",
        "",
        "## 判定边界",
        "- PCA/t-SNE 只能证明分布是否分离，不能单独证明标签错误。",
        "- 若 Fold 2 被试在 AAMP 特征空间远离训练被试，支持被试域差异假设。",
        "- 若没有明显分离但分类仍接近随机，应优先怀疑任务表征不足、方向不稳定或监督信号弱。",
    ]
    return "\n".join(rows) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("artifacts/fold2_feature_space_audit"))
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else args.repo_root / args.output
    report = build_report(args.repo_root.resolve(), output)
    (output / "fold2_feature_space_audit.md").write_text(
        render(report), encoding="utf-8"
    )
    print(f"Report: {output / 'fold2_feature_space_audit.md'}")
    print(f"Plots: {', '.join(report['plots'])}")


if __name__ == "__main__":
    main()
