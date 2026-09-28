"""Cross-validate a subject-disjoint time/statistical/spectral fusion model."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import welch
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


BANDS = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45))


def features(x):
    # Per-channel features keep subject-specific scale out of the classifier.
    mean = x.mean(axis=-1)
    std = x.std(axis=-1) + 1e-8
    rms = np.sqrt(np.mean(x * x, axis=-1)) + 1e-8
    frequency, power = welch(x, fs=250, nperseg=250, axis=-1)
    total_mask = (frequency >= 1) & (frequency <= 45)
    total = np.trapz(power[..., total_mask], frequency[total_mask], axis=-1) + 1e-12
    spectral = []
    for low, high in BANDS:
        mask = (frequency >= low) & (frequency < high)
        band = np.trapz(power[..., mask], frequency[mask], axis=-1)
        spectral.append(np.log(band / total + 1e-12))
    # Robust relative features plus temporal spread; omit raw amplitude level.
    return np.concatenate([mean / std, np.log(std), np.log(rms), *spectral], axis=1)


def load_development(root):
    x = np.concatenate([np.load(root / f"{s}_waveforms.npy") for s in ("train", "val")])
    y = np.concatenate([np.load(root / f"{s}_labels.npy") for s in ("train", "val")])
    starts = np.r_[0, np.flatnonzero(y[1:] != y[:-1]) + 1, len(y)]
    runs = list(zip(starts[:-1], starts[1:]))
    if len(runs) % 2:
        raise ValueError("Expected paired rest/focus runs per subject")
    groups = np.empty(len(y), dtype=np.int64)
    for subject, index in enumerate(range(0, len(runs), 2)):
        groups[runs[index][0]:runs[index + 1][1]] = subject
    return x, y, groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/mental_arithmetic_250hz_8ch")
    parser.add_argument("--splits", type=int, default=8)
    parser.add_argument("--output", default="artifacts/feature_fusion_cv.json")
    args = parser.parse_args()
    x, y, groups = load_development(Path(args.data))
    x_features = features(x)
    rows = []
    for fold, (train_idx, val_idx) in enumerate(GroupKFold(args.splits).split(x_features, y, groups)):
        model = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                alpha=1e-3,
                batch_size=64,
                learning_rate_init=1e-3,
                max_iter=80,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=42,
            ),
        )
        model.fit(x_features[train_idx], y[train_idx])
        probability = model.predict_proba(x_features[val_idx])[:, 1]
        prediction = probability >= 0.5
        rows.append({
            "fold": fold,
            "validation_subjects": sorted(np.unique(groups[val_idx]).tolist()),
            "balanced_accuracy": float(balanced_accuracy_score(y[val_idx], prediction)),
            "macro_f1": float(f1_score(y[val_idx], prediction, average="macro", zero_division=0)),
            "roc_auc": float(roc_auc_score(y[val_idx], probability)),
        })
    metrics = ("balanced_accuracy", "macro_f1", "roc_auc")
    result = {
        "protocol": "GroupKFold",
        "features": "per-channel normalized temporal statistics + relative log band power",
        "n_subjects": int(np.unique(groups).size),
        "folds": rows,
        "mean": {key: float(np.mean([row[key] for row in rows])) for key in metrics},
        "std": {key: float(np.std([row[key] for row in rows])) for key in metrics},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
