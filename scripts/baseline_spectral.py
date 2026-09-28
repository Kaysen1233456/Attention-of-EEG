"""Train a simple subject-disjoint spectral baseline."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


BANDS = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45))


def features(x):
    frequency, power = welch(x, fs=250, nperseg=250, axis=-1)
    total_mask = (frequency >= 1) & (frequency <= 45)
    total = np.trapz(power[..., total_mask], frequency[total_mask], axis=-1) + 1e-12
    values = []
    for low, high in BANDS:
        mask = (frequency >= low) & (frequency < high)
        band = np.trapz(power[..., mask], frequency[mask], axis=-1)
        values.append(np.log(band / total + 1e-12))
    return np.concatenate(values, axis=1)


def evaluate(model, x, y):
    probability = model.predict_proba(features(x))[:, 1]
    prediction = probability >= 0.5
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/mental_arithmetic_250hz_8ch")
    parser.add_argument("--output", default="artifacts/spectral_baseline.json")
    args = parser.parse_args()
    root = Path(args.data)
    arrays = {
        split: (np.load(root / f"{split}_waveforms.npy"), np.load(root / f"{split}_labels.npy"))
        for split in ("train", "val", "test")
    }
    train_x, train_y = arrays["train"]
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    model.fit(features(train_x), train_y)
    result = {split: evaluate(model, *arrays[split]) for split in arrays}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
