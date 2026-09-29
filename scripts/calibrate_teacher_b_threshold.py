"""Leakage-safe threshold calibration for frozen Teacher B v2.

For every outer fold, thresholds are selected only from out-of-fold
probabilities generated inside outer-train subjects. The outer validation
subjects are never used to choose a threshold.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, TensorDataset

from teacher_b_common import (
    evaluate, load_development, normalize_windows, predict_probabilities, train_fold,
)


def loader_for(x, y, indices, batch_size):
    return DataLoader(
        TensorDataset(torch.from_numpy(x[indices]), torch.from_numpy(y[indices])),
        batch_size=batch_size,
    )


def select_threshold(probabilities, labels):
    candidates = np.linspace(0.05, 0.95, 181)
    scores = [balanced_accuracy_score(labels, probabilities >= value) for value in candidates]
    return float(candidates[int(np.argmax(scores))])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/mental_arithmetic_250hz_8ch")
    parser.add_argument("--config", default="configs/teacher_b_v2.json")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    params = {key: config[key] for key in (
        "d_model", "n_heads", "n_layers", "dropout", "learning_rate", "weight_decay", "batch_size"
    )}
    epochs = args.epochs or int(config.get("epochs", 15))
    x, y, groups = load_development(Path(args.data)); x = normalize_windows(x)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outer_splits = list(GroupKFold(8).split(x, y, groups))
    all_seed_results = []

    for seed in args.seeds:
        rows = []
        for outer_fold, (outer_train, outer_val) in enumerate(outer_splits):
            inner_groups = groups[outer_train]
            inner_x, inner_y = x[outer_train], y[outer_train]
            oof_prob = np.full(len(outer_train), np.nan, dtype=np.float64)
            inner_splits = list(GroupKFold(args.inner_folds).split(inner_x, inner_y, inner_groups))
            for inner_fold, (inner_train, inner_val) in enumerate(inner_splits):
                metrics, model = train_fold(
                    inner_x, inner_y, inner_train, inner_val, params,
                    seed + outer_fold * 100 + inner_fold, epochs, device, return_model=True,
                )
                loader = loader_for(inner_x, inner_y, inner_val, params["batch_size"])
                probabilities, _ = predict_probabilities(model, loader, device)
                oof_prob[inner_val] = probabilities
            if np.isnan(oof_prob).any():
                raise RuntimeError("Inner GroupKFold did not produce complete out-of-fold probabilities")
            threshold = select_threshold(oof_prob, inner_y)

            # Fit the outer-train model using an inner validation split for early stopping.
            fit_train, fit_val = inner_splits[0]
            _, final_model = train_fold(
                inner_x, inner_y, fit_train, fit_val, params,
                seed + outer_fold * 1000 + 999, epochs, device, return_model=True,
            )
            outer_loader = loader_for(x, y, outer_val, params["batch_size"])
            calibrated = evaluate(final_model, outer_loader, device, threshold=threshold)
            fixed = evaluate(final_model, outer_loader, device, threshold=0.5)
            rows.append({
                "fold": outer_fold,
                "validation_subjects": sorted(np.unique(groups[outer_val]).tolist()),
                "selected_threshold": threshold,
                "fixed_05": fixed,
                "calibrated": calibrated,
            })
            print({"seed": seed, **rows[-1]}, flush=True)
        all_seed_results.append({
            "seed": seed,
            "folds": rows,
            "fixed_05_mean": float(np.mean([r["fixed_05"]["balanced_accuracy"] for r in rows])),
            "calibrated_mean": float(np.mean([r["calibrated"]["balanced_accuracy"] for r in rows])),
            "fixed_05_auc": float(np.mean([r["fixed_05"]["roc_auc"] for r in rows])),
            "calibrated_auc": float(np.mean([r["calibrated"]["roc_auc"] for r in rows])),
        })

    result = {
        "stage": "B-v2-inner-threshold-calibration",
        "model": config["model"], "params": params, "epochs": epochs,
        "seeds": args.seeds, "outer_folds": 8, "inner_folds": args.inner_folds,
        "data_protocol": "development_only_nested_group_kfold",
        "test_accessed": False, "seed_results": all_seed_results,
        "overall_fixed_05_ba": float(np.mean([r["fixed_05_mean"] for r in all_seed_results])),
        "overall_calibrated_ba": float(np.mean([r["calibrated_mean"] for r in all_seed_results])),
        "overall_fixed_05_auc": float(np.mean([r["fixed_05_auc"] for r in all_seed_results])),
        "overall_calibrated_auc": float(np.mean([r["calibrated_auc"] for r in all_seed_results])),
    }
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
