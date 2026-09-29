"""Three-seed, eight-fold confirmation of frozen Teacher B v2.

Only development train/val subjects are loaded. Test data is never read.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, TensorDataset

from teacher_b_common import evaluate, load_development, normalize_windows, train_fold

METRICS = ("balanced_accuracy", "macro_f1", "roc_auc")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/mental_arithmetic_250hz_8ch")
    parser.add_argument("--config", default="configs/teacher_b_v2.json")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    params = {key: config[key] for key in (
        "d_model", "n_heads", "n_layers", "dropout", "learning_rate", "weight_decay", "batch_size"
    )}
    epochs = args.epochs or int(config.get("epochs", 15))
    x, y, groups = load_development(Path(args.data))
    x = normalize_windows(x)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = list(GroupKFold(8).split(x, y, groups))
    seed_results = []

    for seed in args.seeds:
        folds = []
        subjects = []
        for fold, (train_idx, val_idx) in enumerate(splits):
            metrics, model = train_fold(
                x, y, train_idx, val_idx, params, seed + fold, epochs, device, return_model=True
            )
            val_subjects = sorted(np.unique(groups[val_idx]).tolist())
            folds.append({"fold": fold, "validation_subjects": val_subjects, **metrics})
            for subject in val_subjects:
                subject_idx = val_idx[groups[val_idx] == subject]
                loader = DataLoader(
                    TensorDataset(torch.from_numpy(x[subject_idx]), torch.from_numpy(y[subject_idx])),
                    batch_size=params["batch_size"],
                )
                subjects.append({"subject": subject, **evaluate(model, loader, device)})
            print({"seed": seed, **folds[-1]}, flush=True)
        seed_results.append({
            "seed": seed,
            "folds": folds,
            "subjects": sorted(subjects, key=lambda row: row["subject"]),
            "mean": {key: float(np.mean([row[key] for row in folds])) for key in METRICS},
            "std": {key: float(np.std([row[key] for row in folds])) for key in METRICS},
        })

    summary = {
        "stage": "B-v2-seed-confirmation",
        "model": config["model"],
        "config": str(Path(args.config)),
        "params": params,
        "epochs": epochs,
        "seeds": args.seeds,
        "folds": 8,
        "data_protocol": "development_only_subject_group_kfold",
        "test_accessed": False,
        "seed_results": seed_results,
        "overall_mean": {key: float(np.mean([row["mean"][key] for row in seed_results])) for key in METRICS},
        "overall_seed_std": {key: float(np.std([row["mean"][key] for row in seed_results])) for key in METRICS},
        "minimum_fold_balanced_accuracy": float(min(
            fold["balanced_accuracy"] for result in seed_results for fold in result["folds"]
        )),
        "acceptance": {
            "overall_mean_ba_at_least": 0.72,
            "overall_seed_ba_std_at_most": 0.05,
            "minimum_seed_mean_ba": 0.70,
            "minimum_fold_ba_at_least": 0.62,
        },
    }
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
