"""Hyperparameter search for Teacher B on development subjects only.

The test split is intentionally never loaded here.  Each trial uses the same
subject-disjoint GroupKFold splits so that random, Sobol, and Bayesian runs
are directly comparable.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

# Make the source package importable when this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from teacher_b_common import load_development, normalize_windows, train_fold


SEARCH_SPACE = {
    "learning_rate": [1e-4, 1e-3, "log_float"],
    "weight_decay": [1e-6, 1e-3, "log_float"],
    "dropout": [0.10, 0.35, "float"],
    "d_model": [96, 128],
    "n_heads": [4, 8],
    "n_layers": [1, 2],
    "batch_size": [128, 256],
}
METRIC_KEYS = ("balanced_accuracy", "macro_f1", "roc_auc")


def random_params(rng):
    """Sample one valid point, using log-uniform values for optimizer rates."""
    return {
        "learning_rate": float(np.exp(rng.uniform(np.log(1e-4), np.log(1e-3)))),
        "weight_decay": float(np.exp(rng.uniform(np.log(1e-6), np.log(1e-3)))),
        "dropout": float(rng.uniform(0.10, 0.35)),
        "d_model": int(rng.choice([96, 128])),
        "n_heads": int(rng.choice([4, 8])),
        "n_layers": int(rng.choice([1, 2])),
        "batch_size": int(rng.choice([128, 256])),
    }


def make_searcher(method, trials, seed):
    if method == "random":
        return None
    if method == "sobol":
        from attention_model.search.quasi_random import QuasiRandomSearch
        return QuasiRandomSearch(
            SEARCH_SPACE, n_trials=trials, method="sobol", seed=seed,
            objective_metric="objective",
        )
    if method == "bayesian":
        from attention_model.search.bayesian_optimization import BayesianOptimization
        return BayesianOptimization(
            SEARCH_SPACE, n_trials=trials, n_initial=min(5, trials),
            acquisition="ei", seed=seed, objective_metric="objective",
        )
    raise ValueError(f"Unsupported method: {method}")


def evaluate_trial(x, y, groups, splits, params, seed, epochs, device):
    rows = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        metrics = train_fold(x, y, train_idx, val_idx, params, seed + fold, epochs, device)
        rows.append({"fold": fold, **metrics})
    means = {key: float(np.mean([row[key] for row in rows])) for key in METRIC_KEYS}
    stds = {key: float(np.std([row[key] for row in rows])) for key in METRIC_KEYS}
    objective = means["balanced_accuracy"] - 0.25 * stds["balanced_accuracy"]
    return {
        "folds": rows, "mean": means, "std": stds,
        "objective": float(objective),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/mental_arithmetic_250hz_8ch")
    parser.add_argument("--method", choices=("random", "sobol", "bayesian"), required=True)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--folds", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.trials < 1 or args.folds < 2:
        raise ValueError("trials must be >= 1 and folds must be >= 2")

    x, y, groups = load_development(Path(args.data))
    x = normalize_windows(x)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = list(GroupKFold(args.folds).split(x, y, groups))
    rng = np.random.RandomState(args.seed)
    searcher = make_searcher(args.method, args.trials, args.seed)
    trials = []

    for trial_idx in range(args.trials):
        if args.method == "random":
            params = random_params(rng)
        elif args.method == "sobol":
            params = searcher.get_trial_params(trial_idx)
        else:
            params = searcher.get_next_params()
        params = {key: value for key, value in params.items() if key in SEARCH_SPACE}
        print(f"trial {trial_idx + 1}/{args.trials}: {params}", flush=True)
        try:
            metrics = evaluate_trial(x, y, groups, splits, params, args.seed, args.epochs, device)
            row = {"trial_idx": trial_idx, "params": params, **metrics, "status": "completed"}
            if args.method == "sobol":
                searcher.record_result(trial_idx, params, {"objective": metrics["objective"]})
            elif args.method == "bayesian":
                searcher.record_result(params, {"objective": metrics["objective"]})
            print(json.dumps(row, sort_keys=True), flush=True)
        except Exception as exc:
            row = {"trial_idx": trial_idx, "params": params, "status": "failed", "error": repr(exc)}
            print(json.dumps(row, sort_keys=True), flush=True)
        trials.append(row)
        out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
        (out / "trials.json").write_text(json.dumps(trials, indent=2), encoding="utf-8")

    completed = [row for row in trials if row["status"] == "completed"]
    if not completed:
        raise RuntimeError("No search trial completed successfully")
    best = max(completed, key=lambda row: row["objective"])
    summary = {
        "stage": "B-search", "method": args.method,
        "data_protocol": "development_only_subject_group_kfold",
        "test_accessed": False, "folds": args.folds, "epochs_per_trial": args.epochs,
        "protocol_alignment": "same fold count and epoch budget as fixed-model verification",
        "search_space": SEARCH_SPACE, "seed": args.seed,
        "best_params": best["params"], "best_objective": best["objective"],
        "best_metrics": {key: best[key] for key in ("mean", "std", "folds")},
        "n_trials": args.trials, "n_completed": len(completed), "trials": trials,
    }
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "best_params.json").write_text(json.dumps(best["params"], indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
