"""Sobol exploration followed by Gaussian-process Bayesian tuning."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from attention_model.search import BayesianOptimization, QuasiRandomSearch


SEARCH_SPACE = {
    "learning_rate": [5e-5, 1e-3, "log_float"],
    "batch_size": [8, 16, 24, 32],
    "weight_decay": [1e-6, 1e-2, "log_float"],
    "dropout": [0.0, 0.3, "float"],
    "d_model": [48, 72, 96],
    "n_layers": [2, 3, 4],
    "temporal_pool": [2, 4, 8],
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pretrain_aamp_mental_arithmetic.yaml")
    parser.add_argument("--output", default="artifacts/mental_arithmetic_frequency_search_v1")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quasi-trials", type=int, default=12)
    parser.add_argument("--bayes-trials", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def objective_from_history(history):
    if not history:
        raise ValueError("Training history is empty")
    best = min(history, key=lambda row: row["val_loss"])
    gap = float(best["val_loss"] - best["train_loss"])
    return {
        "objective": float(best["val_loss"] + 0.25 * max(0.0, gap)),
        "best_val_loss": float(best["val_loss"]),
        "train_loss_at_best_val": float(best["train_loss"]),
        "generalization_gap": gap,
        "best_epoch": int(best["epoch"]),
        "epochs_run": len(history),
    }


def run_trial(args, trial_index, params, phase):
    trial_dir = Path(args.output) / "trials" / f"trial_{trial_index:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "scripts/pretrain_aamp.py",
        "--config", args.config,
        "--output", str(trial_dir),
        "--epochs", str(args.epochs),
        "--batch-size", str(params["batch_size"]),
        "--lr", f"{params['learning_rate']:.12g}",
        "--weight-decay", f"{params['weight_decay']:.12g}",
        "--dropout", f"{params['dropout']:.8g}",
        "--d-model", str(params["d_model"]),
        "--n-layers", str(params["n_layers"]),
        "--temporal-pool", str(params["temporal_pool"]),
        "--seed", str(args.seed),
        "--unlabeled-splits", "train_only",
        "--checkpoint-selection", "validation",
        "--patience", str(args.patience),
        "--num-workers", str(args.num_workers),
        "--device", args.device,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    print(f"\n[{phase}] trial {trial_index:03d}: {params}", flush=True)
    with (trial_dir / "run.log").open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            command, cwd=PROJECT_ROOT, env=env, stdout=log_file,
            stderr=subprocess.STDOUT, text=True,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"training command failed ({completed.returncode}); see {trial_dir / 'run.log'}")

    history_path = trial_dir / "pretrain_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    metrics = objective_from_history(history)
    metrics.update({"phase": phase, "trial_output": str(trial_dir)})
    (trial_dir / "trial_result.json").write_text(
        json.dumps({"params": params, "metrics": metrics}, indent=2), encoding="utf-8"
    )
    return metrics


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    args = parse_args()
    if args.quasi_trials < 4 or args.bayes_trials < 1:
        raise ValueError("Use at least 4 Sobol trials and 1 Bayesian trial")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "search_protocol.json", {
        "objective": "best_val_loss + 0.25 * max(0, val_loss - train_loss)",
        "objective_domain": "masked alpha-band complex rFFT magnitude MAE",
        "lower_is_better": True,
        "training_pool": "train split only",
        "validation": "validation split, fixed mask seed per run",
        "test_exposed": False,
        "masking": "frequency; alpha 8-13 Hz",
        "search_space": SEARCH_SPACE,
        "seed": args.seed,
        "quasi_trials": args.quasi_trials,
        "bayesian_trials": args.bayes_trials,
        "epochs_per_trial": args.epochs,
        "patience": args.patience,
        "device": args.device,
    })

    quasi = QuasiRandomSearch(
        SEARCH_SPACE, n_trials=args.quasi_trials, method="sobol", seed=args.seed,
        objective_metric="objective", maximize=False,
    )
    all_rows = []
    for index in range(args.quasi_trials):
        params = quasi.get_trial_params(index)
        try:
            metrics = run_trial(args, index, params, "sobol")
            quasi.record_result(index, params, metrics)
            all_rows.append({"trial_idx": index, "params": params, "metrics": metrics, "status": "success"})
        except Exception as exc:
            all_rows.append({"trial_idx": index, "params": params, "status": "failed", "error": repr(exc)})
        write_json(output / "search_results.json", {"trials": all_rows})
        quasi.save_results(output / "sobol_results")

    successful = [row for row in all_rows if row["status"] == "success"]
    if len(successful) < 2:
        raise RuntimeError("Fewer than two successful Sobol trials; cannot fit Bayesian surrogate")

    bayes = BayesianOptimization(
        SEARCH_SPACE, n_trials=args.bayes_trials, n_initial=0, acquisition="ei",
        kernel="matern", seed=args.seed + 1, objective_metric="objective", maximize=False,
    )
    for row in successful:
        bayes.record_result(row["params"], row["metrics"])

    for offset in range(args.bayes_trials):
        params = bayes.get_next_params()
        index = args.quasi_trials + offset
        try:
            metrics = run_trial(args, index, params, "bayesian")
            bayes.record_result(params, metrics)
            all_rows.append({"trial_idx": index, "params": params, "metrics": metrics, "status": "success"})
        except Exception as exc:
            all_rows.append({"trial_idx": index, "params": params, "status": "failed", "error": repr(exc)})
        write_json(output / "search_results.json", {"trials": all_rows})
        bayes.save_results(output / "bayesian_results")

    summary = bayes.get_summary()
    summary.update({
        "best_params": bayes.best_params,
        "best_objective": bayes.best_value,
        "search_protocol": str(output / "search_protocol.json"),
        "n_successful_trials": sum(row["status"] == "success" for row in all_rows),
    })
    write_json(output / "search_summary.json", summary)
    if bayes.best_params is None:
        raise RuntimeError("No successful search result")
    write_json(output / "best_pretrain_params.json", {
        "params": bayes.best_params,
        "objective": bayes.best_value,
        "source": "Sobol + Gaussian-process Bayesian optimization",
        "config": args.config,
    })
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
