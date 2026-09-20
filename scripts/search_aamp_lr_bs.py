"""AAMP 学习率/批大小搜索：准随机覆盖后接贝叶斯优化。"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from attention_model.search import BayesianOptimization, QuasiRandomSearch


def parse_args():
    parser = argparse.ArgumentParser(description="AAMP learning-rate/batch-size search")
    parser.add_argument("--config", default="configs/pretrain_aamp_ear_saad.yaml")
    parser.add_argument("--output", default="artifacts/aamp_lr_bs_search_v1")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quasi-trials", type=int, default=8)
    parser.add_argument("--bayes-trials", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=None)
    return parser.parse_args()


def run_trial(args, trial_index, params, phase):
    output_dir = Path(args.output) / "trials" / f"trial_{trial_index:03d}"
    command = [
        sys.executable, "scripts/pretrain_aamp.py",
        "--config", args.config, "--output", str(output_dir),
        "--epochs", str(args.epochs), "--batch-size", str(params["batch_size"]),
        "--lr", f"{params['learning_rate']:.12g}", "--seed", str(args.seed),
        "--unlabeled-splits", "train_only", "--checkpoint-selection", "validation",
        "--patience", str(args.patience), "--device", args.device,
    ]
    if args.num_workers is not None:
        command.extend(["--num-workers", str(args.num_workers)])

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    print("\n启动试验:", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env)
    if completed.returncode != 0:
        raise RuntimeError(f"trial failed with exit code {completed.returncode}")

    history = json.loads((output_dir / "pretrain_history.json").read_text(encoding="utf-8"))
    best = min(history, key=lambda item: item["val_loss"])
    gap = float(best["val_loss"] - best["train_loss"])
    metrics = {
        "objective": float(best["val_loss"] + 0.25 * max(0.0, gap)),
        "best_val_loss": float(best["val_loss"]),
        "train_loss_at_best_val": float(best["train_loss"]),
        "generalization_gap": gap,
        "best_epoch": int(best["epoch"]),
        "last_val_loss": float(history[-1]["val_loss"]),
        "n_epochs_run": len(history),
        "phase": phase,
        "trial_output": str(output_dir),
    }
    print("试验结果:", json.dumps(metrics, ensure_ascii=False), flush=True)
    return metrics


def main():
    args = parse_args()
    if args.quasi_trials < 2 or args.bayes_trials < 1:
        raise ValueError("quasi-trials 至少为2，bayes-trials至少为1")

    search_space = {
        "learning_rate": [1e-5, 5e-4, "log_float"],
        "batch_size": [16, 32, 48, 64],
    }
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "search_protocol.json").write_text(json.dumps({
        "objective": "best_val_loss + 0.25 * max(0, generalization_gap)",
        "lower_is_better": True,
        "validation_is_train_only": True,
        "test_exposed": False,
        "search_space": search_space,
        "seed": args.seed,
        "quasi_trials": args.quasi_trials,
        "bayes_trials": args.bayes_trials,
        "epochs_per_trial": args.epochs,
        "patience": args.patience,
    }, indent=2), encoding="utf-8")

    quasi = QuasiRandomSearch(
        search_space=search_space, n_trials=args.quasi_trials,
        method="sobol", seed=args.seed, objective_metric="objective", maximize=False,
    )
    for trial_index in range(args.quasi_trials):
        params = quasi.get_trial_params(trial_index)
        metrics = run_trial(args, trial_index, params, "quasi_random")
        quasi.record_result(trial_index, params, metrics)
        quasi.save_results(output_dir / "quasi_random")

    bayes = BayesianOptimization(
        search_space=search_space, n_trials=args.bayes_trials, n_initial=0,
        acquisition="ei", kernel="matern", seed=args.seed + 1,
        objective_metric="objective", maximize=False,
    )
    for result in quasi.results:
        bayes.record_result(result["params"], result["metrics"])
    for offset in range(args.bayes_trials):
        params = bayes.get_next_params()
        trial_index = args.quasi_trials + offset
        metrics = run_trial(args, trial_index, params, "bayesian")
        bayes.record_result(params, metrics)
        bayes.save_results(output_dir / "bayesian")

    summary = bayes.get_summary()
    summary["quasi_random_summary"] = quasi.get_summary()
    summary["search_protocol_path"] = str(output_dir / "search_protocol.json")
    (output_dir / "search_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    (output_dir / "recommended_params.json").write_text(
        json.dumps(summary["best_params"], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n搜索完成。推荐参数:", json.dumps(summary["best_params"], ensure_ascii=False))


if __name__ == "__main__":
    main()
