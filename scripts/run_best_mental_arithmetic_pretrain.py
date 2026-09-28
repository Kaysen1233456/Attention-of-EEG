"""Run the final fixed-epoch pretraining from a completed hyperparameter search."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pretrain_aamp_mental_arithmetic.yaml")
    parser.add_argument("--search-dir", default="artifacts/mental_arithmetic_frequency_search_v1")
    parser.add_argument("--output", default="artifacts/pretrain_mental_arithmetic_frequency_best")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    params_path = Path(args.search_dir) / "best_pretrain_params.json"
    if not params_path.exists():
        raise FileNotFoundError(f"Search result not found: {params_path}")
    params = json.loads(params_path.read_text(encoding="utf-8"))["params"]
    command = [
        sys.executable, "scripts/pretrain_aamp.py",
        "--config", args.config,
        "--output", args.output,
        "--epochs", str(args.epochs),
        "--batch-size", str(params["batch_size"]),
        "--lr", f"{params['learning_rate']:.12g}",
        "--weight-decay", f"{params['weight_decay']:.12g}",
        "--dropout", f"{params['dropout']:.8g}",
        "--d-model", str(params["d_model"]),
        "--n-layers", str(params["n_layers"]),
        "--temporal-pool", str(params["temporal_pool"]),
        "--seed", str(args.seed),
        "--unlabeled-splits", "train_val",
        "--checkpoint-selection", "fixed_epochs",
        "--num-workers", str(args.num_workers),
        "--device", args.device,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    print("Running final pretraining with:", json.dumps(params, ensure_ascii=False), flush=True)
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
