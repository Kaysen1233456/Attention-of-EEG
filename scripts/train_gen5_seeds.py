"""Run the fixed Gen5 protocol for seeds 42, 43, and 44."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/mental_arithmetic_250hz_8ch")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", default="artifacts/gen5_seeds")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        target = root / f"seed_{seed}"
        command = [
            sys.executable, "scripts/final_spectral_cnn.py",
            "--data", args.data,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--seed", str(seed),
            "--output", str(target),
        ]
        subprocess.run(command, check=True)
        result = json.loads((target / "results.json").read_text(encoding="utf-8"))
        rows.append(result)
    metric_names = ("balanced_accuracy", "macro_f1", "roc_auc")
    label_hashes = sorted({row["test_labels_sha256"] for row in rows})
    summary = {
        "protocol": "fixed Gen5; identical test labels across seeds; one evaluation per seed",
        "seeds": args.seeds,
        "same_test_labels": len(label_hashes) == 1,
        "test_labels_sha256": label_hashes,
        "per_seed": [
            {"seed": row["seed"], "test_metrics": row["test_metrics"]}
            for row in rows
        ],
        "mean": {
            key: sum(row["test_metrics"][key] for row in rows) / len(rows)
            for key in metric_names
        },
        "std": {
            key: (
                sum((row["test_metrics"][key] - sum(item["test_metrics"][key] for item in rows) / len(rows)) ** 2 for row in rows)
                / len(rows)
            ) ** 0.5
            for key in metric_names
        },
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
