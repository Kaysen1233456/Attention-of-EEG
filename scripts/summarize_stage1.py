"""Summarize Stage 1 B0/B1 fold-seed runs without reading model data."""
import argparse
import json
from pathlib import Path

import numpy as np


METRICS = ("balanced_accuracy", "macro_f1", "roc_auc")


def parse_args():
    parser = argparse.ArgumentParser(description="汇总 B0/B1 三折多 seed 实验")
    parser.add_argument("--root", required=True, help="例如 artifacts/stage1_b0_b1_cv_v1")
    parser.add_argument("--output", default=None, help="报告输出目录，默认 root/reports")
    parser.add_argument("--baseline", default="B0", choices=("B0", "B1"))
    parser.add_argument("--candidate", default="B1", choices=("B0", "B1"))
    return parser.parse_args()


def load_runs(root: Path):
    rows = []
    for variant_dir in sorted(root.glob("B[01]")):
        variant = variant_dir.name
        for fold_dir in sorted(variant_dir.glob("fold_*")):
            fold = int(fold_dir.name.split("_")[-1])
            for seed_dir in sorted(fold_dir.glob("seed_*")):
                result_path = seed_dir / "results.json"
                if not result_path.exists():
                    continue
                result = json.loads(result_path.read_text(encoding="utf-8"))
                val = result["best_val_metrics"]
                train = result["best_checkpoint_train_metrics"]
                rows.append({
                    "variant": variant,
                    "fold": fold,
                    "seed": int(seed_dir.name.split("_")[-1]),
                    "run_dir": str(seed_dir),
                    "best_epoch": result["best_epoch"],
                    "validation": {key: val[key] for key in METRICS},
                    "confusion_matrix": val.get("confusion_matrix"),
                    "train": {key: train[key] for key in METRICS},
                    "gap": {
                        key: train[key] - val[key]
                        for key in METRICS
                    },
                    "per_subject": val.get("per_subject", {}),
                })
    if not rows:
        raise FileNotFoundError(f"No completed results.json found under {root}")
    return rows


def aggregate(rows, variant):
    selected = [row for row in rows if row["variant"] == variant]
    report = {"variant": variant, "n_runs": len(selected), "metrics": {}, "runs": selected}
    for split in ("validation", "train", "gap"):
        report["metrics"][split] = {}
        for key in METRICS:
            values = [row[split][key] for row in selected]
            report["metrics"][split][key] = {
                "mean": float(np.mean(values)),
                "std_population": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
    for fold in sorted({row["fold"] for row in selected}):
        fold_rows = [row for row in selected if row["fold"] == fold]
        report.setdefault("by_fold", {})[str(fold)] = {
            key: {
                "mean": float(np.mean([row["validation"][key] for row in fold_rows])),
                "std_population": float(np.std([row["validation"][key] for row in fold_rows])),
            }
            for key in METRICS
        }
    return report


def subject_report(rows):
    output = {}
    for row in rows:
        variant = row["variant"]
        for subject, metrics in row["per_subject"].items():
            output.setdefault(variant, {}).setdefault(subject, []).append({
                "fold": row["fold"],
                "seed": row["seed"],
                **metrics,
            })
    summary = {}
    for variant, subjects in output.items():
        summary[variant] = {}
        for subject, entries in subjects.items():
            summary[variant][subject] = {
                "n_runs": len(entries),
                "balanced_accuracy": {
                    "mean": float(np.mean([item["balanced_accuracy"] for item in entries])),
                    "std_population": float(np.std([item["balanced_accuracy"] for item in entries])),
                    "min": float(np.min([item["balanced_accuracy"] for item in entries])),
                    "max": float(np.max([item["balanced_accuracy"] for item in entries])),
                },
                "macro_f1": {
                    "mean": float(np.mean([item["macro_f1"] for item in entries])),
                    "std_population": float(np.std([item["macro_f1"] for item in entries])),
                },
                "roc_auc": {
                    "mean": float(np.mean([
                        item["roc_auc"] for item in entries if item["roc_auc"] is not None
                    ])) if any(item["roc_auc"] is not None for item in entries) else None,
                },
                "runs": entries,
            }
    return summary


def decision(baseline, candidate):
    deltas = {
        key: candidate["metrics"]["validation"][key]["mean"]
        - baseline["metrics"]["validation"][key]["mean"]
        for key in METRICS
    }
    fold_deltas = []
    for fold in sorted(set(baseline.get("by_fold", {})) & set(candidate.get("by_fold", {}))):
        fold_deltas.append(
            candidate["by_fold"][fold]["balanced_accuracy"]["mean"]
            - baseline["by_fold"][fold]["balanced_accuracy"]["mean"]
        )
    candidate_runs = candidate["runs"]
    collapsed = 0
    for run in candidate_runs:
        matrix = np.asarray(run["confusion_matrix"], dtype=float)
        recalls = np.diag(matrix) / np.maximum(matrix.sum(axis=1), 1.0)
        if np.any(recalls < 0.10):
            collapsed += 1
    passed = (
        deltas["balanced_accuracy"] >= 0.01
        and sum(delta > 0 for delta in fold_deltas) >= 2
        and deltas["macro_f1"] >= -0.01
        and deltas["roc_auc"] >= -0.01
        and collapsed == 0
    )
    return {
        "baseline": baseline["variant"],
        "candidate": candidate["variant"],
        "delta_mean": deltas,
        "fold_balanced_accuracy_deltas": fold_deltas,
        "candidate_collapsed_runs": collapsed,
        "teacher_retain_decision": "PASS" if passed else "FAIL",
        "criteria": {
            "delta_balanced_accuracy_at_least_0.01": deltas["balanced_accuracy"] >= 0.01,
            "at_least_two_folds_improve": sum(delta > 0 for delta in fold_deltas) >= 2,
            "macro_f1_not_materially_lower": deltas["macro_f1"] >= -0.01,
            "roc_auc_not_materially_lower": deltas["roc_auc"] >= -0.01,
            "no_subject_class_collapse": collapsed == 0,
        },
    }


def main():
    args = parse_args()
    root = Path(args.root)
    output = Path(args.output) if args.output else root / "reports"
    output.mkdir(parents=True, exist_ok=True)
    rows = load_runs(root)
    baseline = aggregate(rows, args.baseline)
    candidate = aggregate(rows, args.candidate)
    report = {
        "protocol": {
            "root": str(root),
            "variants": [args.baseline, args.candidate],
            "test_used": False,
            "selection_metric": "validation balanced_accuracy",
        },
        "baseline": baseline,
        "candidate": candidate,
        "decision": decision(baseline, candidate),
    }
    (output / "stage1_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (output / "subject_level_summary.json").write_text(
        json.dumps(subject_report(rows), indent=2), encoding="utf-8"
    )
    print(json.dumps(report["decision"], indent=2))


if __name__ == "__main__":
    main()
