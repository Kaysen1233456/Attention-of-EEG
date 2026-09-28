"""Aggregate completed teacher experiments without loading model checkpoints."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


TARGET_BA = 0.70
EXPECTED_FOLDS = {0, 1, 2}
EXPECTED_SEEDS = {42, 43, 44}


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _std(values: list[float]) -> float | None:
    return statistics.pstdev(values) if values else None


def _metric_summary(rows: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return {
        "mean": _mean(values),
        "std_population": _std(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def _run_row(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    validation = result["best_val_metrics"]
    train = result.get("best_checkpoint_train_metrics") or {}
    confusion = validation["confusion_matrix"]
    recalls = [
        confusion[0][0] / sum(confusion[0]),
        confusion[1][1] / sum(confusion[1]),
    ]
    return {
        "fold": int(path.parts[-3].split("_")[1]),
        "seed": int(path.parts[-2].split("_")[1]),
        "path": str(path),
        "best_epoch": result.get("best_epoch"),
        "balanced_accuracy": float(validation["balanced_accuracy"]),
        "macro_f1": float(validation["macro_f1"]),
        "roc_auc": float(validation["roc_auc"]),
        "subject_trial_balanced_accuracy": float(
            validation["subject_trial_balanced_accuracy"]
        ),
        "train_balanced_accuracy": float(train["balanced_accuracy"]),
        "train_validation_gap": float(
            train["balanced_accuracy"] - validation["balanced_accuracy"]
        ),
        "class_recalls": recalls,
        "class_collapse": min(recalls) < 0.10,
        "test_metrics": result.get("test_metrics"),
    }


def load_runs(root: Path) -> list[dict[str, Any]]:
    paths = sorted(Path(root).glob("fold_*/seed_*/results.json"))
    return [_run_row(path) for path in paths]


def summarize_condition(name: str, root: Path) -> dict[str, Any]:
    rows = load_runs(root)
    folds = sorted({row["fold"] for row in rows})
    seeds = sorted({row["seed"] for row in rows})
    expected_pairs = [
        {"fold": fold, "seed": seed}
        for fold in sorted(EXPECTED_FOLDS)
        for seed in sorted(EXPECTED_SEEDS)
    ]
    present_pairs = [{"fold": row["fold"], "seed": row["seed"]} for row in rows]
    missing_pairs = [
        pair for pair in expected_pairs if pair not in present_pairs
    ]
    for row in rows:
        if row["test_metrics"] is not None:
            raise ValueError(f"Test metrics are present in development run: {row['path']}")
    collapsed = [
        {
            "fold": row["fold"],
            "seed": row["seed"],
            "class_recalls": row["class_recalls"],
            "path": row["path"],
        }
        for row in rows
        if row["class_collapse"]
    ]
    metrics = {
        metric: _metric_summary(rows, metric)
        for metric in (
            "balanced_accuracy",
            "macro_f1",
            "roc_auc",
            "subject_trial_balanced_accuracy",
            "train_validation_gap",
        )
    }
    fold_summary = {}
    for fold in sorted(set(folds)):
        fold_rows = [row for row in rows if row["fold"] == fold]
        fold_summary[str(fold)] = {
            "n_runs": len(fold_rows),
            "metrics": {
                metric: _metric_summary(fold_rows, metric)
                for metric in ("balanced_accuracy", "macro_f1", "roc_auc")
            },
        }
    return {
        "name": name,
        "root": str(root),
        "n_runs": len(rows),
        "folds": folds,
        "seeds": seeds,
        "complete_3x3": not missing_pairs,
        "missing_pairs": missing_pairs,
        "metrics": metrics,
        "folds_summary": fold_summary,
        "collapsed_runs": collapsed,
        "runs": rows,
        "teacher_gate": {
            "target_mean_balanced_accuracy": TARGET_BA,
            "current_mean_balanced_accuracy": metrics["balanced_accuracy"]["mean"],
            "mean_ba_pass": (
                metrics["balanced_accuracy"]["mean"] is not None
                and metrics["balanced_accuracy"]["mean"] >= TARGET_BA
            ),
            "no_class_collapse": not collapsed,
            "gate_pass": (
                not missing_pairs
                and metrics["balanced_accuracy"]["mean"] is not None
                and metrics["balanced_accuracy"]["mean"] >= TARGET_BA
                and not collapsed
            ),
        },
    }


def compare_conditions(
    full: dict[str, Any],
    frozen: dict[str, Any],
    b0: dict[str, Any],
    b1: dict[str, Any],
) -> dict[str, Any]:
    full_mean = full["metrics"]["balanced_accuracy"]["mean"]
    frozen_mean = frozen["metrics"]["balanced_accuracy"]["mean"]
    b0_mean = b0["metrics"]["balanced_accuracy"]["mean"]
    b1_mean = b1["metrics"]["balanced_accuracy"]["mean"]
    return {
        "balanced_accuracy_means": {
            "AAMP_full_finetune": full_mean,
            "AAMP_frozen_completed_runs": frozen_mean,
            "random_init_B1": b1_mean,
            "CNN_B0": b0_mean,
        },
        "delta_vs_random_init_B1": (
            full_mean - b1_mean
            if full_mean is not None and b1_mean is not None
            else None
        ),
        "delta_vs_CNN_B0": (
            full_mean - b0_mean
            if full_mean is not None and b0_mean is not None
            else None
        ),
        "interpretation": (
            "AAMP full fine-tuning improves over the available B1 and B0 "
            "development references, but remains below the teacher gate."
        ),
    }


def build_report(repo_root: Path) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    full = summarize_condition(
        "AAMP full fine-tuning",
        repo_root / "artifacts/teacher_aamp_finetune_ear_saad",
    )
    frozen = summarize_condition(
        "AAMP frozen encoder",
        repo_root / "artifacts/teacher_aamp_frozen_ear_saad",
    )
    b0 = summarize_condition(
        "CNN B0",
        repo_root / "artifacts/stage1_b0_b1_cv_v1/B0",
    )
    b1 = summarize_condition(
        "Random-init Transformer B1",
        repo_root / "artifacts/stage1_b0_b1_cv_v1/B1",
    )
    comparison = compare_conditions(full, frozen, b0, b1)
    return {
        "protocol": {
            "test_used": False,
            "target_mean_balanced_accuracy": TARGET_BA,
            "required_folds": sorted(EXPECTED_FOLDS),
            "required_seeds": sorted(EXPECTED_SEEDS),
        },
        "conditions": {
            "full_finetune": full,
            "frozen": frozen,
            "random_init_B1": b1,
            "cnn_B0": b0,
        },
        "comparison": comparison,
        "decision": {
            "teacher_retain": False,
            "cnn_distillation_allowed": False,
            "statement": (
                "AAMP reconstruction effective but downstream classification "
                "unqualified under the current fixed development protocol."
            ),
            "reasons": [
                "AAMP full fine-tuning mean validation BA is below 0.70.",
                "Frozen encoder evaluation is incomplete: fold 0 seeds 43 and 44 are missing.",
                "Class-collapse or near-collapse runs remain.",
                "Fold 2 validation is near chance across AAMP conditions.",
            ],
        },
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def render_markdown(report: dict[str, Any]) -> str:
    conditions = report["conditions"]
    lines = [
        "# AAMP Teacher Evaluation Report",
        "",
        "Generated from existing development artifacts without loading checkpoints.",
        "",
        "## Decision",
        "",
        f"- Teacher retain: **{'PASS' if report['decision']['teacher_retain'] else 'FAIL'}**",
        f"- CNN distillation allowed: **{'YES' if report['decision']['cnn_distillation_allowed'] else 'NO'}**",
        f"- Required mean validation BA: **>= {report['protocol']['target_mean_balanced_accuracy']:.2f}**",
        f"- Decision: {report['decision']['statement']}",
        "",
        "## Aggregate Metrics",
        "",
        "| Condition | Runs | Complete 3x3 | BA mean +/- std | Macro-F1 mean | ROC-AUC mean | Collapsed runs |",
        "|---|---:|:---:|---:|---:|---:|---:|",
    ]
    for key in ("full_finetune", "frozen", "random_init_B1", "cnn_B0"):
        condition = conditions[key]
        metrics = condition["metrics"]
        lines.append(
            f"| {condition['name']} | {condition['n_runs']} | "
            f"{'YES' if condition['complete_3x3'] else 'NO'} | "
            f"{_fmt(metrics['balanced_accuracy']['mean'])} +/- {_fmt(metrics['balanced_accuracy']['std_population'])} | "
            f"{_fmt(metrics['macro_f1']['mean'])} | "
            f"{_fmt(metrics['roc_auc']['mean'])} | "
            f"{len(condition['collapsed_runs'])} |"
        )
    lines.extend(
        [
            "",
            "## Fold Means",
            "",
            "| Condition | Fold 0 BA | Fold 1 BA | Fold 2 BA |",
            "|---|---:|---:|---:|",
        ]
    )
    for key in ("full_finetune", "frozen", "random_init_B1", "cnn_B0"):
        condition = conditions[key]
        fold_values = []
        for fold in range(3):
            summary = condition["folds_summary"].get(str(fold))
            value = summary["metrics"]["balanced_accuracy"]["mean"] if summary else None
            fold_values.append(_fmt(value))
        lines.append(f"| {condition['name']} | {' | '.join(fold_values)} |")
    lines.extend(
        [
            "",
            "## Missing Runs",
            "",
            f"- Frozen encoder missing pairs: {conditions['frozen']['missing_pairs'] or 'none'}",
            "",
            "## Guidance",
            "",
            "1. Do not start CNN distillation.",
            "2. Treat the completed AAMP frozen encoder as Baseline A.",
            "3. Fold 2 is a multi-subject robustness failure cluster, not a single-subject failure.",
            "4. Audit source-level label semantics and video/speaker mapping before changing model or training parameters.",
            "5. Keep Baseline A unchanged and compare one factor at a time in subsequent experiments.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_report(repo_root: Path, output_dir: Path) -> dict[str, Any]:
    report = build_report(repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "teacher_evaluation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "teacher_evaluation.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/teacher_aamp_evaluation_v1"),
    )
    args = parser.parse_args(argv)
    output = args.output
    if not output.is_absolute():
        output = args.repo_root / output
    report = write_report(args.repo_root, output)
    print(f"Report: {output / 'teacher_evaluation.md'}")
    print(f"JSON: {output / 'teacher_evaluation.json'}")
    print(
        "AAMP full fine-tuning mean BA: "
        f"{_fmt(report['conditions']['full_finetune']['metrics']['balanced_accuracy']['mean'])}"
    )
    print(
        "Teacher retain: "
        f"{'PASS' if report['decision']['teacher_retain'] else 'FAIL'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
