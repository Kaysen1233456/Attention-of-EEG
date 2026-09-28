"""Read-only audit for the fold-2 validation subjects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


TARGET_SUBJECTS = [3, 6, 11, 13]


def _float(value: Any) -> float:
    return float(value) if np.isfinite(value) else None


def _summary(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": _float(np.mean(values)),
        "std": _float(np.std(values)),
        "min": _float(np.min(values)),
        "max": _float(np.max(values)),
        "p05": _float(np.percentile(values, 5)),
        "p50": _float(np.percentile(values, 50)),
        "p95": _float(np.percentile(values, 95)),
    }


def _trial_report(labels: np.ndarray, subjects: np.ndarray, trials: np.ndarray) -> dict[str, Any]:
    conflicts = []
    windows_per_trial = []
    for subject, trial in sorted(set(zip(subjects.tolist(), trials.tolist()))):
        mask = (subjects == subject) & (trials == trial)
        trial_labels = np.unique(labels[mask])
        windows_per_trial.append(int(mask.sum()))
        if len(trial_labels) != 1:
            conflicts.append(
                {
                    "subject": int(subject),
                    "trial": int(trial),
                    "labels": trial_labels.astype(int).tolist(),
                    "n_windows": int(mask.sum()),
                }
            )
    return {
        "n_trials": len(windows_per_trial),
        "windows_per_trial": _summary(np.asarray(windows_per_trial)),
        "trial_label_conflicts": len(conflicts),
        "conflicts": conflicts,
    }


def audit_split_arrays(
    waveforms: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    trials: np.ndarray,
) -> dict[str, Any]:
    """Summarize raw split integrity without modifying any input."""
    waveforms = np.asarray(waveforms)
    labels = np.asarray(labels)
    subjects = np.asarray(subjects)
    trials = np.asarray(trials)
    zero_channels = np.all(waveforms == 0.0, axis=-1)
    subject_report: dict[str, Any] = {}
    for subject in sorted(np.unique(subjects).tolist()):
        mask = subjects == subject
        subject_report[str(int(subject))] = {
            "n_windows": int(mask.sum()),
            "labels": {
                str(int(label)): int((labels[mask] == label).sum())
                for label in sorted(np.unique(labels[mask]).tolist())
            },
            "zero_channel_fraction": _float(zero_channels[mask].mean()),
            "zero_window_fraction": _float(np.all(zero_channels[mask], axis=1).mean()),
            "amplitude_abs": _summary(np.abs(waveforms[mask])),
            "per_window_rms": _summary(
                np.sqrt(np.mean(np.square(waveforms[mask].astype(np.float64)), axis=(1, 2)))
            ),
            "trial_report": _trial_report(labels[mask], subjects[mask], trials[mask]),
        }
    return {
        "n_windows": int(len(waveforms)),
        "shape": list(waveforms.shape),
        "labels": {
            str(int(label)): int((labels == label).sum())
            for label in sorted(np.unique(labels).tolist())
        },
        "zero_channel_fraction": _float(zero_channels.mean()),
        "zero_window_fraction": _float(np.all(zero_channels, axis=1).mean()),
        "amplitude_abs": _summary(np.abs(waveforms)),
        "per_window_rms": _summary(
            np.sqrt(np.mean(np.square(waveforms.astype(np.float64)), axis=(1, 2)))
        ),
        "trial_report": _trial_report(labels, subjects, trials),
        "subjects": subject_report,
    }


def classify_prediction_direction(
    balanced_accuracy: float,
    predicted_positive_rate: float,
    roc_auc: float | None = None,
) -> str:
    if (roc_auc is not None and roc_auc < 0.40) or balanced_accuracy < 0.40:
        return "systematic_inversion_candidate"
    if (roc_auc is not None and roc_auc > 0.60) or balanced_accuracy > 0.60:
        return "aligned_candidate"
    return "non_directional"


def _normalization_report(
    train_waveforms: np.ndarray,
    val_waveforms: np.ndarray,
    val_subjects: np.ndarray,
) -> dict[str, Any]:
    mean = train_waveforms.mean(axis=(0, 2), keepdims=True)
    std = train_waveforms.std(axis=(0, 2), keepdims=True) + 1e-8
    normalized = (val_waveforms - mean) / std
    clipped = np.clip(normalized, -8.0, 8.0)
    zero_channels = np.all(val_waveforms == 0.0, axis=-1)
    by_subject = {}
    for subject in TARGET_SUBJECTS:
        mask = val_subjects == subject
        if not np.any(mask):
            continue
        values = normalized[mask]
        by_subject[str(subject)] = {
            "n_windows": int(mask.sum()),
            "normalized_mean": _float(values.mean()),
            "normalized_std": _float(values.std()),
            "clip_fraction": _float(np.mean(np.abs(values) > 8.0)),
            "zero_channel_fraction": _float(zero_channels[mask].mean()),
        }
    return {
        "mode": "train_subjects_only",
        "clip_std": 8.0,
        "train_channel_mean_summary": _summary(mean.ravel()),
        "train_channel_std_summary": _summary(std.ravel()),
        "validation_normalized_mean": _float(normalized.mean()),
        "validation_normalized_std": _float(normalized.std()),
        "validation_clip_fraction": _float(np.mean(np.abs(normalized) > 8.0)),
        "validation_zero_channel_fraction": _float(zero_channels.mean()),
        "by_subject": by_subject,
    }


def _prediction_audit(results_root: Path) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for condition in ("teacher_aamp_finetune_ear_saad", "teacher_aamp_frozen_ear_saad"):
        rows = []
        for seed in (42, 43, 44):
            path = results_root / condition / "fold_2" / f"seed_{seed}" / "results.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            metrics = data["best_val_metrics"]
            for subject, item in metrics.get("per_subject", {}).items():
                if int(subject) not in TARGET_SUBJECTS:
                    continue
                cm = np.asarray(item["confusion_matrix"], dtype=int)
                predicted_positive_rate = float((cm[0, 1] + cm[1, 1]) / cm.sum())
                rows.append(
                    {
                        "seed": seed,
                        "subject": int(subject),
                        "balanced_accuracy": float(item["balanced_accuracy"]),
                        "roc_auc": float(item["roc_auc"]),
                        "label_positive_rate": float(cm[1].sum() / cm.sum()),
                        "predicted_positive_rate": predicted_positive_rate,
                        "prediction_direction": classify_prediction_direction(
                            float(item["balanced_accuracy"]),
                            predicted_positive_rate,
                            float(item["roc_auc"]),
                        ),
                        "confusion_matrix": cm.tolist(),
                    }
                )
        output[condition] = rows
    return output


def build_report(repo_root: Path) -> dict[str, Any]:
    data_dir = repo_root / "data" / "processed" / "ear_saad"
    train_w = np.load(data_dir / "train_waveforms.npy")
    train_y = np.load(data_dir / "train_labels.npy")
    train_s = np.load(data_dir / "train_subjects.npy")
    train_t = np.load(data_dir / "train_trials.npy")
    val_w = np.load(data_dir / "val_waveforms.npy")
    val_y = np.load(data_dir / "val_labels.npy")
    val_s = np.load(data_dir / "val_subjects.npy")
    val_t = np.load(data_dir / "val_trials.npy")

    fold = json.loads((repo_root / "configs" / "development_folds.json").read_text())["folds"][2]
    combined_w = np.concatenate([train_w, val_w])
    combined_y = np.concatenate([train_y, val_y])
    combined_s = np.concatenate([train_s, val_s])
    combined_t = np.concatenate([train_t, val_t])
    train_mask = np.isin(combined_s, fold["train_subjects"])
    val_mask = np.isin(combined_s, fold["val_subjects"])
    fold_train_w = combined_w[train_mask]
    fold_val_w = combined_w[val_mask]
    fold_val_y = combined_y[val_mask]
    fold_val_s = combined_s[val_mask]
    fold_val_t = combined_t[val_mask]
    return {
        "fold": fold,
        "target_subjects": TARGET_SUBJECTS,
        "raw_validation_audit": audit_split_arrays(
            fold_val_w, fold_val_y, fold_val_s, fold_val_t
        ),
        "fold_train_audit": audit_split_arrays(
            fold_train_w, combined_y[train_mask], combined_s[train_mask], combined_t[train_mask]
        ),
        "normalization_audit": _normalization_report(
            fold_train_w, fold_val_w, fold_val_s
        ),
        "prediction_audit": _prediction_audit(repo_root / "artifacts"),
    }


def _markdown(report: dict[str, Any]) -> str:
    raw = report["raw_validation_audit"]
    norm = report["normalization_audit"]
    lines = [
        "# Fold 2 Subject Audit",
        "",
        "Read-only audit of validation subjects 3/6/11/13. No model, config, or source array was modified.",
        "",
        "## Data integrity",
        f"- Windows: `{raw['n_windows']}`; shape: `{raw['shape']}`",
        f"- Labels: `{raw['labels']}`; trial-label conflicts: `{raw['trial_report']['trial_label_conflicts']}`",
        f"- Windows per trial: mean `{raw['trial_report']['windows_per_trial']['mean']:.2f}`, "
        f"min `{raw['trial_report']['windows_per_trial']['min']}`, max `{raw['trial_report']['windows_per_trial']['max']}`",
        f"- Zero-channel fraction: `{raw['zero_channel_fraction']:.6f}`; all-zero window fraction: `{raw['zero_window_fraction']:.6f}`",
        "",
        "## Per-subject distribution",
        "| Subject | Windows | Label 0/1 | Zero-channel | Abs amplitude p50 | RMS p50 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for subject in report["target_subjects"]:
        item = raw["subjects"][str(subject)]
        lines.append(
            f"| {subject} | {item['n_windows']} | "
            f"{item['labels'].get('0', 0)}/{item['labels'].get('1', 0)} | "
            f"{item['zero_channel_fraction']:.6f} | "
            f"{item['amplitude_abs']['p50']:.6g} | {item['per_window_rms']['p50']:.6g} |"
        )
    lines += [
        "",
        "## Train-subject normalization",
        f"- Mode: `{norm['mode']}`, clip: `±{norm['clip_std']}`",
        f"- Fold-train channel mean range: `{norm['train_channel_mean_summary']['min']:.6g}` to `{norm['train_channel_mean_summary']['max']:.6g}`",
        f"- Fold-train channel std range: `{norm['train_channel_std_summary']['min']:.6g}` to `{norm['train_channel_std_summary']['max']:.6g}`",
        f"- Fold-2 validation normalized mean/std: `{norm['validation_normalized_mean']:.6g}` / `{norm['validation_normalized_std']:.6g}`",
        f"- Clipped fraction: `{norm['validation_clip_fraction']:.6f}`",
        "",
        "## Prediction direction",
        "| Condition | Seed | Subject | BA | ROC-AUC | Label + | Pred + | Direction flag |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for condition, rows in report["prediction_audit"].items():
        for row in rows:
            lines.append(
                f"| {condition} | {row['seed']} | {row['subject']} | "
                f"{row['balanced_accuracy']:.4f} | {row['roc_auc']:.4f} | "
                f"{row['label_positive_rate']:.4f} | {row['predicted_positive_rate']:.4f} | "
                f"{row['prediction_direction']} |"
            )
    lines += [
        "",
        "## Interpretation rule",
        "- `systematic_inversion_candidate` means subject-level ROC-AUC < 0.40 or BA < 0.40; it is a diagnostic flag, not proof that labels are wrong.",
        "- Trial/window alignment is considered clean only when every `(subject, trial)` has one label and consistent window counts.",
        "- Prediction tables use confusion matrices already recorded by the completed fold-2 runs; no test data is read.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output", default="artifacts/fold2_subject_audit")
    args = parser.parse_args()
    root = Path(args.repo_root).resolve()
    output = root / args.output
    output.mkdir(parents=True, exist_ok=True)
    report = build_report(root)
    (output / "fold2_subject_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    (output / "fold2_subject_audit.md").write_text(_markdown(report), encoding="utf-8")
    print(f"Report: {output / 'fold2_subject_audit.md'}")
    print(f"JSON: {output / 'fold2_subject_audit.json'}")


if __name__ == "__main__":
    main()
