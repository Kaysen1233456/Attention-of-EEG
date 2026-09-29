"""Audit development data for the difficult Teacher B validation fold.

This script reads only train and val arrays. It reconstructs subject runs from
the processed label sequence, checks run boundaries and signal integrity, and
compares Fold 5 subjects with the remaining development subjects.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import welch

from cv_spectral_cnn import load_development

BANDS = {
    "delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0), "gamma": (30.0, 45.0),
}
FOLD_5 = {1, 10, 19, 26}


def runs_from_labels(labels):
    starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    return [(int(starts[i]), int(starts[i + 1]), int(labels[starts[i]]))
            for i in range(len(starts) - 1)]


def subject_names(root):
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    return [str(name) for name in metadata["train_subjects"] + metadata["val_subjects"]]


def summarize_subject(x, y, subject_id, name, start, end, sampling_rate):
    xs, ys = x[start:end], y[start:end]
    frequencies, psd = welch(xs, fs=sampling_rate, nperseg=min(250, xs.shape[-1]), axis=-1)
    total_mask = (frequencies >= 1) & (frequencies <= 45)
    total = np.trapz(psd[..., total_mask], frequencies[total_mask], axis=-1) + 1e-12
    band_power = {}
    for band, (low, high) in BANDS.items():
        mask = (frequencies >= low) & (frequencies < high)
        power = np.trapz(psd[..., mask], frequencies[mask], axis=-1)
        band_power[band] = float(np.mean(power / total))
    rms = np.sqrt(np.mean(xs * xs, axis=-1))
    return {
        "subject_id": int(subject_id), "subject_name": name,
        "start": start, "end": end, "windows": int(end - start),
        "label_counts": np.bincount(ys.astype(int), minlength=2).tolist(),
        "nonfinite_samples": int((~np.isfinite(xs)).sum()),
        "constant_windows": int(np.all(np.std(xs, axis=-1) < 1e-8, axis=1).sum()),
        "zero_windows": int(np.all(np.abs(xs) < 1e-8, axis=(1, 2)).sum()),
        "rms_mean": float(np.mean(rms)), "rms_std": float(np.std(rms)),
        "channel_mean": np.mean(xs, axis=(0, 2)).astype(float).tolist(),
        "channel_std": np.std(xs, axis=(0, 2)).astype(float).tolist(),
        "relative_band_power": band_power,
        "waveform_sha256": hashlib.sha256(xs.tobytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/mental_arithmetic_250hz_8ch")
    parser.add_argument("--output", default="artifacts/teacher_b_data_audit.json")
    parser.add_argument("--sampling-rate", type=float, default=250.0)
    args = parser.parse_args()
    root = Path(args.data)
    x, y, groups = load_development(root)
    names = subject_names(root)
    runs = runs_from_labels(y)
    if len(runs) != len(names) * 2:
        raise ValueError(f"Expected two label runs per subject: runs={len(runs)}, subjects={len(names)}")

    subjects = []
    errors = []
    for subject_id, name in enumerate(names):
        first, second = runs[2 * subject_id:2 * subject_id + 2]
        if first[2] != 0 or second[2] != 1:
            errors.append({"subject_id": subject_id, "subject_name": name,
                           "labels": [first[2], second[2]]})
        if first[1] != second[0]:
            errors.append({"subject_id": subject_id, "error": "non-contiguous runs"})
        subjects.append({
            "subject_id": subject_id, "subject_name": name,
            "class0": summarize_subject(x, y, subject_id, name, first[0], first[1], args.sampling_rate),
            "class1": summarize_subject(x, y, subject_id, name, second[0], second[1], args.sampling_rate),
        })

    flat = []
    for row in subjects:
        combined = {key: row["class0"][key] for key in ("subject_id", "subject_name", "windows", "nonfinite_samples", "constant_windows", "zero_windows", "rms_mean", "rms_std", "relative_band_power")}
        combined["label_counts"] = row["class0"]["label_counts"]
        combined["class0_windows"] = row["class0"]["windows"]
        combined["class1_windows"] = row["class1"]["windows"]
        combined["class1_rms_mean"] = row["class1"]["rms_mean"]
        combined["class_delta_rms"] = row["class1"]["rms_mean"] - row["class0"]["rms_mean"]
        flat.append(combined)

    fold5 = [row for row in flat if row["subject_id"] in FOLD_5]
    other = [row for row in flat if row["subject_id"] not in FOLD_5]
    def avg(rows, key):
        return float(np.mean([row[key] for row in rows]))
    report = {
        "data": str(root), "source_splits": ["train", "val"], "test_accessed": False,
        "shape": list(x.shape), "label_counts": np.bincount(y, minlength=2).tolist(),
        "n_subjects": len(names), "run_count": len(runs), "errors": errors,
        "fold_5_subject_ids": sorted(FOLD_5),
        "fold_5_subjects": fold5, "other_subjects": other,
        "comparison": {
            "fold5_mean_windows": avg(fold5, "windows"),
            "other_mean_windows": avg(other, "windows"),
            "fold5_mean_rms": avg(fold5, "rms_mean"),
            "other_mean_rms": avg(other, "rms_mean"),
            "fold5_mean_class_delta_rms": avg(fold5, "class_delta_rms"),
            "other_mean_class_delta_rms": avg(other, "class_delta_rms"),
            "fold5_nonfinite_samples": int(sum(row["nonfinite_samples"] for row in fold5)),
            "fold5_constant_windows": int(sum(row["constant_windows"] for row in fold5)),
            "fold5_zero_windows": int(sum(row["zero_windows"] for row in fold5)),
        },
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "errors": errors, "comparison": report["comparison"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
