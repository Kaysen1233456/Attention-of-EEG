"""Audit processed development arrays and freeze subject-disjoint folds."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/ear_saad")
    parser.add_argument("--output", default="artifacts/development_audit_v1")
    args = parser.parse_args()
    root = Path(args.data)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    arrays = {key: np.concatenate([np.load(root / f"{split}_{key}.npy")
              for split in ("train", "val")])
              for key in ("waveforms", "labels", "subjects", "trials")}
    subjects = np.unique(arrays["subjects"])
    assert len(subjects) == 12
    assert len({len(value) for value in arrays.values()}) == 1
    report = {"subjects": {}, "duplicate_windows": 0,
              "limitations": ["Zero channels are a proxy, not an original missingness mask",
                              "Speaker-side mapping and interpolation provenance require original trials"]}
    hashes = {}
    for index, waveform in enumerate(arrays["waveforms"]):
        digest = hashlib.sha256(waveform.tobytes()).hexdigest()
        if digest in hashes:
            report["duplicate_windows"] += 1
        hashes[digest] = index
    for subject in subjects:
        mask = arrays["subjects"] == subject
        waveforms = arrays["waveforms"][mask]
        labels = arrays["labels"][mask]
        trials = arrays["trials"][mask]
        inconsistent = [int(trial) for trial in np.unique(trials)
                        if len(np.unique(labels[trials == trial])) != 1]
        assert not inconsistent, (subject, inconsistent)
        report["subjects"][str(int(subject))] = {
            "windows": int(mask.sum()), "trials": int(len(np.unique(trials))),
            "label_counts": np.bincount(labels.astype(int)).tolist(),
            "nonfinite_samples": int((~np.isfinite(waveforms)).sum()),
            "zero_channel_windows": np.all(waveforms == 0, axis=2).sum(axis=0).tolist(),
            "channel_mean": waveforms.mean(axis=(0, 2)).tolist(),
            "channel_std": waveforms.std(axis=(0, 2)).tolist(),
        }
    shuffled = np.random.default_rng(42).permutation(subjects)
    folds = []
    for index, validation in enumerate(np.array_split(shuffled, 3)):
        folds.append({"fold": index, "train_subjects": sorted(set(map(int, subjects)) - set(map(int, validation))),
                      "val_subjects": sorted(map(int, validation))})
    payload = {
        "source_splits": ["train", "val"],
        "seed": 42,
        "n_subjects": int(len(subjects)),
        "folds": folds,
    }
    (output / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "folds.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"duplicates": report["duplicate_windows"], "folds": folds}, indent=2))


if __name__ == "__main__":
    main()
