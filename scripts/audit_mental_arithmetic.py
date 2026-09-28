"""Audit subject and spectral distribution shifts in mental-arithmetic data."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import welch


BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def summarize(x, y, subject_names, sampling_rate):
    frequencies, psd = welch(x, fs=sampling_rate, nperseg=min(250, x.shape[-1]), axis=-1)
    total_mask = (frequencies >= 1.0) & (frequencies <= 45.0)
    total_power = np.trapz(psd[..., total_mask], frequencies[total_mask], axis=-1) + 1e-12
    rms = np.sqrt(np.mean(x * x, axis=-1))
    starts = np.r_[0, np.flatnonzero(y[1:] != y[:-1]) + 1, len(y)]
    runs = list(zip(starts[:-1], starts[1:]))
    if len(runs) != len(subject_names) * 2:
        raise ValueError(f"Expected two class runs per subject, got {len(runs)} runs")

    subjects = []
    for index, name in enumerate(subject_names):
        subject_row = {"subject": name, "classes": {}}
        for label, (start, end) in zip((0, 1), runs[index * 2:index * 2 + 2]):
            band_values = {}
            for band, (low, high) in BANDS.items():
                mask = (frequencies >= low) & (frequencies < high)
                power = np.trapz(psd[start:end, :, mask], frequencies[mask], axis=-1)
                band_values[band] = float(np.mean(power / total_power[start:end]))
            subject_row["classes"][str(label)] = {
                "n_windows": int(end - start),
                "rms_mean": float(np.mean(rms[start:end])),
                "rms_std": float(np.std(rms[start:end])),
                "channel_std_mean": float(np.mean(np.std(x[start:end], axis=-1))),
                "relative_band_power": band_values,
            }
        subjects.append(subject_row)
    return subjects


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/mental_arithmetic_250hz_8ch")
    parser.add_argument("--output", default="artifacts/mental_arithmetic_audit.json")
    args = parser.parse_args()
    root = Path(args.data)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    report = {"data": str(root), "sampling_rate": metadata["sampling_rate_hz"], "splits": {}}
    for split in ("train", "val", "test"):
        x = np.load(root / f"{split}_waveforms.npy").astype(np.float32)
        y = np.load(root / f"{split}_labels.npy").astype(np.int64)
        names = metadata[f"{split}_subjects"]
        report["splits"][split] = {
            "shape": list(x.shape),
            "label_counts": {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
            "subjects": summarize(x, y, names, metadata["sampling_rate_hz"]),
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
