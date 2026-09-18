"""Convert released Ear-SAAD MATLAB files to EEGWindowDataset NPY files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import scipy.io as sio


MODALITY_CHANNELS = {
    "scalp": ["O1", "Pz", "P3", "P7", "CP1", "CP5", "M1", "Cz", "C3", "T7", "FC1", "FC5", "F3", "F7", "Fp1", "O2", "P4", "P8", "CP2", "CP6", "M2", "C4", "T8", "FC2", "FC6", "Fz", "F4", "F8", "Fp2"],
    "in_ear": ["ELA", "ELB", "ELC", "ELE", "ELI", "ELT", "ERA", "ERB", "ERC", "ERE", "ERI", "ERT"],
    "around_ear": ["cEL1", "cEL2", "cEL3", "cEL4", "cEL5", "cEL6", "cEL7", "cEL8", "cEL9", "cER1", "cER2", "cER3", "cER4", "cER5", "cER6", "cER7", "cER8", "cER9", "cER10"],
}


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("data/raw/ear_saad/preprocessedData/preprocessedData"))
    p.add_argument("--output", type=Path, default=Path("data/processed/ear_saad"))
    p.add_argument("--modality", choices=sorted(MODALITY_CHANNELS), default="around_ear")
    p.add_argument("--window-seconds", type=float, default=5.0)
    p.add_argument("--stride-seconds", type=float, default=5.0)
    p.add_argument("--discard-seconds", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def names(value: np.ndarray) -> List[str]:
    result = []
    for item in np.asarray(value, dtype=object).reshape(-1):
        if isinstance(item, np.ndarray):
            item = item.reshape(-1)[0] if item.size == 1 else "".join(str(x) for x in item.reshape(-1))
        result.append(str(item).strip())
    return result


def load(path: Path):
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    trials = [np.asarray(x, dtype=np.float32) for x in np.asarray(m["eegTrials"], dtype=object).reshape(-1)]
    return (names(m["channels"]), int(np.asarray(m["fs"]).reshape(-1)[0]), trials,
            np.asarray(m["attSpeaker"]).reshape(-1).astype(np.int64),
            np.asarray(m["attendedEar"]).reshape(-1).astype(np.int64),
            np.asarray(m["videoCondition"]).reshape(-1).astype(np.int64))


def repair_selected_signal(signal: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """Interpolate gaps per channel and zero channels with no finite sample."""
    repaired = signal.astype(np.float32, copy=True)
    missing = int(np.isnan(repaired).sum())
    fully_missing = 0
    for channel in range(repaired.shape[1]):
        values = repaired[:, channel]
        finite = np.isfinite(values)
        if not finite.any():
            repaired[:, channel] = 0.0
            fully_missing += 1
            continue
        if not finite.all():
            positions = np.arange(values.size)
            repaired[:, channel] = np.interp(positions, positions[finite], values[finite])
    if not np.isfinite(repaired).all():
        raise ValueError("Non-finite values remain after channel repair")
    return repaired, missing, fully_missing


def split(subjects: Sequence[int], seed: int) -> Dict[str, List[int]]:
    values = np.asarray(sorted(subjects), dtype=int)
    np.random.default_rng(seed).shuffle(values)
    return {"train": values[:9].tolist(), "val": values[9:12].tolist(), "test": values[12:].tolist()}


def main() -> None:
    a = args()
    files = sorted(a.input.glob("dataSubject*.mat"), key=lambda p: int(p.stem[11:]))
    if not files:
        raise FileNotFoundError(f"No dataSubject*.mat files found in {a.input}")
    if a.window_seconds <= 0 or a.stride_seconds <= 0 or a.discard_seconds < 0:
        raise ValueError("window/stride must be positive and discard must not be negative")
    subjects = [int(p.stem[11:]) for p in files]
    subject_split = split(subjects, a.seed)
    subject_to_split = {s: group for group, ids in subject_split.items() for s in ids}
    selected = MODALITY_CHANNELS[a.modality]
    store: Dict[str, List] = {k: [] for k in ("train", "val", "test", "train_labels", "val_labels", "test_labels", "train_subjects", "val_subjects", "test_subjects", "train_trials", "val_trials", "test_trials")}
    reference_channels = None
    reference_fs = None
    raw_labels, video_values = set(), set()
    quality = {group: {"trials": 0, "nan_samples_repaired": 0, "fully_missing_channels": 0} for group in ("train", "val", "test")}
    for path in files:
        subject = int(path.stem[11:])
        channels, fs, trials, labels, attended_ear, video = load(path)
        if reference_channels is None:
            reference_channels, reference_fs = channels, fs
        if channels != reference_channels or fs != reference_fs:
            raise ValueError(f"Inconsistent channel list or fs in {path.name}")
        missing = [x for x in selected if x not in channels]
        if missing:
            raise ValueError(f"Missing channels in {path.name}: {missing}")
        if not (len(trials) == len(labels) == len(attended_ear) == len(video)):
            raise ValueError(f"Trial metadata lengths differ in {path.name}")
        if not np.array_equal(labels, attended_ear):
            raise ValueError(f"attSpeaker and attendedEar differ in {path.name}")
        indices = [channels.index(x) for x in selected]
        window = int(round(a.window_seconds * fs))
        stride = int(round(a.stride_seconds * fs))
        discard = int(round(a.discard_seconds * fs))
        group = subject_to_split[subject]
        for trial_id, (trial, raw_label, video_condition) in enumerate(zip(trials, labels, video), 1):
            if trial.ndim != 2 or trial.shape[1] != len(channels):
                raise ValueError(f"Invalid trial {trial_id} in {path.name}: shape={trial.shape}")
            if int(raw_label) not in (1, 2):
                raise ValueError(f"Unexpected label {raw_label} in {path.name}")
            signal, nan_count, fully_missing = repair_selected_signal(trial[discard:, indices])
            signal = signal.T
            quality[group]["trials"] += 1
            quality[group]["nan_samples_repaired"] += nan_count
            quality[group]["fully_missing_channels"] += fully_missing
            if signal.shape[1] < window:
                raise ValueError(f"Trial {trial_id} in {path.name} is shorter than one window")
            for start in range(0, signal.shape[1] - window + 1, stride):
                store[group].append(signal[:, start:start + window].copy())
                store[f"{group}_labels"].append(int(raw_label) - 1)
                store[f"{group}_subjects"].append(subject)
                store[f"{group}_trials"].append(trial_id)
            raw_labels.add(int(raw_label))
            video_values.add(int(video_condition))
    a.output.mkdir(parents=True, exist_ok=True)
    for group in ("train", "val", "test"):
        np.save(a.output / f"{group}_waveforms.npy", np.stack(store[group]).astype(np.float32))
        for field in ("labels", "subjects", "trials"):
            np.save(a.output / f"{group}_{field}.npy", np.asarray(store[f"{group}_{field}"], dtype=np.int64))
    metadata = {"dataset": "Ear-SAAD", "modality": a.modality, "channel_names": selected, "source_channel_names": reference_channels, "sampling_rate_hz": reference_fs, "window_seconds": a.window_seconds, "stride_seconds": a.stride_seconds, "discard_seconds": a.discard_seconds, "label_mapping": {"0": "attend speaker 1", "1": "attend speaker 2"}, "raw_label_values": sorted(raw_labels), "video_condition_values": sorted(video_values), "subject_split": subject_split, "leakage_control": "subject-disjoint; windows stay within trials", "quality": quality, "preprocessing_note": "Released files are already filtered and downsampled to 20 Hz; local NaN gaps are linearly interpolated; fully missing selected channels are zero-filled and counted in quality."}
    (a.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({g: {"samples": len(store[g]), "subjects": subject_split[g]} for g in subject_split}, indent=2))


if __name__ == "__main__":
    main()
