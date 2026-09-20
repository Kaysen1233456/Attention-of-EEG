
import os
import numpy as np
import pandas as pd
import mne
from pathlib import Path
from tqdm import tqdm
import json

DATA_ROOT = Path("/mnt/workspace/Attention-of-EEG/ds007169-download")
OUTPUT_DIR = Path("/mnt/workspace/Attention-of-EEG/data/processed/nback")
WINDOW_SEC = 2.0
SFREQ = 250.0
N_CHANNELS = 19

EEG_CHANNELS = ["FP1","FP2","F7","F3","Fz","F4","F8","T3","C3","Cz","C4","T4",
                "T5","P3","Pz","P4","T6","O1","O2"]

VALID_TRIAL_TYPES = {"1-back": 0, "2-back": 1, "3-back": 2, "4-back": 3}

def process_subject(subj_dir):
    sid = subj_dir.name
    eeg_path = subj_dir / "eeg" / f"{sid}_task-nback_eeg.vhdr"
    events_path = subj_dir / "eeg" / f"{sid}_task-nback_events.tsv"
    if not eeg_path.exists() or not events_path.exists():
        print(f"  Skip {sid}: missing files")
        return None
    raw = mne.io.read_raw_brainvision(str(eeg_path), preload=True, verbose=False)
    raw.pick_channels(EEG_CHANNELS)
    raw.filter(1, 45, verbose=False)
    events_df = pd.read_csv(events_path, sep="\t")
    trial_events = events_df[events_df["trial_type"].isin(VALID_TRIAL_TYPES.keys())].copy()
    sfreq = raw.info["sfreq"]
    window_samples = int(WINDOW_SEC * sfreq)
    data = raw.get_data()
    windows, labels = [], []
    for _, row in trial_events.iterrows():
        onset_sample = int(row["onset"] * sfreq)
        end_sample = onset_sample + window_samples
        if end_sample > data.shape[1]:
            continue
        win = data[:, onset_sample:end_sample]
        windows.append(win)
        labels.append(VALID_TRIAL_TYPES[row["trial_type"]])
    if len(windows) == 0:
        print(f"  Skip {sid}: no valid trials")
        return None
    return {
        "windows": np.array(windows, dtype=np.float32),
        "labels": np.array(labels, dtype=np.int64),
        "subjects": np.array([sid] * len(windows)),
    }

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    subj_dirs = sorted([d for d in DATA_ROOT.iterdir() if d.name.startswith("sub-")])
    print(f"Found {len(subj_dirs)} subjects")
    all_windows, all_labels, all_subjects = [], [], []
    for subj_dir in tqdm(subj_dirs):
        result = process_subject(subj_dir)
        if result is not None:
            all_windows.append(result["windows"])
            all_labels.append(result["labels"])
            all_subjects.append(result["subjects"])
            print(f"  {subj_dir.name}: {len(result['windows'])} windows")
    windows = np.concatenate(all_windows, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    subjects = np.concatenate(all_subjects, axis=0)
    print(f"Total: {windows.shape}, labels: {np.bincount(labels)}")
    unique_subjs = np.unique(subjects)
    np.random.seed(42)
    np.random.shuffle(unique_subjs)
    n = len(unique_subjs)
    train_subjs = unique_subjs[:int(n*0.7)]
    val_subjs = unique_subjs[int(n*0.7):int(n*0.85)]
    test_subjs = unique_subjs[int(n*0.85):]
    def split_data(subset_subjs):
        mask = np.isin(subjects, subset_subjs)
        return windows[mask], labels[mask], subjects[mask]
    train_w, train_l, train_s = split_data(train_subjs)
    val_w, val_l, val_s = split_data(val_subjs)
    test_w, test_l, test_s = split_data(test_subjs)
    print(f"Train: {len(train_w)} ({len(train_subjs)}s), Val: {len(val_w)} ({len(val_subjs)}s), Test: {len(test_w)} ({len(test_subjs)}s)")
    np.save(OUTPUT_DIR / "train_windows.npy", train_w)
    np.save(OUTPUT_DIR / "train_labels.npy", train_l)
    np.save(OUTPUT_DIR / "train_subjects.npy", train_s)
    np.save(OUTPUT_DIR / "val_windows.npy", val_w)
    np.save(OUTPUT_DIR / "val_labels.npy", val_l)
    np.save(OUTPUT_DIR / "val_subjects.npy", val_s)
    np.save(OUTPUT_DIR / "test_windows.npy", test_w)
    np.save(OUTPUT_DIR / "test_labels.npy", test_l)
    np.save(OUTPUT_DIR / "test_subjects.npy", test_s)
    meta = {
        "n_subjects_total": len(unique_subjs),
        "n_channels": N_CHANNELS,
        "window_sec": WINDOW_SEC,
        "sfreq": SFREQ,
        "n_classes": 4,
        "channels": EEG_CHANNELS,
        "class_names": ["1-back","2-back","3-back","4-back"],
        "train_subjects": list(train_subjs),
        "val_subjects": list(val_subjs),
        "test_subjects": list(test_subjs),
    }
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
