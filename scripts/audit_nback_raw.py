"""Audit ds007169 event metadata before any EEG preprocessing or training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


LEVELS = (1, 2, 3, 4)
FORMAL_MARKER_STREAM = "n-backMarkers"
FORMAL_PATTERN = r"^[1-4]-back$"


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _contiguous_runs(values: np.ndarray) -> list[dict[str, int]]:
    if values.size == 0:
        return []
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
    ends = np.r_[starts[1:], values.size]
    return [
        {
            "level": int(values[start]),
            "n_events": int(end - start),
        }
        for start, end in zip(starts, ends)
    ]


def audit_subject(events_path: Path, sfreq: float, epoch_seconds: float) -> dict:
    events = pd.read_csv(events_path, sep="\t")
    marker_mask = events["marker_stream"].eq(FORMAL_MARKER_STREAM)
    task_mask = events["trial_type"].astype("string").str.match(FORMAL_PATTERN, na=False)
    task_events = events.loc[marker_mask & task_mask].copy()
    formal_mask = task_events["istutorial"].map(_as_bool).eq(False)
    tutorial_mask = task_events["istutorial"].map(_as_bool).eq(True)
    formal = task_events.loc[formal_mask].copy()
    tutorial = task_events.loc[tutorial_mask].copy()

    dropped = events.loc[events["trial_type"].eq("dropped_samples")].copy()
    dropped_start = dropped["onset"].to_numpy(dtype=float)
    dropped_end = dropped_start + dropped["duration"].to_numpy(dtype=float)

    formal_start = formal["onset"].to_numpy(dtype=float)
    formal_end = formal_start + epoch_seconds
    overlap_count = 0
    if dropped_start.size:
        overlap_count = int(
            np.logical_or.reduce(
                [
                    np.logical_and(dropped_start[:, None] < formal_end[None, :], formal_start[None, :] < dropped_end[:, None])
                ],
                axis=0,
            ).sum()
        )

    level_counts = {
        str(level): int((formal["nback_level"] == level).sum()) for level in LEVELS
    }
    tutorial_level_counts = {
        str(level): int((tutorial["nback_level"] == level).sum()) for level in LEVELS
    }
    runs = _contiguous_runs(formal["nback_level"].to_numpy(dtype=int))
    onset_diffs = np.diff(formal_start)

    raw_duration = float(events["onset"].max() + events["duration"].max())
    out_of_bounds = int((formal_end > raw_duration).sum())

    return {
        "subject": events_path.parts[-3],
        "events_file": str(events_path),
        "all_rows": int(len(events)),
        "task_rows": int(len(task_events)),
        "formal_rows": int(len(formal)),
        "tutorial_rows": int(len(tutorial)),
        "formal_level_counts": level_counts,
        "tutorial_level_counts": tutorial_level_counts,
        "formal_level_runs": runs,
        "dropped_sample_rows": int(len(dropped)),
        "formal_epochs_overlapping_dropped_samples": overlap_count,
        "formal_epochs_out_of_bounds": out_of_bounds,
        "formal_onset_min_seconds": float(formal_start.min()) if formal_start.size else None,
        "formal_onset_max_seconds": float(formal_start.max()) if formal_start.size else None,
        "formal_onset_diff_median_seconds": float(np.median(onset_diffs)) if onset_diffs.size else None,
        "formal_onset_diff_min_seconds": float(onset_diffs.min()) if onset_diffs.size else None,
        "formal_onset_diff_max_seconds": float(onset_diffs.max()) if onset_diffs.size else None,
        "sampling_frequency": sfreq,
        "epoch_seconds": epoch_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("ds007169-download"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/nback_raw_audit_v1/audit.json"))
    parser.add_argument("--sampling-frequency", type=float, default=250.0)
    parser.add_argument("--epoch-seconds", type=float, default=1.7)
    args = parser.parse_args()

    participant_file = args.data_root / "participants.tsv"
    participants = pd.read_csv(participant_file, sep="\t")
    included = set(
        participants.loc[participants["analysis_included"].map(_as_bool), "participant_id"].astype(str)
    )
    event_files = sorted(args.data_root.glob("sub-*/eeg/*_task-nback_events.tsv"))
    reports = [
        audit_subject(path, args.sampling_frequency, args.epoch_seconds)
        for path in event_files
        if path.parts[-3] in included
    ]

    summary = {
        "data_root": str(args.data_root),
        "n_included_subjects": len(included),
        "n_event_files": len(reports),
        "formal_events_total": int(sum(item["formal_rows"] for item in reports)),
        "tutorial_events_total": int(sum(item["tutorial_rows"] for item in reports)),
        "formal_epochs_overlapping_dropped_samples": int(
            sum(item["formal_epochs_overlapping_dropped_samples"] for item in reports)
        ),
        "formal_epochs_out_of_bounds": int(sum(item["formal_epochs_out_of_bounds"] for item in reports)),
        "subjects": reports,
        "preprocessing_contract": {
            "include_tutorial": False,
            "label": "nback_level",
            "epoch_start": "event_onset",
            "epoch_seconds": args.epoch_seconds,
            "reject_epochs_overlapping_dropped_samples": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps({key: summary[key] for key in (
        "n_included_subjects",
        "n_event_files",
        "formal_events_total",
        "tutorial_events_total",
        "formal_epochs_overlapping_dropped_samples",
        "formal_epochs_out_of_bounds",
    )}, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
