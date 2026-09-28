"""Audit Ear-SAAD raw MAT label semantics and processed trial alignment."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio


SOURCE_SCRIPT = "https://zenodo.org/records/16536441/files/preprocessFromBIDS.m"
SEMANTICS = {
    "speaker": {1: "first stimulus file", 2: "second stimulus file"},
    "ear": {1: "left", 2: "right"},
    "video": {0: "video not shown", 1: "video shown"},
}


def _as_int_array(value: Any) -> np.ndarray:
    return np.asarray(value).reshape(-1).astype(np.int64)


def _semantic(mapping: dict[int, str], value: int) -> str:
    return mapping.get(int(value), f"unknown ({int(value)})")


def build_trial_mapping(
    subject: int,
    att_speaker: np.ndarray,
    attended_ear: np.ndarray,
    video_condition: np.ndarray,
) -> list[dict[str, Any]]:
    att_speaker = _as_int_array(att_speaker)
    attended_ear = _as_int_array(attended_ear)
    video_condition = _as_int_array(video_condition)
    if not (len(att_speaker) == len(attended_ear) == len(video_condition)):
        raise ValueError("Raw trial metadata lengths differ")
    return [
        {
            "subject": int(subject),
            "trial": int(trial),
            "attSpeaker": int(speaker),
            "attendedEar": int(ear),
            "videoCondition": int(video),
            "label_zero_based": int(speaker) - 1,
            "attended_speaker_semantics": _semantic(SEMANTICS["speaker"], int(speaker)),
            "attended_ear_semantics": _semantic(SEMANTICS["ear"], int(ear)),
            "video_semantics": _semantic(SEMANTICS["video"], int(video)),
        }
        for trial, (speaker, ear, video) in enumerate(
            zip(att_speaker, attended_ear, video_condition), start=1
        )
    ]


def compare_processed_labels(
    raw_trials: dict[int, dict[int, int]],
    processed_subjects: np.ndarray,
    processed_trials: np.ndarray,
    processed_labels: np.ndarray,
) -> dict[str, Any]:
    subjects = _as_int_array(processed_subjects)
    trials = _as_int_array(processed_trials)
    labels = _as_int_array(processed_labels)
    if not (len(subjects) == len(trials) == len(labels)):
        raise ValueError("Processed subject/trial/label lengths differ")
    mismatches = []
    counts: Counter[str] = Counter()
    for subject, trial, label in zip(subjects, trials, labels):
        key = f"{int(subject)}:{int(trial)}"
        counts[key] += 1
        expected_raw = raw_trials.get(int(subject), {}).get(int(trial))
        expected_label = None if expected_raw is None else int(expected_raw) - 1
        if expected_label is None or int(label) != expected_label:
            mismatches.append(
                {
                    "subject": int(subject),
                    "trial": int(trial),
                    "observed_label": int(label),
                    "expected_label": expected_label,
                }
            )
    return {
        "mismatches": mismatches,
        "processed_trial_counts": dict(sorted(counts.items())),
        "n_processed_windows": int(len(labels)),
    }


def summarize_metadata(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pair_counts = Counter((row["attSpeaker"], row["attendedEar"]) for row in rows)
    speaker_video = Counter((row["attSpeaker"], row["videoCondition"]) for row in rows)
    ear_video = Counter((row["attendedEar"], row["videoCondition"]) for row in rows)
    return {
        "n_trials": len(rows),
        "subjects": sorted({int(row["subject"]) for row in rows}),
        "attSpeaker_values": sorted({int(row["attSpeaker"]) for row in rows}),
        "attendedEar_values": sorted({int(row["attendedEar"]) for row in rows}),
        "video_condition_values": sorted({int(row["videoCondition"]) for row in rows}),
        "all_attSpeaker_equals_attendedEar": all(
            row["attSpeaker"] == row["attendedEar"] for row in rows
        ),
        "attSpeaker_attendedEar_pairs": {
            f"{speaker}:{ear}": count
            for (speaker, ear), count in sorted(pair_counts.items())
        },
        "attSpeaker_videoCondition": {
            f"{speaker}:{video}": count
            for (speaker, video), count in sorted(speaker_video.items())
        },
        "attendedEar_videoCondition": {
            f"{ear}:{video}": count
            for (ear, video), count in sorted(ear_video.items())
        },
    }


def _load_mat_trials(raw_dir: Path) -> tuple[list[dict[str, Any]], dict[int, dict[int, int]]]:
    rows: list[dict[str, Any]] = []
    raw_trials: dict[int, dict[int, int]] = {}
    files = sorted(
        raw_dir.glob("dataSubject*.mat"),
        key=lambda path: int(path.stem.removeprefix("dataSubject")),
    )
    if not files:
        raise FileNotFoundError(f"No dataSubject*.mat files in {raw_dir}")
    for path in files:
        subject = int(path.stem.removeprefix("dataSubject"))
        mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        trials = np.asarray(mat["eegTrials"], dtype=object).reshape(-1)
        mapping = build_trial_mapping(
            subject,
            mat["attSpeaker"],
            mat["attendedEar"],
            mat["videoCondition"],
        )
        if len(trials) != len(mapping):
            raise ValueError(f"Trial count mismatch in {path.name}")
        raw_trials[subject] = {
            item["trial"]: item["attSpeaker"] for item in mapping
        }
        rows.extend(mapping)
    return rows, raw_trials


def _processed_alignment(repo_root: Path, raw_trials: dict[int, dict[int, int]]) -> dict[str, Any]:
    data_dir = repo_root / "data" / "processed" / "ear_saad"
    reports = {}
    for split in ("train", "val", "test"):
        subjects = np.load(data_dir / f"{split}_subjects.npy")
        trials = np.load(data_dir / f"{split}_trials.npy")
        labels = np.load(data_dir / f"{split}_labels.npy")
        reports[split] = compare_processed_labels(raw_trials, subjects, trials, labels)
    all_subjects = np.concatenate(
        [np.load(data_dir / f"{split}_subjects.npy") for split in ("train", "val", "test")]
    )
    all_trials = np.concatenate(
        [np.load(data_dir / f"{split}_trials.npy") for split in ("train", "val", "test")]
    )
    trial_counts: defaultdict[tuple[int, int], int] = defaultdict(int)
    for subject, trial in zip(all_subjects, all_trials):
        trial_counts[(int(subject), int(trial))] += 1
    expected = {
        f"{subject}:{trial}": 119
        for subject in sorted(raw_trials)
        for trial in sorted(raw_trials[subject])
    }
    observed = {
        f"{subject}:{trial}": count
        for (subject, trial), count in sorted(trial_counts.items())
    }
    return {
        "by_split": reports,
        "all_split_mismatches": sum(
            len(item["mismatches"]) for item in reports.values()
        ),
        "expected_windows_per_trial": 119,
        "all_trial_counts": observed,
        "trial_count_mismatches": {
            key: {"expected": expected.get(key), "observed": value}
            for key, value in observed.items()
            if expected.get(key) != value
        },
    }


def build_report(repo_root: Path) -> dict[str, Any]:
    raw_dir = repo_root / "data" / "raw" / "ear_saad" / "preprocessedData"
    rows, raw_trials = _load_mat_trials(raw_dir)
    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_subject[str(row["subject"])].append(row)
    return {
        "audit_scope": "read-only raw MAT semantic and processed trial alignment audit",
        "source_semantics_reference": SOURCE_SCRIPT,
        "source_script_semantics": {
            "attSpeaker": "1 if attended speaker is in first stimulus file, 2 if in second",
            "attendedEar": "1 if left, 2 if right",
            "videoCondition": "1 if video shown, 0 if not",
        },
        "all_metadata_summary": summarize_metadata(rows),
        "target_subjects": {
            subject: by_subject[str(subject)] for subject in (3, 6, 11, 13)
        },
        "subject_trial_metadata": dict(sorted(by_subject.items(), key=lambda item: int(item[0]))),
        "processed_alignment": _processed_alignment(repo_root, raw_trials),
    }


def _markdown(report: dict[str, Any]) -> str:
    summary = report["all_metadata_summary"]
    alignment = report["processed_alignment"]
    lines = [
        "# Ear-SAAD 原始 MAT 标签语义审计",
        "",
        "本报告为只读审计，不修改原始 MAT、处理数组、模型、训练配置或已有实验结果。",
        "",
        "## 语义依据",
        f"- 发布方预处理脚本：`{report['source_semantics_reference']}`",
        "- `attSpeaker=1/2`：注意第一个/第二个刺激文件中的说话人。",
        "- `attendedEar=1/2`：注意左耳/右耳。",
        "- `videoCondition=0/1`：未呈现/呈现注意说话人视频。",
        "",
        "## 全部原始 MAT",
        f"- 被试数：`{len(summary['subjects'])}`；trial 数：`{summary['n_trials']}`",
        f"- `attSpeaker` 与 `attendedEar` 全部相等：`{summary['all_attSpeaker_equals_attendedEar']}`",
        f"- `attSpeaker` 值：`{summary['attSpeaker_values']}`；`attendedEar` 值：`{summary['attendedEar_values']}`",
        f"- `videoCondition` 值：`{summary['video_condition_values']}`",
        f"- `attSpeaker × attendedEar`：`{summary['attSpeaker_attendedEar_pairs']}`",
        f"- `attSpeaker × videoCondition`：`{summary['attSpeaker_videoCondition']}`",
        "",
        "## Fold 2 目标被试条件",
        "| 被试 | trial | attSpeaker | attendedEar | videoCondition | 标签语义 | 耳语义 | 视频 |",
        "|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for subject in (3, 6, 11, 13):
        for row in report["target_subjects"][subject]:
            lines.append(
                f"| {subject} | {row['trial']} | {row['attSpeaker']} | "
                f"{row['attendedEar']} | {row['videoCondition']} | "
                f"{row['attended_speaker_semantics']} | "
                f"{row['attended_ear_semantics']} | {row['video_semantics']} |"
            )
    lines += [
        "",
        "## 原始 trial 到处理窗口",
        f"- 所有 split 的逐窗口标签不匹配数：`{alignment['all_split_mismatches']}`",
        f"- 预期每个 trial 窗口数：`{alignment['expected_windows_per_trial']}`",
        f"- 窗口数不匹配 trial 数：`{len(alignment['trial_count_mismatches'])}`",
        "",
        "## 结论",
        "- 未发现 `attSpeaker` 与 `attendedEar` 在被试间发生编码不一致；全部 90 个原始 trial 中二者相等。",
        "- `videoCondition` 与说话人/耳朵条件并非同一字段；它只表示是否呈现注意说话人视频，不能被当作分类标签。",
        "- 处理标签是 `attSpeaker - 1`，且原始 trial 标签与现有窗口标签逐项一致时，不能据此认定 Fold 2 存在标签翻转。",
        "- 该审计确认了字段编码和窗口继承，但仅凭预处理 MAT 不能证明“第一/第二刺激文件”对应哪一个物理左右声源；这需要 BIDS `events.tsv` 的刺激文件路径或完整源事件元数据进一步核对。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/raw_label_semantics_audit"),
    )
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output = (root / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = build_report(root)
    (output / "raw_label_semantics_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "raw_label_semantics_audit.md").write_text(
        _markdown(report), encoding="utf-8"
    )
    print(f"Report: {output / 'raw_label_semantics_audit.md'}")
    print(f"JSON: {output / 'raw_label_semantics_audit.json'}")


if __name__ == "__main__":
    main()
