import numpy as np

from scripts.audit_ear_saad_raw_labels import (
    build_trial_mapping,
    compare_processed_labels,
    summarize_metadata,
)


def test_build_trial_mapping_contains_source_semantics():
    mapping = build_trial_mapping(
        subject=3,
        att_speaker=np.array([2, 1]),
        attended_ear=np.array([2, 1]),
        video_condition=np.array([1, 0]),
    )

    assert mapping[0] == {
        "subject": 3,
        "trial": 1,
        "attSpeaker": 2,
        "attendedEar": 2,
        "videoCondition": 1,
        "label_zero_based": 1,
        "attended_speaker_semantics": "second stimulus file",
        "attended_ear_semantics": "right",
        "video_semantics": "video shown",
    }


def test_compare_processed_labels_reports_exact_trial_alignment():
    result = compare_processed_labels(
        raw_trials={3: {1: 2, 2: 1}},
        processed_subjects=np.array([3, 3, 3]),
        processed_trials=np.array([1, 1, 2]),
        processed_labels=np.array([1, 1, 0]),
    )

    assert result["mismatches"] == []
    assert result["processed_trial_counts"] == {"3:1": 2, "3:2": 1}


def test_summarize_metadata_detects_equal_speaker_and_ear():
    report = summarize_metadata(
        [
            {
                "subject": 3,
                "trial": 1,
                "attSpeaker": 2,
                "attendedEar": 2,
                "videoCondition": 1,
            },
            {
                "subject": 3,
                "trial": 2,
                "attSpeaker": 1,
                "attendedEar": 1,
                "videoCondition": 0,
            },
        ]
    )

    assert report["all_attSpeaker_equals_attendedEar"] is True
    assert report["video_condition_values"] == [0, 1]
