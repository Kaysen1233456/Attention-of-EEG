import numpy as np

from scripts.audit_fold2_subjects import (
    audit_split_arrays,
    classify_prediction_direction,
)


def test_audit_split_arrays_reports_trial_label_conflicts_and_zero_channels():
    waveforms = np.zeros((4, 2, 3), dtype=np.float32)
    waveforms[1, 0, :] = 1.0
    subjects = np.array([3, 3, 6, 6])
    trials = np.array([10, 10, 20, 20])
    labels = np.array([0, 1, 0, 0])

    report = audit_split_arrays(waveforms, labels, subjects, trials)

    assert report["n_windows"] == 4
    assert report["trial_label_conflicts"] == 1
    assert report["zero_channel_fraction"] == 0.875
    assert report["subjects"]["3"]["n_windows"] == 2


def test_prediction_direction_marks_inverted_subject():
    assert classify_prediction_direction(0.25, 0.30, 0.50) == "systematic_inversion_candidate"
    assert classify_prediction_direction(0.50, 0.50, 0.50) == "non_directional"
    assert classify_prediction_direction(0.75, 0.70, 0.70) == "aligned_candidate"
