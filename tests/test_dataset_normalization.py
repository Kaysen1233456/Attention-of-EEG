import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from attention_model.data import EEGWindowDataset


class EEGWindowDatasetNormalizationTests(unittest.TestCase):
    def test_window_local_normalization_uses_each_channel_window_statistics(self):
        dataset = EEGWindowDataset(
            waveforms=np.array([[[1.0, 3.0], [10.0, 14.0]]], dtype=np.float32),
            labels=np.array([0]),
            normalize=True,
            normalization_mode="window_local",
        )

        waveform = dataset[0]["waveform"].numpy()

        np.testing.assert_allclose(
            waveform,
            np.array([[-1.0, 1.0], [-1.0, 1.0]], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_zero_after_normalization_removes_zero_channel_proxy_signal(self):
        dataset = EEGWindowDataset(
            waveforms=np.array([[[0.0, 0.0], [2.0, 4.0]]], dtype=np.float32),
            labels=np.array([0]),
            normalize=True,
            mean=np.array([[1.0], [3.0]], dtype=np.float32),
            std=np.array([[1.0], [1.0]], dtype=np.float32),
            zero_channel_policy="zero_after_normalization",
        )

        waveform = dataset[0]["waveform"].numpy()

        np.testing.assert_allclose(
            waveform,
            np.array([[0.0, 0.0], [-1.0, 1.0]], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
