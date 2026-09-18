import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from attention_model.config import AttentionConfig
from scripts.evaluate import build_model_for_evaluation


def test_evaluation_model_matches_frozen_b1_checkpoint():
    config = AttentionConfig.from_yaml(
        "artifacts/validation_B1_best_hparams_seeds25/config.yaml"
    )
    model = build_model_for_evaluation(config)
    checkpoint = torch.load(
        "artifacts/validation_B1_best_hparams_seeds25/seed_42/best_model.pt",
        map_location="cpu",
    )

    model.load_state_dict(checkpoint, strict=True)
