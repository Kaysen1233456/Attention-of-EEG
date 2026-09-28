import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from attention_model.config import AttentionConfig
from scripts.search_hyperparams import apply_search_params
from attention_model.search import BayesianOptimization


def test_apply_search_params_preserves_original_params():
    config = AttentionConfig()
    params = {
        "learning_rate": 1.2e-5,
        "dropout": 0.03,
        "rest_weight": 0.9,
        "focus_weight": 2.1,
        "trial_idx": 7,
    }
    original = params.copy()

    applied = apply_search_params(config, params)

    assert params == original
    assert applied == {"learning_rate": 1.2e-5, "dropout": 0.03, "trial_idx": 7}
    assert config.training.learning_rate == 1.2e-5
    assert config.model.dropout == 0.03
    assert config.training.class_weights == [0.9, 2.1]


def test_bayesian_discrete_candidates_round_trip_and_constant_dimension():
    search = BayesianOptimization(
        {"d_model": [48, 72, 96], "batch_size": [16, 16, "int"]},
        n_trials=1,
    )
    assert search._decode_sample(__import__("numpy").array([0.5, 0.0]))["d_model"] == 72
    assert search._decode_sample(__import__("numpy").array([0.0, 0.0]))["batch_size"] == 16
    encoded = search._encode_params({"d_model": 96, "batch_size": 16})
    assert encoded.tolist() == [1.0, 0.0]
