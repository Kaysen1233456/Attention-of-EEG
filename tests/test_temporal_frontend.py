"""CPU tests; no EEG data or checkpoints required."""
import torch
from attention_model.models.mini_neuript import MiniNeurIPTClassifier


def make(frontend='none', pool=4):
    return MiniNeurIPTClassifier(d_model=24, n_heads=4, d_ff=48, n_layers=1,
        n_channels=4, dropout=0.0, use_pmoe=False, use_iilp_pooling=False,
        classifier_type='mlp', temporal_pool=pool, temporal_frontend=frontend)


def test_default_checkpoint_compatibility():
    a, b = make(), make('none')
    assert not any('temporal_filter' in k for k in a.state_dict())
    b.load_state_dict(a.state_dict(), strict=True)
    a.eval(); b.eval()
    x = torch.randn(2, 4, 100)
    torch.testing.assert_close(a(x)['logits'], b(x)['logits'])


def test_conv_shape_and_gradient():
    m = make('conv', 2)
    x = torch.randn(2, 4, 100)
    assert m.encoder.encode(x).shape == (2, 4, 50, 24)
    logits = m(x)['logits']
    assert logits.shape == (2, 2)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    grads = [p.grad for p in m.encoder.temporal_filter.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum().item() for g in grads) > 0


def test_invalid_frontend():
    import pytest
    with pytest.raises(ValueError):
        make('invalid')
