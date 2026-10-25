import torch

from av_contradiction.config import ModelConfig
from av_contradiction.models.alignment import (
    TemporalProximityBias,
)
from av_contradiction.models.classifier import AVContradictionModel


def test_temporal_bias_prefers_close_keys():
    bias = TemporalProximityBias(1.0)(torch.tensor([[1.0]]), torch.tensor([[0.0, 1.0, 3.0]]))
    assert bias.shape == (1, 1, 3)
    assert bias[0, 0, 1] == 0 and bias[0, 0, 0] > bias[0, 0, 2]
    assert torch.isclose(bias[0, 0, 0], torch.tensor(-0.5))


def _batch(perm=None):
    torch.manual_seed(0)
    v = torch.randn(1, 5, 6)
    vt = torch.tensor([[0.2, 0.6, 1.0, 1.4, 1.8]])
    if perm is not None:
        v, vt = v[:, perm], vt[:, perm]
    return {
        "audio": torch.randn(1, 4, 3, generator=torch.Generator().manual_seed(1)),
        "audio_ts": torch.tensor([[[0.0, 0.5], [0.5, 1.0], [1.0, 1.5], [1.5, 2.0]]]),
        "audio_mask": torch.ones(1, 4, dtype=torch.bool),
        "video": v, "video_ts": vt, "video_mask": torch.ones(1, 5, dtype=torch.bool),
    }


def test_alignment_uses_timestamps_not_indices():
    """Reordering frames *together with* their timestamps must not change the output: correspondence
    comes from timestamps, not array positions."""
    m = AVContradictionModel(ModelConfig(hidden_dim=16, num_heads=2, num_layers=1, fusion_dim=8, classifier_hidden=8, dropout=0.0), 3, 6).eval()
    perm = torch.tensor([3, 0, 4, 1, 2])
    with torch.no_grad():
        a = m(_batch())["logits"]
        b = m(_batch(perm))["logits"]
    assert torch.allclose(a, b, atol=1e-5)


def test_timestamp_mode_is_sensitive_to_time_shift():
    m = AVContradictionModel(ModelConfig(hidden_dim=16, num_heads=2, num_layers=1, fusion_dim=8, classifier_hidden=8, dropout=0.0), 3, 6).eval()
    b1 = _batch()
    b2 = _batch()
    b2["video_ts"] = b2["video_ts"].flip(-1)  # same features, different times
    with torch.no_grad():
        assert not torch.allclose(m(b1)["logits"], m(b2)["logits"])
