import pytest
import torch

from av_contradiction.config import ModelConfig
from av_contradiction.evaluation.ablations import PRESETS
from av_contradiction.models.classifier import AVContradictionModel
from av_contradiction.models.cross_attention import BidirectionalCrossAttention
from av_contradiction.training.losses import build_loss

DA, DV = 7, 9


def small_cfg(**kw) -> ModelConfig:
    base = dict(hidden_dim=16, num_heads=4, num_layers=2, fusion_dim=8, fusion_rank=2, classifier_hidden=8, dropout=0.0)
    base.update(kw)
    return ModelConfig(**base)


def make_batch(b=3, ta=(4, 2, 1), tv=(5, 3, 2), seed=0):
    g = torch.Generator().manual_seed(seed)
    TA, TV = max(ta), max(tv)
    batch = {
        "audio": torch.randn(b, TA, DA, generator=g), "audio_ts": torch.zeros(b, TA, 2),
        "audio_mask": torch.zeros(b, TA, dtype=torch.bool),
        "video": torch.randn(b, TV, DV, generator=g), "video_ts": torch.zeros(b, TV),
        "video_mask": torch.zeros(b, TV, dtype=torch.bool), "label": torch.tensor([0, 1, 0])[:b],
    }
    for i in range(b):
        batch["audio_mask"][i, : ta[i]] = True
        batch["video_mask"][i, : tv[i]] = True
        batch["audio_ts"][i, : ta[i], 0] = torch.arange(ta[i]) * 0.5
        batch["audio_ts"][i, : ta[i], 1] = torch.arange(ta[i]) * 0.5 + 0.5
        batch["video_ts"][i, : tv[i]] = torch.arange(tv[i]) * 0.4 + 0.2
    return batch


@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_forward_backward_all_presets(preset):
    cfg = small_cfg(**PRESETS[preset])
    m = AVContradictionModel(cfg, DA, DV)
    out = m(make_batch())
    assert out["logits"].shape == (3, 2)
    loss = build_loss()(out["logits"], make_batch()["label"])
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)
    assert all(torch.isfinite(g).all() for g in grads if g is not None)


def test_padding_does_not_change_output():
    m = AVContradictionModel(small_cfg(), DA, DV).eval()
    b = make_batch()
    b2 = {k: v.clone() for k, v in b.items()}
    b2["audio"][~b2["audio_mask"]] = 1e3  # garbage in padded positions
    b2["video"][~b2["video_mask"]] = -1e3
    with torch.no_grad():
        assert torch.allclose(m(b)["logits"], m(b2)["logits"], atol=1e-5)


def test_missing_modality_is_finite():
    m = AVContradictionModel(small_cfg(), DA, DV).eval()
    b = make_batch()
    b["audio_mask"][1] = False  # sample 1 has no audio
    b["video_mask"][2] = False  # sample 2 has no video
    with torch.no_grad():
        out = m(b, return_attention=True)
    assert torch.isfinite(out["logits"]).all()
    a2v = out["attn_a2v"][-1]
    assert torch.allclose(a2v[2, :, 0], torch.ones_like(a2v[2, :, 0]))  # all mass on null token when video missing


def test_attention_weights_respect_masks():
    x = BidirectionalCrossAttention(16, 4, 1, dropout=0.0).eval()
    a, v = torch.randn(2, 3, 16), torch.randn(2, 4, 16)
    ma = torch.tensor([[True, True, False], [True, True, True]])
    mv = torch.tensor([[True, False, False, False], [True, True, True, True]])
    out = x(a, ma, v, mv, need_weights=True)
    w = out["attn_a2v"][0]  # [B, Ta, 1+Tv]
    assert torch.allclose(w.sum(-1), torch.ones(2, 3), atol=1e-5)
    assert (w[0, :, 2:] == 0).all()  # padded video keys (cols 2..4) get zero weight


def test_sdpa_and_explicit_attention_agree():
    x = BidirectionalCrossAttention(16, 4, 2, dropout=0.0).eval()
    a, v = torch.randn(2, 3, 16), torch.randn(2, 5, 16)
    ma, mv = torch.ones(2, 3, dtype=torch.bool), torch.tensor([[True] * 5, [True, True, False, False, False]])
    bias = -torch.rand(2, 3, 5)
    o1 = x(a, ma, v, mv, bias, need_weights=False)
    o2 = x(a, ma, v, mv, bias, need_weights=True)
    assert torch.allclose(o1["audio"], o2["audio"], atol=1e-5) and torch.allclose(o1["video"], o2["video"], atol=1e-5)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_forward_amp():
    m = AVContradictionModel(small_cfg(), DA, DV).cuda()
    b = {k: v.cuda() for k, v in make_batch().items()}
    with torch.autocast("cuda", dtype=torch.float16):
        out = m(b)
    assert torch.isfinite(out["logits"]).all()
