"""Audio-swap benchmark: balance, timeline handling, and that a single modality cannot solve it."""

import pytest
import torch

from av_contradiction.config import replace_section
from av_contradiction.data.dataset import CachedFeatureDataset
from av_contradiction.data.features import cache_path
from av_contradiction.data.manifest import write_manifest
from av_contradiction.data.swaps import build_audio_swaps, check_swap_balance
from av_contradiction.evaluation.ablations import run_suite
from tests.conftest import tiny_config


def _save(cfg, sid, audio, video, silent=None):
    n = audio.shape[0]
    p = cache_path(cfg, sid)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "audio": audio, "video": video,
        "audio_ts": torch.stack([torch.arange(n) * 1.0, torch.arange(n) * 1.0 + 2]).T.float() + 0.01 * n,
        "video_ts": torch.linspace(0, n, video.shape[0]) + 0.01 * n,
        "audio_silent": torch.zeros(n, dtype=torch.bool) if silent is None else silent,
        "duration_sec": float(n + 1),
    }, p)


def _clips(cfg, splits=(("train", 12), ("val", 6), ("test", 6)), lengths=(3, 4), dim=8, seed=0, coupled=False):
    """Consistent clips with cached features. ``coupled``: the video embedding is a fixed function of the
    clip's own audio, so a model that compares the two can tell swaps apart."""
    g = torch.Generator().manual_seed(seed)
    mix = torch.randn(dim, dim, generator=g)
    rows = []
    for split, n in splits:
        for i in range(n):
            sid = f"{split}{i}"
            n_win = lengths[i % len(lengths)]
            key = torch.randn(dim, generator=g)
            audio = key + 0.1 * torch.randn(n_win, dim, generator=g)
            video = (key @ mix if coupled else torch.randn(dim, generator=g)) + 0.1 * torch.randn(4, dim, generator=g)
            _save(cfg, sid, audio, video)
            rows.append({"sample_id": sid, "video_path": f"{sid}.mp4", "label": "consistent", "split": split, "group_id": sid})
    return rows


def test_swaps_are_balanced_and_split_internal(tmp_path):
    cfg = tiny_config(tmp_path)
    rows = _clips(cfg)
    # a clip with no same-length partner and a silent clip are dropped
    _save(cfg, "lonely", torch.randn(9, 8), torch.randn(4, 8))
    _save(cfg, "quiet", torch.randn(3, 8), torch.randn(4, 8), silent=torch.ones(3, dtype=torch.bool))
    rows += [{"sample_id": s, "video_path": f"{s}.mp4", "label": "consistent", "split": "train"} for s in ("lonely", "quiet")]
    for pairing in ("random", "hard"):
        out, rep = build_audio_swaps(cfg, rows, pairing, seed=1)
        assert rep["balance_check"]["ok"], rep["balance_check"]
        assert rep["by_split"] == {"test": {"consistent": 6, "contradictory": 6}, "train": {"consistent": 12, "contradictory": 12},
                                   "val": {"consistent": 6, "contradictory": 6}}
        assert rep["dropped"] == {"no partner of the same length": 1, "mostly silent audio": 1}
        by_id = {r["sample_id"]: r for r in out}
        for r in out:
            if r["label"] == "contradictory":
                assert r["video_feature_id"] != r["audio_feature_id"]
                assert by_id[r["audio_feature_id"]]["split"] == r["split"]


def test_balance_check_catches_reused_audio():
    rows = [
        {"sample_id": "a", "label": "consistent", "split": "train"},
        {"sample_id": "b", "label": "consistent", "split": "train"},
        {"sample_id": "swap1", "label": "contradictory", "split": "train", "video_feature_id": "a", "audio_feature_id": "b"},
        {"sample_id": "swap2", "label": "contradictory", "split": "train", "video_feature_id": "b", "audio_feature_id": "b"},
    ]
    assert not check_swap_balance(rows)["ok"]


def test_hard_pairing_picks_similar_audio(tmp_path):
    cfg = tiny_config(tmp_path)
    base = torch.eye(8)
    # clips 0/1 have near-identical audio, as do 2/3; hard pairing must swap within those pairs
    for i, k in enumerate((0, 0, 1, 1)):
        _save(cfg, f"c{i}", base[k] + 0.01 * i + torch.zeros(3, 8), torch.randn(4, 8))
    rows = [{"sample_id": f"c{i}", "video_path": "x", "label": "consistent", "split": "train"} for i in range(4)]
    out, _ = build_audio_swaps(cfg, rows, "hard")
    pairs = {r["video_feature_id"]: r["audio_feature_id"] for r in out if r["label"] == "contradictory"}
    assert pairs == {"c0": "c1", "c1": "c0", "c2": "c3", "c3": "c2"}


def test_swapped_row_keeps_video_timeline(tmp_path):
    cfg = tiny_config(tmp_path)
    _save(cfg, "a", torch.zeros(3, 8), torch.ones(4, 8))
    _save(cfg, "b", torch.full((3, 8), 5.0), torch.full((4, 8), 7.0), silent=torch.tensor([True, False, False]))
    _save(cfg, "short", torch.zeros(2, 8), torch.ones(4, 8))
    rows = [{"sample_id": "s", "video_path": "a.mp4", "label": "contradictory", "video_feature_id": "a", "audio_feature_id": "b"},
            {"sample_id": "bad", "video_path": "a.mp4", "label": "contradictory", "video_feature_id": "a", "audio_feature_id": "short"}]
    ds = CachedFeatureDataset(cfg, rows)
    a = torch.load(cache_path(cfg, "a"), weights_only=True)
    it = ds[0]
    assert torch.equal(it["video"], a["video"]) and torch.equal(it["video_ts"], a["video_ts"])
    assert torch.equal(it["audio_ts"], a["audio_ts"])
    assert it["audio"].eq(5.0).all() and it["audio_silent"].tolist() == [True, False, False]
    with pytest.raises(ValueError, match="windows"):
        ds[1]


def test_single_modality_is_at_chance_but_fusion_is_not(tmp_path):
    """The point of the benchmark: by construction a unimodal model cannot beat chance, while a model
    that compares audio and video can."""
    cfg = tiny_config(tmp_path)
    rows = _clips(cfg, splits=(("train", 160), ("val", 40), ("test", 60)), coupled=True)
    out, rep = build_audio_swaps(cfg, rows, "random", seed=0)
    assert rep["balance_check"]["ok"]
    write_manifest(out, cfg.data.manifest)
    cfg = replace_section(cfg, training={"epochs": 40, "early_stopping_patience": 40, "lr": 3e-3})
    cfg.experiment_name = "swap_test"
    res = run_suite(cfg, ["audio_only", "video_only", "concat"], seeds=[0])
    f1 = {r["preset"]: r["metrics"]["macro_f1"] for r in res["runs"]}
    assert f1["audio_only"] < 0.7 and f1["video_only"] < 0.7
    assert f1["concat"] > 0.8
