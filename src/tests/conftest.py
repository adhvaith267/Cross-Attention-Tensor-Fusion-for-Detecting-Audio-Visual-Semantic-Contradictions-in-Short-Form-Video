"""Shared fixtures. Everything is CPU-only, uses handcrafted encoders and tiny rendered clips;
no network access or pretrained weights are needed."""

from __future__ import annotations

from pathlib import Path

import pytest

from av_contradiction.config import Config, build_config


def tiny_config(tmp: Path, **extra) -> Config:
    raw = {
        "experiment_name": "tiny",
        "data": {"manifest": str(tmp / "manifest.csv"), "root_dir": str(tmp / "data"), "feature_cache_dir": str(tmp / "cache")},
        "preprocess": {"num_frames": 4, "frame_size": 32, "audio_sample_rate": 16000, "audio_window_sec": 0.5,
                       "audio_hop_sec": 0.5, "max_audio_windows": 8},
        "encoders": {"video": "handcrafted", "audio": "handcrafted"},
        "model": {"hidden_dim": 32, "num_heads": 4, "num_layers": 1, "fusion_dim": 16, "fusion_rank": 2, "classifier_hidden": 16},
        "training": {"batch_size": 4, "epochs": 2, "lr": 1e-3, "checkpoint_dir": str(tmp / "ckpt"),
                     "device": "cpu", "early_stopping_patience": 5},
        "evaluation": {"output_dir": str(tmp / "reports"), "n_bootstrap": 50, "latency_iters": 2, "latency_warmup": 1},
    }
    for section, vals in extra.items():
        raw.setdefault(section, {}).update(vals)
    return build_config(raw)


@pytest.fixture(scope="session")
def mini_dataset(tmp_path_factory):
    """8 scenes -> 16 rendered clips with group-aware splits and cached features."""
    from av_contradiction.data.features import extract_manifest
    from av_contradiction.data.manifest import write_manifest
    from av_contradiction.data.splits import group_stratified_split
    from av_contradiction.data.synthetic import generate_colortone

    tmp = tmp_path_factory.mktemp("mini")
    cfg = tiny_config(tmp)
    rows = generate_colortone(tmp / "data", n_scenes=8, seed=3, seg_sec=1.0)
    rows = group_stratified_split(rows, (0.5, 0.25, 0.25), seed=0)
    write_manifest(rows, cfg.data.manifest)
    report = extract_manifest(cfg, rows, "cpu")
    assert not report["failed"], report["failed"]
    return {"tmp": tmp, "cfg": cfg, "rows": rows}


@pytest.fixture(scope="session")
def trained_checkpoint(mini_dataset):
    from av_contradiction.training.trainer import train_from_config

    result = train_from_config(mini_dataset["cfg"], preset="xattn_tensor")
    assert result.best_checkpoint is not None
    return Path(result.best_checkpoint)
