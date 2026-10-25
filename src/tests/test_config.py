import pytest

from av_contradiction.config import PROFILES, build_config, load_config, parse_overrides


def test_overrides_parse_types():
    out = parse_overrides(["model.hidden_dim=64", "training.amp=false", "data.label_names=[a, b]"])
    assert out == {"model": {"hidden_dim": 64}, "training": {"amp": False}, "data": {"label_names": ["a", "b"]}}


def test_unknown_key_raises():
    with pytest.raises(KeyError):
        build_config({"model": {"hiden_dim": 3}})


def test_profile_precedence_yaml_wins():
    cfg = build_config({"hardware": {"profile": "low_memory"}, "model": {"hidden_dim": 96, "num_heads": 4}})
    assert cfg.model.hidden_dim == 96  # explicit YAML beats profile
    assert cfg.training.grad_accum_steps == PROFILES["low_memory"]["training"]["grad_accum_steps"]


def test_cli_beats_yaml():
    cfg = build_config({"model": {"hidden_dim": 96}}, ["model.hidden_dim=128"])
    assert cfg.model.hidden_dim == 128


def test_defaults_need_no_config_file():
    cfg = load_config(None)
    assert (cfg.encoders.video, cfg.encoders.audio) == ("clip", "clap")
    assert cfg.preprocess.num_frames == 16 and cfg.training.batch_size == 32


def test_invalid_combinations():
    with pytest.raises(ValueError):
        build_config({"model": {"use_audio": False, "use_video": False}})
    with pytest.raises(ValueError):
        build_config({"encoders": {"audio": "clap"}, "preprocess": {"audio_sample_rate": 16000}})


def test_feature_hash_tracks_preprocessing():
    a = build_config({"preprocess": {"num_frames": 8}})
    b = build_config({"preprocess": {"num_frames": 16}})
    c = build_config({"preprocess": {"num_frames": 8}, "encoders": {"batch_size": 2}})
    assert a.feature_hash() != b.feature_hash()
    assert a.feature_hash() == c.feature_hash()  # batch size does not change features


def test_checkpoint_configs_tolerate_obsolete_keys():
    from av_contradiction.config import config_from_dict

    with pytest.warns(UserWarning, match="obsolete"):
        cfg = config_from_dict({"data": {"removed_option": True}, "model": {"hidden_dim": 32, "num_heads": 4}})
    assert cfg.model.hidden_dim == 32
