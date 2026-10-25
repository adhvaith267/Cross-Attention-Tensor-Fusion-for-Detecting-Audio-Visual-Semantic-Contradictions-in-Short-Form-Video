"""Typed experiment configuration.

The dataclass defaults below are the settings for training on a Kaggle GPU (T4 / P100, ~16 GB)
with frozen CLIP + CLAP features, so no config file is needed.

Configuration is resolved in this order (later wins):

1. Dataclass defaults below.
2. Hardware-profile overrides (``hardware.profile``; ``auto`` picks one from detected VRAM).
3. Values in an optional YAML file (``--config``; may inherit from a ``_base_`` file).
4. Command-line ``--set section.key=value`` overrides.

Unknown keys raise an error so typos never silently fall back to defaults.
"""

from __future__ import annotations

import copy
import hashlib
import json
import warnings
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    manifest: str = "data/manifests/manifest.csv"
    # Relative video paths in the manifest are resolved against this directory.
    root_dir: str = "data/raw"
    feature_cache_dir: str = "output/cache/features"
    label_names: list[str] = field(default_factory=lambda: ["consistent", "contradictory"])


@dataclass
class PreprocessConfig:
    num_frames: int = 16
    frame_size: int = 224
    # 48000 for CLAP; 16000 is enough for the handcrafted encoder.
    audio_sample_rate: int = 48000
    audio_window_sec: float = 2.0
    audio_hop_sec: float = 1.0
    max_audio_windows: int = 64
    min_duration_sec: float = 0.5
    max_duration_sec: float = 60.0
    # None -> $AVC_FFMPEG, then system ffmpeg, then the imageio-ffmpeg bundled binary.
    ffmpeg_path: str | None = None
    ffmpeg_timeout_sec: float = 120.0
    # Windows whose RMS is below this are treated as silent.
    silence_rms_threshold: float = 1e-3


@dataclass
class EncoderConfig:
    # "clip" | "handcrafted"
    video: str = "clip"
    video_model_name: str = "openai/clip-vit-base-patch32"
    # "clap" | "handcrafted"
    audio: str = "clap"
    audio_model_name: str = "laion/clap-htsat-unfused"
    # True: never download weights; they must already be in the local HF cache.
    local_files_only: bool = True
    batch_size: int = 32
    # Optional Whisper transcription (used only for evidence / audio_summary, never as the audio encoder).
    transcribe: bool = False
    whisper_model_name: str = "openai/whisper-small"


@dataclass
class ModelConfig:
    hidden_dim: int = 256
    num_heads: int = 4
    num_layers: int = 2
    ff_mult: int = 2
    dropout: float = 0.1
    use_audio: bool = True
    use_video: bool = True
    use_cross_attention: bool = True
    # "tensor" (low-rank tensor fusion) | "concat" (MLP over concatenation)
    fusion: str = "tensor"
    fusion_rank: int = 4
    fusion_dim: int = 128
    # "timestamp": sinusoidal time encodings + temporal-proximity attention bias; "global": neither.
    alignment: str = "timestamp"
    time_bias_sigma_sec: float = 1.0
    classifier_hidden: int = 128
    num_classes: int = 2


@dataclass
class TrainingConfig:
    # "cached": train fusion layers on pre-extracted frozen features.
    # "end_to_end": decode videos on the fly and optionally fine-tune encoders (experimental).
    mode: str = "cached"
    seed: int = 42
    device: str = "auto"
    batch_size: int = 32
    num_workers: int = 0
    epochs: int = 30
    max_steps: int | None = None
    lr: float = 3e-4
    encoder_lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    # "cosine" | "linear" | "constant"
    scheduler: str = "cosine"
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    amp: bool = True
    early_stopping_patience: int = 6
    monitor: str = "macro_f1"
    # "auto": inverse-frequency weights only if majority/minority ratio > imbalance_threshold; "none"; "balanced".
    class_weighting: str = "auto"
    imbalance_threshold: float = 1.5
    label_smoothing: float = 0.0
    checkpoint_dir: str = "output/checkpoints"
    log_dir: str = "output/runs"
    save_every_epochs: int = 1
    max_nan_steps: int = 10
    freeze_encoders: bool = True
    unfreeze_last_n_layers: int = 0


@dataclass
class EvaluationConfig:
    split: str = "test"
    output_dir: str = "output/reports"
    # Temperature scaling fitted on the validation split only.
    calibrate: bool = True
    ece_bins: int = 10
    n_bootstrap: int = 1000
    latency_warmup: int = 3
    latency_iters: int = 20


@dataclass
class InferenceConfig:
    # Contradiction probability inside [uncertain_low, uncertain_high] -> label "uncertain".
    uncertain_low: float = 0.35
    uncertain_high: float = 0.65
    max_upload_mb: float = 100.0
    allowed_extensions: list[str] = field(default_factory=lambda: [".mp4", ".mov", ".mkv", ".webm", ".avi"])
    return_attention: bool = True
    ood_zscore_threshold: float = 3.0


@dataclass
class HardwareConfig:
    # "auto" | "low_memory" | "standard_gpu" | "experimental" | "none"
    profile: str = "none"


@dataclass
class Config:
    experiment_name: str = "default"
    data: DataConfig = field(default_factory=DataConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    encoders: EncoderConfig = field(default_factory=EncoderConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def feature_hash(self) -> str:
        """Hash of every setting that changes cached features (used as cache key)."""
        relevant = {
            "preprocess": {k: v for k, v in asdict(self.preprocess).items() if k not in ("ffmpeg_path", "ffmpeg_timeout_sec")},
            "encoders": {k: v for k, v in asdict(self.encoders).items() if k not in ("batch_size", "local_files_only")},
        }
        return hashlib.sha1(json.dumps(relevant, sort_keys=True).encode()).hexdigest()[:12]


# Overrides applied on top of defaults when a hardware profile is selected.
PROFILES: dict[str, dict[str, Any]] = {
    "low_memory": {
        "preprocess": {"num_frames": 8, "max_audio_windows": 32},
        "model": {"hidden_dim": 128, "num_layers": 1, "fusion_dim": 64, "fusion_rank": 4},
        "training": {"batch_size": 8, "grad_accum_steps": 4, "amp": True, "freeze_encoders": True},
        "encoders": {"batch_size": 8},
    },
    "standard_gpu": {
        "preprocess": {"num_frames": 16, "max_audio_windows": 64},
        "model": {"hidden_dim": 256, "num_layers": 2, "fusion_dim": 128},
        "training": {"batch_size": 32, "grad_accum_steps": 1, "amp": True, "freeze_encoders": True},
        "encoders": {"batch_size": 32},
    },
    "experimental": {
        "preprocess": {"num_frames": 32, "max_audio_windows": 128},
        "model": {"hidden_dim": 384, "num_layers": 3, "fusion_dim": 192, "fusion_rank": 8},
        "training": {"batch_size": 16, "grad_accum_steps": 2, "amp": True},
        "encoders": {"batch_size": 32},
    },
}


def _merge_into(obj: Any, updates: dict[str, Any], path: str = "", strict: bool = True) -> None:
    valid = {f.name: f for f in fields(obj)}
    for key, value in updates.items():
        if key not in valid:
            if strict:
                raise KeyError(f"Unknown config key '{path}{key}'. Valid keys: {sorted(valid)}")
            warnings.warn(f"Ignoring obsolete config key '{path}{key}' (e.g. from an older checkpoint)", stacklevel=2)
            continue
        current = getattr(obj, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise TypeError(f"Config section '{path}{key}' must be a mapping")
            _merge_into(current, value, f"{path}{key}.", strict)
        else:
            setattr(obj, key, copy.deepcopy(value))


def _load_yaml_with_base(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    base = data.pop("_base_", None)
    if base is None:
        return data
    merged = _load_yaml_with_base((path.parent / base).resolve())
    return _deep_update(merged, data)


def _deep_update(base: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def parse_overrides(overrides: list[str] | None) -> dict[str, Any]:
    """Turn ``["model.hidden_dim=128", "training.amp=false"]`` into a nested dict (values parsed as YAML)."""
    result: dict[str, Any] = {}
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must look like section.key=value")
        dotted, raw = item.split("=", 1)
        node = result
        parts = dotted.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return result


def resolve_profile(name: str) -> str:
    if name != "auto":
        return name
    from av_contradiction.utils.device import recommend_profile

    return recommend_profile()


def build_config(raw: dict[str, Any] | None = None, overrides: list[str] | None = None, strict: bool = True) -> Config:
    """``strict=False`` drops unknown keys in ``raw`` with a warning (used for stored checkpoint configs);
    CLI overrides are always strict."""
    raw = copy.deepcopy(raw or {})
    cli = parse_overrides(overrides)
    requested = cli.get("hardware", {}).get("profile") or raw.get("hardware", {}).get("profile", "none")
    profile = resolve_profile(requested)
    cfg = Config()
    if profile not in ("none", None):
        if profile not in PROFILES:
            raise ValueError(f"Unknown hardware profile '{profile}'. Options: {sorted(PROFILES)} or auto/none")
        _merge_into(cfg, PROFILES[profile])
    _merge_into(cfg, raw, strict=strict)
    _merge_into(cfg, cli)
    cfg.hardware.profile = profile
    validate_config(cfg)
    return cfg


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    raw = _load_yaml_with_base(Path(path)) if path else {}
    return build_config(raw, overrides)


def config_from_dict(data: dict[str, Any]) -> Config:
    """Rebuild a Config from ``Config.to_dict()`` output (e.g. stored in a checkpoint)."""
    data = copy.deepcopy(data)
    data.setdefault("hardware", {})["profile"] = "none"
    return build_config(data, strict=False)


def validate_config(cfg: Config) -> None:
    m = cfg.model
    if not (m.use_audio or m.use_video):
        raise ValueError("At least one of model.use_audio / model.use_video must be true")
    if m.hidden_dim % m.num_heads != 0:
        raise ValueError("model.hidden_dim must be divisible by model.num_heads")
    if m.fusion not in ("tensor", "concat"):
        raise ValueError("model.fusion must be 'tensor' or 'concat'")
    if m.alignment not in ("timestamp", "global"):
        raise ValueError("model.alignment must be 'timestamp' or 'global'")
    if cfg.training.mode not in ("cached", "end_to_end"):
        raise ValueError("training.mode must be 'cached' or 'end_to_end'")
    if cfg.training.scheduler not in ("cosine", "linear", "constant"):
        raise ValueError("training.scheduler must be cosine|linear|constant")
    if cfg.training.class_weighting not in ("auto", "none", "balanced"):
        raise ValueError("training.class_weighting must be auto|none|balanced")
    if cfg.encoders.video not in ("clip", "handcrafted") or cfg.encoders.audio not in ("clap", "handcrafted"):
        raise ValueError("encoders.video must be clip|handcrafted and encoders.audio clap|handcrafted")
    if cfg.encoders.audio == "clap" and cfg.preprocess.audio_sample_rate != 48000:
        raise ValueError("encoders.audio=clap requires preprocess.audio_sample_rate=48000")
    if not 0.0 <= cfg.inference.uncertain_low <= cfg.inference.uncertain_high <= 1.0:
        raise ValueError("Need 0 <= inference.uncertain_low <= inference.uncertain_high <= 1")
    if cfg.preprocess.num_frames < 1:
        raise ValueError("preprocess.num_frames must be >= 1")


def save_config(cfg: Config, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg.to_dict(), fh, sort_keys=False)


def replace_section(cfg: Config, **sections: dict[str, Any]) -> Config:
    """Return a deep copy of ``cfg`` with nested section overrides applied."""
    new = copy.deepcopy(cfg)
    _merge_into(new, sections)
    validate_config(new)
    return new


__all__ = [
    "Config",
    "DataConfig",
    "PreprocessConfig",
    "EncoderConfig",
    "ModelConfig",
    "TrainingConfig",
    "EvaluationConfig",
    "InferenceConfig",
    "HardwareConfig",
    "PROFILES",
    "load_config",
    "build_config",
    "config_from_dict",
    "parse_overrides",
    "save_config",
    "replace_section",
]
