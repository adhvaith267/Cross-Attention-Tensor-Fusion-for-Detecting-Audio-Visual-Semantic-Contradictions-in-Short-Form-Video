"""Per-window audio encoders: ``float32 [T_a, samples]`` waveform windows -> ``[T_a, D]``.

They encode the acoustic signal directly (speech, music, environmental sound) and are always frozen.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from av_contradiction.config import EncoderConfig, PreprocessConfig


def mel_filterbank(n_fft: int, n_mels: int, sample_rate: int, fmin: float = 0.0, fmax: float | None = None) -> torch.Tensor:
    """Triangular HTK-style mel filterbank, shape [n_mels, n_fft // 2 + 1]."""
    fmax = fmax or sample_rate / 2

    def hz_to_mel(f: np.ndarray | float) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)

    def mel_to_hz(m: np.ndarray) -> np.ndarray:
        return 700.0 * (10 ** (m / 2595.0) - 1.0)

    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz = mel_to_hz(mels)
    bins = np.linspace(0, sample_rate / 2, n_fft // 2 + 1)
    fb = np.zeros((n_mels, len(bins)), np.float32)
    for i in range(n_mels):
        lo, mid, hi = hz[i], hz[i + 1], hz[i + 2]
        up = (bins - lo) / max(mid - lo, 1e-8)
        down = (hi - bins) / max(hi - mid, 1e-8)
        fb[i] = np.maximum(0.0, np.minimum(up, down))
    return torch.from_numpy(fb)


class AudioEncoder(nn.Module):
    dim: int = 0
    sample_rate: int = 16000

    def prepare(self, windows: np.ndarray) -> dict[str, torch.Tensor]:
        raise NotImplementedError


class HandcraftedAudioEncoder(AudioEncoder):
    """Parameter-free log-mel statistics (mean and std over time of 40 mel bands). Tests/sanity only."""

    def __init__(self, sample_rate: int = 16000, n_mels: int = 40, n_fft: int = 512, hop: int = 160) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft, self.hop, self.n_mels = n_fft, hop, n_mels
        self.dim = 2 * n_mels
        self.register_buffer("fb", mel_filterbank(n_fft, n_mels, sample_rate), persistent=False)
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

    def prepare(self, windows: np.ndarray) -> dict[str, torch.Tensor]:
        return {"waveforms": torch.from_numpy(np.ascontiguousarray(windows, dtype=np.float32))}

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        if waveforms.shape[0] == 0:
            return waveforms.new_zeros((0, self.dim))
        if waveforms.shape[1] < self.n_fft:
            waveforms = torch.nn.functional.pad(waveforms, (0, self.n_fft - waveforms.shape[1]))
        spec = torch.stft(waveforms, self.n_fft, self.hop, window=self.window, return_complex=True).abs() ** 2
        logmel = torch.log(self.fb @ spec + 1e-6)  # [T, n_mels, frames]
        return torch.cat([logmel.mean(-1), logmel.std(-1)], dim=1) / 10.0


class CLAPAudioEncoder(AudioEncoder):
    """LAION-CLAP audio tower via ``transformers`` (expects 48 kHz audio)."""

    sample_rate = 48000

    def __init__(self, model_name: str, local_files_only: bool = True) -> None:
        super().__init__()
        try:
            from transformers import ClapFeatureExtractor, ClapModel
        except ImportError as exc:
            raise ImportError("CLAP encoder needs `pip install -e .[encoders]`") from exc
        try:
            self.model = ClapModel.from_pretrained(model_name, local_files_only=local_files_only)
            self.extractor = ClapFeatureExtractor.from_pretrained(model_name, local_files_only=local_files_only)
        except OSError as exc:
            raise OSError(
                f"CLAP weights '{model_name}' not in local cache. Download deliberately and set "
                "encoders.local_files_only=false once, or pre-populate HF_HOME."
            ) from exc
        # Only the audio tower is used; drop the text tower to save memory.
        self.model.text_model = None
        self.model.requires_grad_(False).eval()
        self.dim = int(self.model.config.projection_dim)

    def prepare(self, windows: np.ndarray) -> dict[str, torch.Tensor]:
        if len(windows) == 0:
            return {"input_features": torch.zeros(0)}
        feats = self.extractor(list(windows), sampling_rate=self.sample_rate, return_tensors="pt")
        return {k: v for k, v in feats.items() if k in ("input_features", "is_longer")}

    def forward(self, input_features: torch.Tensor, is_longer: torch.Tensor | None = None) -> torch.Tensor:
        if input_features.numel() == 0:
            return torch.zeros((0, self.dim), device=input_features.device)
        out = self.model.get_audio_features(input_features=input_features, is_longer=is_longer)
        if not isinstance(out, torch.Tensor):  # newer transformers return a ModelOutput
            emb = getattr(out, "audio_embeds", None)
            out = emb if emb is not None else out.pooler_output
        return out


def build_audio_encoder(enc: EncoderConfig, pre: PreprocessConfig) -> AudioEncoder:
    if enc.audio == "handcrafted":
        return HandcraftedAudioEncoder(sample_rate=pre.audio_sample_rate)
    if enc.audio == "clap":
        if pre.audio_sample_rate != CLAPAudioEncoder.sample_rate:
            raise ValueError("CLAP expects preprocess.audio_sample_rate=48000")
        return CLAPAudioEncoder(enc.audio_model_name, enc.local_files_only)
    raise ValueError(f"Unknown audio encoder '{enc.audio}'")

