"""Per-frame visual encoders.

Every encoder maps sampled frames ``uint8 [T, S, S, 3]`` to embeddings ``[T, D]`` in two steps:
``prepare`` (preprocessing on CPU) and ``forward`` (frozen encoder). These are *image* encoders
applied frame by frame: temporal reasoning happens later, in the cross-attention layers over
time-stamped frame sequences.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from av_contradiction.config import EncoderConfig

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class VideoEncoder(nn.Module):
    dim: int = 0

    def prepare(self, frames: np.ndarray) -> torch.Tensor:
        raise NotImplementedError


class HandcraftedVideoEncoder(VideoEncoder):
    """Parameter-free colour/layout descriptor (no downloads). For tests and CPU sanity benchmarks only.

    Per frame: 4x4x4 joint RGB histogram (64), channel mean/std (6) and 2x2 spatial channel means (12).
    """

    dim = 82

    def prepare(self, frames: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(frames)).float() / 255.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [T, H, W, 3] in [0, 1]
        t = x.shape[0]
        if t == 0:
            return x.new_zeros((0, self.dim))
        q = (x * 3.999).long().clamp(0, 3)
        idx = q[..., 0] * 16 + q[..., 1] * 4 + q[..., 2]
        hist = F.one_hot(idx.reshape(t, -1), 64).float().mean(1)
        flat = x.reshape(t, -1, 3)
        mean, std = flat.mean(1), flat.std(1)
        quad = F.adaptive_avg_pool2d(x.permute(0, 3, 1, 2), 2).reshape(t, -1)
        return torch.cat([hist, mean, std, quad], dim=1)


class CLIPVideoEncoder(VideoEncoder):
    """CLIP ViT image tower (``transformers``); returns projected image embeddings per frame."""

    def __init__(self, model_name: str, local_files_only: bool = True) -> None:
        super().__init__()
        try:
            from transformers import CLIPVisionModelWithProjection
        except ImportError as exc:
            raise ImportError("CLIP encoder needs `pip install -e .[encoders]`") from exc
        try:
            self.model = CLIPVisionModelWithProjection.from_pretrained(model_name, local_files_only=local_files_only)
        except OSError as exc:
            raise OSError(
                f"CLIP weights '{model_name}' not in local cache. Download deliberately (check disk/license) "
                "and set encoders.local_files_only=false once, or pre-populate HF_HOME."
            ) from exc
        self.model.requires_grad_(False).eval()
        self.dim = int(self.model.config.projection_dim)
        self.image_size = int(self.model.config.image_size)
        self.register_buffer("mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(CLIP_STD).view(1, 3, 1, 1), persistent=False)

    def prepare(self, frames: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float() / 255.0
        if x.shape[0] and x.shape[-1] != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bicubic", align_corners=False)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 0:
            return x.new_zeros((0, self.dim))
        x = (x.to(self.mean.device) - self.mean) / self.std
        return self.model(pixel_values=x).image_embeds


def build_video_encoder(enc: EncoderConfig) -> VideoEncoder:
    if enc.video == "handcrafted":
        return HandcraftedVideoEncoder()
    if enc.video == "clip":
        return CLIPVideoEncoder(enc.video_model_name, enc.local_files_only)
    raise ValueError(f"Unknown video encoder '{enc.video}'")
