"""Full audio-visual contradiction model and an end-to-end wrapper with encoders.

Data flow (B batch; Ta audio windows; Tv frames; D = hidden_dim):

    audio feats [B,Ta,Da] --proj--> a [B,Ta,D] (+ time enc)     video feats [B,Tv,Dv] --proj--> v [B,Tv,D] (+ time enc)
                     \\                                                          /
                      `---- bidirectional cross-attention (masks, temporal bias) --'
                                   a~ [B,Ta,D]            v~ [B,Tv,D]
    masked mean pooling: z_a, z_v, z_a~, z_v~  (each [B, D])
    fusion: low-rank tensor fusion (or concat MLP) -> [B, fusion_dim] -> MLP head -> logits [B, 2]

Ablation switches (ModelConfig): use_audio, use_video, use_cross_attention, fusion, alignment.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from av_contradiction.config import ModelConfig
from av_contradiction.models.alignment import TemporalProximityBias, sinusoidal_time_encoding, window_centers
from av_contradiction.models.cross_attention import BidirectionalCrossAttention
from av_contradiction.models.tensor_fusion import ConcatFusion, LowRankTensorFusion


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """x [B,T,D], mask [B,T] -> [B,D]; all-masked rows give zeros."""
    m = mask.unsqueeze(-1).to(x.dtype)
    return (x * m).sum(1) / m.sum(1).clamp(min=1.0)


class Projection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, out_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(out_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AVContradictionModel(nn.Module):
    def __init__(self, cfg: ModelConfig, audio_dim: int, video_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.audio_dim, self.video_dim = audio_dim, video_dim
        d = cfg.hidden_dim
        self.use_audio, self.use_video = cfg.use_audio, cfg.use_video
        self.use_xattn = cfg.use_cross_attention and cfg.use_audio and cfg.use_video
        self.timestamp = cfg.alignment == "timestamp"

        self.audio_proj = Projection(audio_dim, d, cfg.dropout) if self.use_audio else None
        self.video_proj = Projection(video_dim, d, cfg.dropout) if self.use_video else None
        self.time_proj = nn.Linear(d, d, bias=False) if self.timestamp else None
        self.temporal_bias = TemporalProximityBias(cfg.time_bias_sigma_sec) if (self.timestamp and self.use_xattn) else None
        self.cross = (
            BidirectionalCrossAttention(d, cfg.num_heads, cfg.num_layers, cfg.ff_mult, cfg.dropout) if self.use_xattn else None
        )

        n_inputs = int(self.use_audio) + int(self.use_video) + (2 if self.use_xattn else 0)
        dims = [d] * n_inputs
        if cfg.fusion == "tensor":
            self.fusion: nn.Module = LowRankTensorFusion(dims, cfg.fusion_dim, cfg.fusion_rank, cfg.dropout)
        else:
            self.fusion = ConcatFusion(dims, cfg.fusion_dim, cfg.dropout)
        self.head = nn.Sequential(
            nn.Linear(cfg.fusion_dim, cfg.classifier_hidden), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.classifier_hidden, cfg.num_classes),
        )

    def _encode(self, proj: nn.Module, feats: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        x = proj(feats)
        if self.time_proj is not None:
            x = x + self.time_proj(sinusoidal_time_encoding(ts, x.shape[-1]).to(x.dtype))
        return x

    def forward(self, batch: dict[str, torch.Tensor], return_attention: bool = False) -> dict[str, Any]:
        pooled: list[torch.Tensor] = []
        out: dict[str, Any] = {}
        a = v = a_t = v_t = None
        if self.use_audio:
            a_t = window_centers(batch["audio_ts"])
            a = self._encode(self.audio_proj, batch["audio"], a_t)
            pooled.append(masked_mean(a, batch["audio_mask"]))
        if self.use_video:
            v_t = batch["video_ts"]
            v = self._encode(self.video_proj, batch["video"], v_t)
            pooled.append(masked_mean(v, batch["video_mask"]))
        if self.cross is not None:
            bias = self.temporal_bias(a_t, v_t) if self.temporal_bias is not None else None
            x = self.cross(a, batch["audio_mask"], v, batch["video_mask"], bias, need_weights=return_attention)
            pooled.append(masked_mean(x["audio"], batch["audio_mask"]))
            pooled.append(masked_mean(x["video"], batch["video_mask"]))
            if return_attention:
                out["attn_a2v"] = x["attn_a2v"]
                out["attn_v2a"] = x["attn_v2a"]
        fused = self.fusion(pooled)
        out["logits"] = self.head(fused)
        out["fused"] = fused
        return out


def build_model(cfg: ModelConfig, audio_dim: int, video_dim: int) -> AVContradictionModel:
    return AVContradictionModel(cfg, audio_dim, video_dim)


def count_parameters(module: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


class EndToEndModel(nn.Module):
    """Encoders + fusion model for ``training.mode=end_to_end`` (optional encoder fine-tuning).

    Batches carry per-sample *prepared* encoder inputs (lists); features are computed here,
    padded and passed to the fusion model, so gradients reach any unfrozen encoder layers.
    """

    def __init__(self, video_encoder: nn.Module, audio_encoder: nn.Module, fusion_model: AVContradictionModel) -> None:
        super().__init__()
        self.video_encoder, self.audio_encoder, self.fusion_model = video_encoder, audio_encoder, fusion_model

    @staticmethod
    def _pad(seqs: list[torch.Tensor], dim: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        t = max(1, max(s.shape[0] for s in seqs))
        x = torch.zeros(len(seqs), t, dim, device=device)
        m = torch.zeros(len(seqs), t, dtype=torch.bool, device=device)
        for i, s in enumerate(seqs):
            x[i, : s.shape[0]] = s
            m[i, : s.shape[0]] = True
        return x, m

    def forward(self, batch: dict[str, Any], return_attention: bool = False) -> dict[str, Any]:
        device = batch["video_ts"].device
        vfeats = [self.video_encoder(x.to(device)).float() for x in batch["video_inputs"]]
        afeats = [self.audio_encoder(**{k: t.to(device) for k, t in inp.items()}).float() for inp in batch["audio_inputs"]]
        video, video_mask = self._pad(vfeats, self.fusion_model.video_dim, device)
        audio, audio_mask = self._pad(afeats, self.fusion_model.audio_dim, device)
        feats = {
            "video": video, "video_mask": video_mask & batch["video_mask"], "video_ts": batch["video_ts"],
            "audio": audio, "audio_mask": audio_mask & batch["audio_mask"], "audio_ts": batch["audio_ts"],
        }
        return self.fusion_model(feats, return_attention)
