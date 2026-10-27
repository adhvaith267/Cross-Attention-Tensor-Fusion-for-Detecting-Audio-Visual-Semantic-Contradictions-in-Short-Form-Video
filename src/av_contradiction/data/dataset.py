"""Cached-feature dataset and its padding collate function.

Batch layout produced by ``collate_features`` (B batch, Ta/Tv = longest sequence in batch, >= 1):

    audio       float [B, Ta, Da]     audio_ts  float [B, Ta, 2] (start, end sec)   audio_mask bool [B, Ta]
    video       float [B, Tv, Dv]     video_ts  float [B, Tv]                        video_mask bool [B, Tv]
    label       long  [B]             sample_id list[str]

Masks are True for real elements. A missing modality yields an all-False mask row.

A row may name ``video_feature_id`` and ``audio_feature_id``: the cached features of other rows that
supply its video and audio. This builds audio-swapped clips from cached features without writing new
videos (see ``data/swaps.py``). Such a row keeps the video source's timeline (``video_ts`` and
``audio_ts``) and takes only the audio embeddings and silence flags from the audio source, so the two
sources must have the same number of audio windows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from av_contradiction.config import Config
from av_contradiction.data.features import cache_path
from av_contradiction.data.manifest import Row


class CachedFeatureDataset(Dataset):
    """Features are read from disk once and then kept in memory (a few hundred MB for ~10k clips),
    so later epochs do no file I/O. Keep ``num_workers=0`` so that memory cache is shared."""

    def __init__(self, cfg: Config, rows: list[Row]) -> None:
        self.label_to_id = {n: i for i, n in enumerate(cfg.data.label_names)}
        self.items: list[tuple[Row, Path, Path]] = []
        self.missing: list[str] = []
        self._memory: dict[int, dict[str, Any]] = {}
        self._files: dict[Path, dict[str, Any]] = {}
        for r in rows:
            vp, ap = (cache_path(cfg, fid) for fid in feature_sources(r))
            if vp.exists() and ap.exists():
                self.items.append((r, vp, ap))
            else:
                self.missing.append(r["sample_id"])

    def __len__(self) -> int:
        return len(self.items)

    @property
    def labels(self) -> np.ndarray:
        return np.array([self.label_to_id[r["label"]] for r, _, _ in self.items], dtype=np.int64)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx not in self._memory:
            row, vpath, apath = self.items[idx]
            feats = dict(self._load(vpath))
            if apath != vpath:
                audio = self._load(apath)
                if audio["audio"].shape[0] != feats["audio"].shape[0]:
                    raise ValueError(f"{row['sample_id']}: audio source has {audio['audio'].shape[0]} windows, "
                                     f"the video source's timeline has {feats['audio'].shape[0]}")
                feats["audio"], feats["audio_silent"] = audio["audio"], audio["audio_silent"]
            feats["label"] = self.label_to_id[row["label"]]
            feats["sample_id"] = row["sample_id"]
            self._memory[idx] = feats
        return self._memory[idx]

    def _load(self, path: Path) -> dict[str, Any]:
        if path not in self._files:
            self._files[path] = torch.load(path, map_location="cpu", weights_only=True)
        return self._files[path]


def feature_sources(row: Row) -> tuple[str, str]:
    """Sample ids whose cached features supply this row's (video, audio); both are its own id by default."""
    return row.get("video_feature_id") or row["sample_id"], row.get("audio_feature_id") or row["sample_id"]


def _pad_stack(seqs: list[torch.Tensor], trailing: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    t = max(1, max(s.shape[0] for s in seqs))
    out = torch.zeros((len(seqs), t, *trailing), dtype=torch.float32)
    mask = torch.zeros((len(seqs), t), dtype=torch.bool)
    for i, s in enumerate(seqs):
        n = s.shape[0]
        if n:
            out[i, :n] = s.reshape(n, *trailing).float()
            mask[i, :n] = True
    return out, mask


def collate_features(items: list[dict[str, Any]]) -> dict[str, Any]:
    da = items[0]["audio"].shape[-1]
    dv = items[0]["video"].shape[-1]
    audio, audio_mask = _pad_stack([it["audio"] for it in items], (da,))
    audio_ts, _ = _pad_stack([it["audio_ts"] for it in items], (2,))
    video, video_mask = _pad_stack([it["video"] for it in items], (dv,))
    video_ts, _ = _pad_stack([it["video_ts"] for it in items], ())
    return {
        "audio": audio, "audio_ts": audio_ts, "audio_mask": audio_mask,
        "video": video, "video_ts": video_ts, "video_mask": video_mask,
        "label": torch.tensor([it["label"] for it in items], dtype=torch.long),
        "sample_id": [it["sample_id"] for it in items],
    }


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def feature_statistics(dataset: CachedFeatureDataset) -> dict[str, dict[str, Any]]:
    """Per-modality mean of clip-level (mean-pooled) embeddings and the distribution of distances
    to it. Used at inference for a coarse out-of-distribution warning."""
    stats: dict[str, dict[str, Any]] = {}
    for mod in ("audio", "video"):
        pooled = []
        for i in range(len(dataset)):
            x = dataset[i][mod]
            if x.shape[0]:
                pooled.append(x.float().mean(0))
        if len(pooled) < 2:
            continue
        p = torch.stack(pooled)
        mu = p.mean(0)
        d = (p - mu).norm(dim=1)
        stats[mod] = {"mean": mu.tolist(), "dist_mean": float(d.mean()), "dist_std": float(d.std() + 1e-8)}
    return stats
