"""Offline feature extraction with an on-disk cache.

Cache layout: ``<feature_cache_dir>/<Config.feature_hash()>/<safe sample id>.pt``. The hash covers
every preprocessing and encoder setting, so changing e.g. ``num_frames`` never silently reuses
stale features.

Extraction is the expensive step on a Kaggle GPU, and FFmpeg decoding dominates it. Clips are therefore
decoded in background threads while the GPU encodes the previous ones, and the encoders run in fp16
on CUDA.
"""

from __future__ import annotations

import hashlib
import re
import time
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

from av_contradiction.config import Config
from av_contradiction.data.manifest import Row, resolve_video_path
from av_contradiction.data.preprocessing import DecodedClip, VideoDecodeError, decode_clip
from av_contradiction.models.audio_encoder import AudioEncoder, build_audio_encoder
from av_contradiction.models.video_encoder import VideoEncoder, build_video_encoder
from av_contradiction.utils.logging import get_logger

log = get_logger(__name__)


def safe_name(sample_id: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", sample_id)[:80]
    return f"{stem}-{hashlib.sha1(sample_id.encode()).hexdigest()[:8]}"


def cache_dir_for(cfg: Config) -> Path:
    return Path(cfg.data.feature_cache_dir) / cfg.feature_hash()


def cache_path(cfg: Config, sample_id: str) -> Path:
    return cache_dir_for(cfg) / f"{safe_name(sample_id)}.pt"


def _prefetched(fn: Callable[[Row], Any], items: list[Row], workers: int) -> Iterator[tuple[Row, Any]]:
    """Yield ``(item, fn(item))`` in order while up to ``2 * workers`` later items are processed in
    threads. The bounded window keeps only a few decoded clips (~6 MB each) in memory at a time."""
    workers = max(1, workers)
    with ThreadPoolExecutor(workers) as pool:
        pending: deque = deque()
        for item in items:
            pending.append((item, pool.submit(fn, item)))
            if len(pending) >= 2 * workers:
                head, fut = pending.popleft()
                yield head, fut.result()
        while pending:
            head, fut = pending.popleft()
            yield head, fut.result()


class FeatureExtractor:
    def __init__(self, cfg: Config, device: torch.device | str = "cpu") -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.video_encoder: VideoEncoder = build_video_encoder(cfg.encoders).to(self.device).eval()
        self.audio_encoder: AudioEncoder = build_audio_encoder(cfg.encoders, cfg.preprocess).to(self.device).eval()

    @property
    def dims(self) -> tuple[int, int]:
        return self.audio_encoder.dim, self.video_encoder.dim

    @torch.no_grad()
    def encode(self, clip: DecodedClip) -> dict[str, Any]:
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
            return self._encode(clip)

    def _encode(self, clip: DecodedClip) -> dict[str, Any]:
        bs = self.cfg.encoders.batch_size
        vchunks = []
        for i in range(0, len(clip.frames), bs):
            x = self.video_encoder.prepare(clip.frames[i : i + bs]).to(self.device)
            vchunks.append(self.video_encoder(x).float().cpu())
        achunks = []
        for i in range(0, len(clip.audio_windows), bs):
            inp = self.audio_encoder.prepare(clip.audio_windows[i : i + bs])
            achunks.append(self.audio_encoder(**{k: v.to(self.device) for k, v in inp.items()}).float().cpu())
        video = torch.cat(vchunks) if vchunks else torch.zeros(0, self.video_encoder.dim)
        audio = torch.cat(achunks) if achunks else torch.zeros(0, self.audio_encoder.dim)
        for name, t in (("video", video), ("audio", audio)):
            if not torch.isfinite(t).all():
                raise ValueError(f"Non-finite {name} features")
        return {
            "video": video,
            "video_ts": torch.from_numpy(clip.frame_ts.astype(np.float32)),
            "audio": audio,
            "audio_ts": torch.from_numpy(clip.audio_ts.astype(np.float32)).reshape(-1, 2),
            "audio_silent": torch.from_numpy(clip.audio_silent.astype(bool)),
            "duration_sec": float(clip.duration_sec),
            "feature_hash": self.cfg.feature_hash(),
        }


def extract_manifest(
    cfg: Config, rows: list[Row], device: torch.device | str = "cpu", overwrite: bool = False, decode_workers: int = 4
) -> dict[str, Any]:
    """Extract and cache features for every row. Failures are recorded, never fatal."""
    out_dir = cache_dir_for(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    extractor = FeatureExtractor(cfg, device)
    report: dict[str, Any] = {"feature_hash": cfg.feature_hash(), "cache_dir": str(out_dir), "extracted": 0, "skipped": 0, "failed": {}}
    todo = [r for r in rows if overwrite or not cache_path(cfg, r["sample_id"]).exists()]
    report["skipped"] = len(rows) - len(todo)

    def decode(row: Row) -> DecodedClip | Exception:
        try:
            return decode_clip(resolve_video_path(row, cfg.data.root_dir), cfg.preprocess)
        except (VideoDecodeError, OSError) as exc:
            return exc

    start = time.time()
    for i, (row, clip) in enumerate(_prefetched(decode, todo, decode_workers)):
        try:
            if isinstance(clip, Exception):
                raise clip
            dest = cache_path(cfg, row["sample_id"])
            tmp = dest.with_suffix(".tmp")
            torch.save(extractor.encode(clip), tmp)
            tmp.replace(dest)
            report["extracted"] += 1
        except (VideoDecodeError, ValueError, OSError) as exc:
            report["failed"][row["sample_id"]] = str(exc)
        if (i + 1) % 100 == 0:
            log.info("features: %d/%d (%.1fs)", i + 1, len(todo), time.time() - start)
    report["audio_dim"], report["video_dim"] = extractor.dims
    report["seconds"] = round(time.time() - start, 2)
    return report
