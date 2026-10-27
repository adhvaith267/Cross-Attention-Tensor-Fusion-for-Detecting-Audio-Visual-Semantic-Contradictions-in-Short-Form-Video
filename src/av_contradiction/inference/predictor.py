"""Single-video inference from a trained checkpoint."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from av_contradiction.data.dataset import collate_features, move_batch
from av_contradiction.data.features import FeatureExtractor
from av_contradiction.data.preprocessing import DecodedClip, decode_clip
from av_contradiction.evaluation.evaluate import config_from_checkpoint, load_model
from av_contradiction.inference.schemas import (
    AttentionLink,
    AudioWindowEvidence,
    Evidence,
    FrameEvidence,
    ModelInfo,
    PredictionResult,
)
from av_contradiction.training.checkpointing import load_checkpoint
from av_contradiction.training.metrics import softmax
from av_contradiction.utils.device import select_device
from av_contradiction.utils.logging import get_logger

log = get_logger(__name__)


class Predictor:
    def __init__(self, checkpoint: str | Path, device: str = "auto", overrides: list[str] | None = None) -> None:
        self.checkpoint = Path(checkpoint)
        self.ckpt = load_checkpoint(self.checkpoint, "cpu")
        self.cfg = config_from_checkpoint(self.ckpt, overrides)
        self.device = select_device(device)
        self.model = load_model(self.ckpt, self.cfg, self.device)
        self.extractor = FeatureExtractor(self.cfg, self.device)
        if self.extractor.dims != (self.ckpt["audio_dim"], self.ckpt["video_dim"]):
            raise ValueError(f"Encoder dims {self.extractor.dims} do not match checkpoint dims")
        self.label_names = list(self.ckpt["label_names"])
        self.temperature: float | None = None
        self.calibration_degenerate = False
        cal = self.checkpoint.with_name("calibration.json")
        if cal.is_file():
            cal_data = json.loads(cal.read_text(encoding="utf-8"))
            self.temperature = float(cal_data["temperature"])
            self.calibration_degenerate = bool(cal_data.get("at_bound", False))
        self.feature_stats: dict[str, Any] = self.ckpt.get("feature_stats") or {}
        # Encoder features of the most recent predict_clip call (used by the dashboard's text view).
        self.last_features: dict[str, Any] | None = None

    def info(self) -> ModelInfo:
        return ModelInfo(
            checkpoint=self.checkpoint.name,
            experiment_name=self.cfg.experiment_name,
            preset=self.ckpt.get("preset", ""),
            video_encoder=self.cfg.encoders.video,
            audio_encoder=self.cfg.encoders.audio,
            label_names=self.label_names,
            calibrated=self.temperature is not None and not self.calibration_degenerate,
            temperature=self.temperature,
            manifest_hash=self.ckpt.get("manifest_hash"),
            best_val_metric=self.ckpt.get("best_metric"),
        )

    def _ood_warnings(self, feats: dict[str, Any]) -> list[str]:
        out = []
        for mod, st in self.feature_stats.items():
            x = feats[mod]
            if x.shape[0] == 0:
                continue
            dist = float((x.float().mean(0) - torch.tensor(st["mean"])).norm())
            z = (dist - st["dist_mean"]) / st["dist_std"]
            if z > self.cfg.inference.ood_zscore_threshold:
                out.append(f"{mod} features are far from the training distribution (z={z:.1f}); prediction may be unreliable.")
        return out

    @torch.no_grad()
    def predict_clip(self, path: str | Path) -> tuple[PredictionResult, DecodedClip]:
        timing: dict[str, float] = {}
        t0 = time.perf_counter()
        clip = decode_clip(path, self.cfg.preprocess)
        timing["decode_ms"] = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        feats = self.extractor.encode(clip)
        self.last_features = feats
        timing["encode_ms"] = (time.perf_counter() - t1) * 1000
        t2 = time.perf_counter()
        batch = move_batch(collate_features([{**feats, "label": 0, "sample_id": "input"}]), self.device)
        want_attn = self.model.cross is not None
        out = self.model(batch, return_attention=want_attn)
        timing["model_ms"] = (time.perf_counter() - t2) * 1000
        logits = out["logits"].float().cpu().numpy()
        probs = softmax(logits, self.temperature or 1.0)[0]
        p_con = float(probs[self.label_names.index("contradictory")])

        warnings = list(clip.warnings) + self._ood_warnings(feats)
        needs_audio, needs_video = self.cfg.model.use_audio, self.cfg.model.use_video
        missing = (needs_audio and (not clip.has_audio or bool(clip.audio_silent.all()))) or (needs_video and not clip.has_video)
        inf = self.cfg.inference
        if missing:
            label = "uncertain"
            warnings.append("A required modality is missing or silent; returning 'uncertain'.")
        elif inf.uncertain_low <= p_con <= inf.uncertain_high:
            label = "uncertain"
        else:
            label = "contradictory" if p_con > inf.uncertain_high else "consistent"
        if self.temperature is None:
            warnings.append("Probabilities are raw softmax scores (no calibration.json found); they are not calibrated.")
        elif self.calibration_degenerate:
            warnings.append("Calibration is degenerate (temperature at its bound because validation was separable); "
                            "treat the probability as uncalibrated.")

        evidence = Evidence(
            frames=[FrameEvidence(index=i, timestamp_sec=float(t)) for i, t in enumerate(clip.frame_ts)],
            audio_windows=[
                AudioWindowEvidence(index=i, start_sec=float(s), end_sec=float(e), silent=bool(clip.audio_silent[i]))
                for i, (s, e) in enumerate(clip.audio_ts)
            ],
            attention_links=self._attention_links(out, clip) if want_attn else [],
        )
        return (
            PredictionResult(
                label=label,
                contradiction_probability=p_con,
                consistency_probability=1.0 - p_con,
                probability_calibrated=self.temperature is not None and not self.calibration_degenerate,
                confidence=max(p_con, 1.0 - p_con),
                confidence_interpretation=(
                    "Temperature-scaled probability (fitted on the validation split). It reflects reliability on the "
                    "evaluated benchmark only and is not guaranteed for out-of-distribution videos."
                    if self.temperature is not None and not self.calibration_degenerate
                    else "Raw or degenerately scaled softmax score; not a calibrated probability."
                ),
                evidence=evidence,
                warnings=warnings,
                duration_sec=float(clip.duration_sec),
                model=self.info(),
                timing_ms={k: round(v, 2) for k, v in timing.items()},
            ),
            clip,
        )

    def predict(self, path: str | Path) -> PredictionResult:
        return self.predict_clip(path)[0]

    @staticmethod
    def _attention_links(out: dict[str, Any], clip: DecodedClip) -> list[AttentionLink]:
        maps = out.get("attn_a2v") or []
        if not maps or not clip.has_audio:
            return []
        w = maps[-1][0].float().cpu().numpy()  # [Ta, 1 + Tv]
        links = []
        for i in range(min(len(clip.audio_ts), w.shape[0])):
            row = w[i]
            frame_w = row[1 : 1 + len(clip.frame_ts)]
            j = int(np.argmax(frame_w)) if frame_w.size else None
            links.append(
                AttentionLink(
                    audio_window_index=i,
                    audio_time_sec=float(clip.audio_ts[i].mean()),
                    video_frame_index=j,
                    video_time_sec=float(clip.frame_ts[j]) if j is not None else None,
                    weight=float(frame_w[j]) if j is not None else 0.0,
                    null_weight=float(row[0]),
                )
            )
        return links
