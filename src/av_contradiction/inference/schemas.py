"""Structured prediction output."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class FrameEvidence(BaseModel):
    index: int
    timestamp_sec: float


class AudioWindowEvidence(BaseModel):
    index: int
    start_sec: float
    end_sec: float
    silent: bool


class TranscriptSegment(BaseModel):
    start_sec: float | None
    end_sec: float | None
    text: str


class AttentionLink(BaseModel):
    """Audio window -> most-attended video frame in the last audio->video cross-attention layer."""

    audio_window_index: int
    audio_time_sec: float
    video_frame_index: int | None
    video_time_sec: float | None
    weight: float
    null_weight: float = Field(description="Attention mass on the learned 'no matching evidence' token")


class Evidence(BaseModel):
    frames: list[FrameEvidence] = []
    audio_windows: list[AudioWindowEvidence] = []
    transcript: list[TranscriptSegment] = []
    attention_links: list[AttentionLink] = []
    note: str = (
        "Evidence lists the time-stamped inputs the model saw. Attention links show where audio "
        "queries attended in the video; attention weights are not causal explanations and do not "
        "localise the contradiction."
    )


class ModelInfo(BaseModel):
    checkpoint: str
    experiment_name: str
    preset: str
    video_encoder: str
    audio_encoder: str
    label_names: list[str]
    calibrated: bool
    temperature: float | None
    manifest_hash: str | None
    best_val_metric: float | None


class PredictionResult(BaseModel):
    label: Literal["consistent", "contradictory", "uncertain"]
    contradiction_probability: float
    consistency_probability: float
    probability_calibrated: bool
    confidence: float
    confidence_interpretation: str
    audio_summary: str | None
    visual_summary: str | None
    evidence: Evidence
    warnings: list[str]
    duration_sec: float
    model: ModelInfo
    timing_ms: dict[str, float]
