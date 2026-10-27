"""Text view of a prediction: what the frames show and what the audio sounds like, in words.

CLIP and CLAP were trained on image-caption and audio-caption pairs, so their embeddings can be compared
with text. The frozen encoders' features (the same ones the classifier sees) are scored against short
captions of everyday concepts (``vocabulary.VISUAL_CONCEPTS`` / ``SOUND_CONCEPTS``), giving zero-shot
"seen" and "heard" tags. A concept heard in the audio is then looked up in the frames: a sound whose
source never appears in the picture is a hint of a contradiction. Optionally the speech is transcribed (Whisper), its keywords are
extracted and each keyword is looked up in the frames the same way.

These tags are explanations for a human reader. They do not feed the classifier, and a low visual match
is a hint, not proof (the source of a sound can be off screen).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from av_contradiction.config import Config
from av_contradiction.data.preprocessing import extract_audio
from av_contradiction.inference.vocabulary import (
    AUDIO_TEMPLATE,
    OFF_SCREEN_SOUNDS,
    SOUND_CONCEPTS,
    STOPWORDS,
    VISUAL_CONCEPTS,
    VISUAL_TEMPLATE,
)

# Scale applied to cosine similarities before the softmax, as in CLIP's zero-shot classifier.
LOGIT_SCALE = 100.0
WHISPER_SAMPLE_RATE = 16000


@dataclass
class Tag:
    concept: str
    score: float  # softmax share among all concepts (relative, sums to 1 over the vocabulary)


@dataclass
class CrossCheck:
    phrase: str
    source: str          # "heard" (audio tag) or "said" (transcript keyword)
    visual_rank: int     # rank of the phrase among the visual concepts for the frames (1 = best match)
    visual_percentile: float
    n_concepts: int


@dataclass
class TextAnalysis:
    seen: list[Tag]
    heard: list[Tag]
    heard_timeline: list[dict[str, Any]]
    seen_timeline: list[dict[str, Any]]
    cross_checks: list[CrossCheck]
    transcript: str | None = None
    keywords: list[tuple[str, int]] = field(default_factory=list)


def _norm(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x.float(), dim=-1)


def _softmax(sims: torch.Tensor) -> torch.Tensor:
    return torch.softmax(LOGIT_SCALE * sims, dim=-1)


def keywords(text: str, top: int = 8) -> list[tuple[str, int]]:
    """Most frequent content words (lower-cased, stopwords and short words removed)."""
    words = [w.strip("'") for w in re.findall(r"[a-zA-Z][a-zA-Z']+", text.lower())]
    counts = Counter(w for w in words if len(w) > 2 and w not in STOPWORDS)
    return counts.most_common(top)


def top_tags(emb: torch.Tensor, concept_emb: torch.Tensor, concepts: list[str], k: int) -> list[Tag]:
    """Tags for the mean of ``emb`` [T, D] against ``concept_emb`` [N, D]."""
    probs = _softmax(_norm(emb.mean(0)) @ _norm(concept_emb).T)
    vals, idx = probs.topk(min(k, len(concepts)))
    return [Tag(concepts[i], float(v)) for v, i in zip(vals.tolist(), idx.tolist(), strict=True)]


def visual_rank(frames: torch.Tensor, phrase_emb: torch.Tensor, concept_emb: torch.Tensor) -> tuple[int, float]:
    """Where ``phrase_emb`` ranks among the concepts for the best-matching frame (1 = best of all)."""
    f = _norm(frames)
    phrase = float((f @ _norm(phrase_emb)).max())
    others = (f @ _norm(concept_emb).T).max(0).values
    rank = int((others > phrase).sum()) + 1
    return rank, 1.0 - (rank - 1) / max(1, len(concept_emb))


class TextEncoders:
    """CLIP and CLAP text towers matching the checkpoint's encoders (needed only for the text view)."""

    def __init__(self, cfg: Config, device: torch.device) -> None:
        from transformers import AutoTokenizer, ClapTextModelWithProjection, CLIPTextModelWithProjection

        enc = cfg.encoders
        if enc.video != "clip" or enc.audio != "clap":
            raise ValueError("The text view needs a model trained on CLIP + CLAP features")
        self.device = device
        lfo = enc.local_files_only
        self.clip = CLIPTextModelWithProjection.from_pretrained(enc.video_model_name, local_files_only=lfo).to(device).eval()
        self.clip_tok = AutoTokenizer.from_pretrained(enc.video_model_name, local_files_only=lfo)
        self.clap = ClapTextModelWithProjection.from_pretrained(enc.audio_model_name, local_files_only=lfo).to(device).eval()
        self.clap_tok = AutoTokenizer.from_pretrained(enc.audio_model_name, local_files_only=lfo)
        # Without its vocabulary files transformers can return an empty tokenizer that maps every word to
        # the same token, which would make every tag identical.
        for name, tok in ((enc.video_model_name, self.clip_tok), (enc.audio_model_name, self.clap_tok)):
            if len(tok) < 1000:
                raise OSError(f"Tokenizer files for '{name}' are missing; run python -m av_contradiction.cli.download_encoders")
        self.visual_concepts = self.visual([VISUAL_TEMPLATE.format(c) for c in VISUAL_CONCEPTS])
        self.audio_concepts = self.audio([AUDIO_TEMPLATE.format(c) for c in SOUND_CONCEPTS])

    @torch.no_grad()
    def visual(self, texts: list[str]) -> torch.Tensor:
        tok = self.clip_tok(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        return self.clip(**tok).text_embeds.float().cpu()

    @torch.no_grad()
    def audio(self, texts: list[str]) -> torch.Tensor:
        tok = self.clap_tok(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        return self.clap(**tok).text_embeds.float().cpu()


class Transcriber:
    """Speech-to-text with Whisper (``inference.asr_model_name``)."""

    def __init__(self, cfg: Config, device: torch.device) -> None:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        name, lfo = cfg.inference.asr_model_name, cfg.encoders.local_files_only
        model = AutoModelForSpeechSeq2Seq.from_pretrained(name, local_files_only=lfo)
        proc = AutoProcessor.from_pretrained(name, local_files_only=lfo)
        self.asr = pipeline("automatic-speech-recognition", model=model, tokenizer=proc.tokenizer,
                            feature_extractor=proc.feature_extractor, device=device)

    def __call__(self, video: str | Path) -> str:
        wav = extract_audio(video, WHISPER_SAMPLE_RATE)
        if wav.size < WHISPER_SAMPLE_RATE // 2:
            return ""
        out = self.asr({"raw": wav, "sampling_rate": WHISPER_SAMPLE_RATE}, chunk_length_s=30,
                       generate_kwargs={"task": "transcribe"})
        return out["text"].strip()


def analyse(
    feats: dict[str, Any], frame_ts: np.ndarray, audio_ts: np.ndarray, text: TextEncoders,
    transcript: str | None = None, k: int = 5,
) -> TextAnalysis:
    """Build the text view from the clip's cached-format features (``Predictor.last_features``)."""
    video, audio = feats["video"].float(), feats["audio"].float()
    audible = ~feats["audio_silent"].bool() if audio.shape[0] else torch.zeros(0, dtype=torch.bool)
    seen = top_tags(video, text.visual_concepts, VISUAL_CONCEPTS, k) if video.shape[0] else []
    heard = top_tags(audio[audible], text.audio_concepts, SOUND_CONCEPTS, k) if bool(audible.any()) else []

    seen_timeline = []
    for i, emb in enumerate(video):
        tag = top_tags(emb[None], text.visual_concepts, VISUAL_CONCEPTS, 1)[0]
        seen_timeline.append({"frame": i, "time": f"{frame_ts[i]:.1f}s", "seen": tag.concept, "share": round(tag.score, 2)})
    heard_timeline = []
    for i, emb in enumerate(audio):
        s, e = audio_ts[i]
        if not bool(audible[i]):
            heard_timeline.append({"window": i, "time": f"{s:.1f}-{e:.1f}s", "heard": "(silence)", "share": None})
            continue
        tag = top_tags(emb[None], text.audio_concepts, SOUND_CONCEPTS, 1)[0]
        heard_timeline.append({"window": i, "time": f"{s:.1f}-{e:.1f}s", "heard": tag.concept, "share": round(tag.score, 2)})

    checks: list[CrossCheck] = []
    words = keywords(transcript) if transcript else []
    phrases = [(t.concept, "heard") for t in heard[:3] if t.concept not in OFF_SCREEN_SOUNDS] + [(w, "said") for w, _ in words]
    if video.shape[0] and phrases:
        embs = text.visual([VISUAL_TEMPLATE.format(p) for p, _ in phrases])
        for (phrase, source), e in zip(phrases, embs, strict=True):
            r, pct = visual_rank(video, e, text.visual_concepts)
            checks.append(CrossCheck(phrase, source, r, pct, len(VISUAL_CONCEPTS)))
    return TextAnalysis(seen, heard, heard_timeline, seen_timeline, checks, transcript, words)
