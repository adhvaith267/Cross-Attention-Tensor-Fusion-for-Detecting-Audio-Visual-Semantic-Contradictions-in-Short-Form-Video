"""Streamlit dashboard for the AV contradiction detector.

Run from the project root: streamlit run src/av_contradiction/serving/dashboard.py

One tab per stage of the pipeline: dataset, preprocessing, feature extraction, text mining (zero-shot
tags, Whisper transcript, keywords), the models and their attention, and the evaluation results. The
dataset and evaluation tabs need no upload; the others show the analysis of the uploaded video.
All displayed numbers come from the loaded models, the manifests or saved evaluation reports.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from av_contradiction.inference.schemas import PredictionResult

# Checkpoints, manifests and reports are configured relative to the project root, where the app is launched.
ROOT = Path.cwd()
# Part of every cache key: Streamlit keeps cached models across code reloads, so objects built by an older
# version of the code (e.g. a Predictor without newer attributes) must not be reused.
CACHE_VERSION = 3
COLOURS = {"contradictory": "#d64545", "consistent": "#2e9d5b", "uncertain": "#e0a030"}
MODEL_ROLES = {
    "avid": "Edited-audio detector (trained on AVID-Bench audio injections)",
}

st.set_page_config(page_title="AV Contradiction Analytics", page_icon="🎬", layout="wide")

CSS = """
<style>
.block-container {padding-top: 1.6rem;}
.verdict {border-radius: 14px; padding: 16px 22px; color: white; margin-bottom: 10px;}
.verdict h2 {margin: 0; color: white; font-size: 1.9rem; letter-spacing: .04em;}
.verdict p {margin: 4px 0 0 0; opacity: .92;}
.card {border: 1px solid rgba(128,128,128,.25); border-radius: 12px; padding: 10px 14px; margin-bottom: 8px;}
.card .k {font-size: .75rem; opacity: .7; text-transform: uppercase; letter-spacing: .05em;}
.card .v {font-size: 1.2rem; font-weight: 600;}
.pill {display: inline-block; border-radius: 999px; padding: 2px 10px; margin: 2px 4px 2px 0; font-size: .85rem;
       background: rgba(100,120,200,.15);}
.step {border-left: 4px solid #5b7bd5; padding: 4px 12px; margin: 6px 0;}
.hero {border: 1px solid rgba(91,123,213,.28); border-radius: 18px; padding: 26px 30px; margin: 12px 0 20px;
       background: linear-gradient(135deg, rgba(91,123,213,.12), rgba(46,157,91,.08));}
.hero h2 {margin: 0 0 8px 0; font-size: 2rem;}
.hero p {margin: 4px 0; font-size: 1.05rem; opacity: .82;}
.stage {border: 1px solid rgba(128,128,128,.22); border-radius: 14px; padding: 16px; min-height: 140px;}
.stage h4 {margin: 0 0 8px 0;}
.stage p {font-size: .92rem; opacity: .78;}
.stage-num {color: #5b7bd5; font-weight: 700;}
</style>
"""


# ---------------------------------------------------------------- loading


@st.cache_resource(show_spinner="Loading model...")
def _load_predictor(checkpoint: str, device: str, version: int):
    from av_contradiction.inference.predictor import Predictor

    return Predictor(checkpoint, device)


def load_predictor(checkpoint: str, device: str):
    return _load_predictor(checkpoint, device, CACHE_VERSION)


@st.cache_resource(show_spinner="Loading the CLIP + CLAP text encoders...")
def _load_text_encoders(checkpoint: str, device: str, version: int):
    from av_contradiction.inference.text_analysis import TextEncoders

    pred = load_predictor(checkpoint, device)
    return TextEncoders(pred.cfg, pred.device)


@st.cache_resource(show_spinner="Loading Whisper...")
def _load_transcriber(checkpoint: str, device: str, version: int):
    import torch
    from av_contradiction.inference.text_analysis import Transcriber

    pred = load_predictor(checkpoint, device)
    # Whisper is an optional explanation feature. Keep it on CPU so enabling
    # transcription cannot evict the CUDA prediction/text models on small GPUs.
    return Transcriber(pred.cfg, torch.device("cpu"))


def default_model_name() -> str | None:
    f = ROOT / "models" / "default.txt"
    return f.read_text(encoding="utf-8").strip() if f.is_file() else None


def find_checkpoints() -> list[str]:
    """Released models (models/) first, then local training runs."""
    base = ROOT / "models"
    released = sorted(str(p) for p in base.glob("*/best.pt")) if base.exists() else []
    default = default_model_name()
    released.sort(key=lambda p: Path(p).parent.name != default)
    runs = ROOT / "output" / "checkpoints"
    local = sorted(str(p) for p in runs.rglob("best.pt")) if runs.exists() else []
    return released + local


def model_label(ckpt: str) -> str:
    name = Path(ckpt).parent.name
    info = Path(ckpt).with_name("best_model.json")
    if info.is_file():
        b = json.loads(info.read_text(encoding="utf-8"))
        return f"{name} · {b['preset']} · test F1 {b['test_macro_f1']:.3f}"
    return name


def find_report(pred, checkpoint: str) -> tuple[Path, dict] | None:
    """Test report next to the checkpoint (e.g. a best_model downloaded from Kaggle), else where training wrote it."""
    for out in (Path(checkpoint).with_name("metrics_test.json"),
                ROOT / pred.cfg.evaluation.output_dir / pred.cfg.experiment_name / "metrics_test.json"):
        if out.is_file():
            return out, json.loads(out.read_text(encoding="utf-8"))
    return None


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:  # file outside the project folder
        return str(path)


def run_local(checkpoint: str, device: str, video: Path):
    return load_predictor(checkpoint, device).predict_clip(video)


def text_analysis_for(ckpt: str, device: str, feats, clip, video: Path, transcribe: bool = True):
    """(TextAnalysis or None, note). Without Whisper the analysis is returned without a transcript, with a note."""
    from av_contradiction.inference.text_analysis import analyse

    try:
        text = _load_text_encoders(ckpt, device, CACHE_VERSION)
    except Exception as exc:
        return None, f"Text mining unavailable: {exc}"
    transcript, note = None, ""
    if transcribe:
        try:
            transcript = _load_transcriber(ckpt, device, CACHE_VERSION)(video)
        except Exception as exc:
            note = (f"Speech transcript unavailable ({exc}). Download Whisper once with "
                    "`python -m av_contradiction.cli.download_encoders --asr`.")
    return analyse(feats, clip.frame_ts, clip.audio_ts, text, transcript), note


def waveform(video: Path, points: int = 1500) -> pd.DataFrame:
    from av_contradiction.data.preprocessing import extract_audio

    sr = 8000
    wav = extract_audio(video, sr)
    if wav.size == 0:
        return pd.DataFrame({"time (s)": [], "amplitude": []})
    step = max(1, wav.size // points)
    env = np.abs(wav[: wav.size - wav.size % step].reshape(-1, step)).max(1)
    return pd.DataFrame({"time (s)": np.arange(env.size) * step / sr, "amplitude": env})


def analyse_upload(video: Path, main_ckpt: str, device: str, transcribe: bool = True) -> dict[str, Any]:
    """Run the released model on the video, then text mining on its features."""
    pred = load_predictor(main_ckpt, device)
    res, clip = pred.predict_clip(video)
    out: dict[str, Any] = {"models": [{"ckpt": main_ckpt, "res": res, "clip": clip,
                                        "feats": pred.last_features, "attention": pred.last_attention}]}
    first = out["models"][0]
    out["text"], out["text_note"] = text_analysis_for(main_ckpt, device, first["feats"], first["clip"], video, transcribe)
    out["wave"] = waveform(video)
    out["size_mb"] = video.stat().st_size / 1e6
    return out


# ---------------------------------------------------------------- small components


def card(col, key: str, value: str) -> None:
    col.markdown(f'<div class="card"><div class="k">{key}</div><div class="v">{value}</div></div>', unsafe_allow_html=True)


def least_supported_moments(res: PredictionResult, top: int = 3) -> list[dict]:
    """Audio windows whose cross-attention went most to the "no matching visual evidence" token."""
    windows = {w.index: w for w in res.evidence.audio_windows}
    ranked = sorted(res.evidence.attention_links, key=lambda a: a.null_weight, reverse=True)[:top]
    return [
        {
            "audio": f"{windows[a.audio_window_index].start_sec:.1f}-{windows[a.audio_window_index].end_sec:.1f}s",
            "no-visual-match weight": round(a.null_weight, 3),
            "closest attended frame": f"#{a.video_frame_index} @ {a.video_time_sec:.2f}s" if a.video_frame_index is not None else "none",
        }
        for a in ranked
    ]


def render_verdict(title: str, res: PredictionResult) -> None:
    colour = COLOURS[res.label]
    meaning = {
        "consistent": "The model found more evidence that the audio and video agree.",
        "contradictory": "The model found more evidence that the audio and video do not agree.",
        "uncertain": "The model did not find enough separation to make a confident decision.",
    }[res.label]
    st.markdown(
        f'<div class="verdict" style="background:{colour}"><p>{title}</p><h2>{res.label.upper()}</h2>'
        f"<p>contradiction score {res.contradiction_probability:.2f} · confidence {res.confidence:.2f}</p></div>",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    c1.metric("Contradiction score", f"{res.contradiction_probability:.3f}",
              help="Calibrated (temperature-scaled on validation)" if res.probability_calibrated else "Raw score, not calibrated")
    c2.metric("Confidence", f"{res.confidence:.3f}")
    st.progress(min(1.0, max(0.0, res.contradiction_probability)), text="0 = consistent evidence · 1 = contradictory evidence")
    st.info(meaning)
    st.caption(res.confidence_interpretation)
    for w in res.warnings:
        st.warning(w)
    if res.evidence.attention_links:
        st.markdown("**Model attention signals**")
        st.caption("These timestamps are inspection hints, not proof that the audio is contradictory.")
        st.table(least_supported_moments(res))


def heatmap(matrix: np.ndarray, x_labels: list[str], y_labels: list[str], x_title: str, y_title: str, scheme: str = "viridis"):
    rows = [{"x": x_labels[j], "y": y_labels[i], "value": float(matrix[i, j])}
            for i in range(matrix.shape[0]) for j in range(matrix.shape[1])]
    return (
        alt.Chart(pd.DataFrame(rows))
        .mark_rect()
        .encode(x=alt.X("x:N", sort=x_labels, title=x_title), y=alt.Y("y:N", sort=y_labels, title=y_title),
                color=alt.Color("value:Q", scale=alt.Scale(scheme=scheme)),
                tooltip=["x", "y", alt.Tooltip("value:Q", format=".3f")])
        .properties(height=320)
    )


def cosine(x: np.ndarray) -> np.ndarray:
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    return x @ x.T


def pca2(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(0)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T


def tag_chart(tags, colour: str):
    df = pd.DataFrame({"concept": [t.concept for t in tags], "share": [t.score for t in tags]})
    return (alt.Chart(df).mark_bar(color=colour)
            .encode(x=alt.X("share:Q", title="relative similarity"), y=alt.Y("concept:N", sort="-x", title=None),
                    tooltip=["concept", alt.Tooltip("share:Q", format=".2f")])
            .properties(height=36 * max(1, len(df)) + 20))


def visual_match(rank: int) -> str:
    return "visible" if rank <= 10 else "unclear" if rank <= 40 else "not visible"


def role(ckpt: str) -> str:
    name = Path(ckpt).parent.name
    return MODEL_ROLES.get(name, name)


# ---------------------------------------------------------------- tabs


def render_start() -> None:
    st.markdown(
        '<div class="hero"><h2>See whether the audio matches the video</h2>'
        '<p>Upload a short video, then follow the result from the raw inputs to the model decision.</p></div>',
        unsafe_allow_html=True,
    )
    st.subheader("What you will see after analysis")
    cols = st.columns(4)
    stages = (
        ("1", "Verdict", "A clear consistent, contradictory, or uncertain result with a calibrated score."),
        ("2", "Input timeline", "The exact frames, audio windows, timestamps, and silence checks used."),
        ("3", "What is seen and heard", "Human-readable visual tags, sound tags, and an optional speech transcript."),
        ("4", "Model signals", "Feature patterns and attention signals that help inspect how the model routed information."),
    )
    for col, (number, title, description) in zip(cols, stages, strict=True):
        col.markdown(
            f'<div class="stage"><div class="stage-num">STEP {number}</div><h4>{title}</h4><p>{description}</p></div>',
            unsafe_allow_html=True,
        )
    st.info("The Dataset tab explains the training data. Every other tab becomes video-specific after you press Analyze.")


def tab_overview(a: dict[str, Any] | None) -> None:
    if a is None:
        render_start()
        return
    st.subheader("1. Verdict for this video")
    st.caption("This is the model's prediction for the uploaded video. The score is not a guarantee outside the evaluated benchmark.")
    first = a["models"][0]
    res = first["res"]
    c = st.columns(3)
    card(c[0], "uploaded video", a.get("video_name", "uploaded video"))
    card(c[1], "duration", f"{res.duration_sec:.1f} s")
    card(c[2], "analysis time", f"{sum(res.timing_ms.values()) / 1000:.1f} s")
    cols = st.columns(len(a["models"]))
    for col, m in zip(cols, a["models"], strict=True):
        with col:
            render_verdict(role(m["ckpt"]), m["res"])
    ta = a["text"]
    if ta is not None:
        st.markdown("#### In words")
        c1, c2, c3 = st.columns(3)
        c1.markdown("**Seen**<br>" + " ".join(f'<span class="pill">{t.concept}</span>' for t in ta.seen[:4]), unsafe_allow_html=True)
        c2.markdown("**Heard**<br>" + (" ".join(f'<span class="pill">{t.concept}</span>' for t in ta.heard[:4]) or "silence"),
                    unsafe_allow_html=True)
        said = ", ".join(w for w, _ in ta.keywords[:6]) if ta.keywords else ("no speech" if ta.transcript is not None else "–")
        c3.markdown(f"**Said (keywords)**<br>{said}", unsafe_allow_html=True)
        missing = [c.phrase for c in ta.cross_checks if visual_match(c.visual_rank) == "not visible"]
        if missing:
            st.markdown("Heard or said but **not visible** in the frames: " + ", ".join(f"*{p}*" for p in missing))


def tab_dataset() -> None:
    st.markdown("#### Training data")
    st.write(
        "AVID-Bench subset: 2,600 YouTube segments forming 1,300 matched pairs (the same clip with its original "
        "audio, and with audio injected by the AVID authors), one pair per source video, split by source video so no "
        "video appears in two splits. The audio-swap benchmark reuses the original clips: each clip once with its own "
        "audio and once with the audio of another clip of the same length."
    )
    shown = False
    for title, path, col in (("AVID-Bench subset (edited audio)", ROOT / "manifests" / "avid_subset.csv", "injection_type"),):
        if not path.is_file():
            continue
        shown = True
        df = pd.read_csv(path)
        st.markdown(f"**{title}** · `{display_path(path)}` · {len(df):,} clips")
        c1, c2 = st.columns(2)
        counts = df.groupby(["split", "label"]).size().reset_index(name="clips")
        c1.altair_chart(alt.Chart(counts).mark_bar().encode(
            x=alt.X("split:N", sort=["train", "val", "test"]), xOffset="label:N", y="clips:Q",
            color=alt.Color("label:N", scale=alt.Scale(domain=["consistent", "contradictory"],
                                                       range=[COLOURS["consistent"], COLOURS["contradictory"]])),
            tooltip=["split", "label", "clips"]).properties(height=250, title="clips per split and label"), width="stretch")
        if col and col in df:
            types = df[df[col].notna()].groupby(col).size().reset_index(name="clips")
            c2.altair_chart(alt.Chart(types).mark_bar(color="#5b7bd5").encode(
                y=alt.Y(f"{col}:N", title=None, sort="-x"), x="clips:Q", tooltip=[col, "clips"]
            ).properties(height=250, title="contradiction types (injected audio)"), width="stretch")
    if not shown:
        st.info("No manifests found under manifests/.")
def tab_preprocessing(a: dict[str, Any] | None, pred) -> None:
    if a is None:
        st.info("Upload and analyze a video to see how it is decoded.")
        return
    st.subheader("2. Input and preprocessing")
    st.caption("This is exactly how the uploaded video was converted into frames and audio windows before prediction.")
    clip, pre = a["models"][0]["clip"], pred.cfg.preprocess
    c = st.columns(5)
    card(c[0], "duration", f"{clip.duration_sec:.1f} s")
    card(c[1], "file size", f"{a['size_mb']:.1f} MB")
    card(c[2], "frames sampled", f"{len(clip.frame_ts)}")
    card(c[3], "audio windows", f"{len(clip.audio_ts)}")
    card(c[4], "silent windows", f"{int(clip.audio_silent.sum())}")
    st.markdown("#### Steps")
    for step in (
        f"**Decode** with FFmpeg; reject files shorter than {pre.min_duration_sec} s or longer than {pre.max_duration_sec:.0f} s.",
        f"**Frames:** {pre.num_frames} frames at evenly spaced exact timestamps, resized to {pre.frame_size}×{pre.frame_size}.",
        f"**Audio:** mono, resampled to {pre.audio_sample_rate / 1000:g} kHz, cut into {pre.audio_window_sec:g} s windows "
        f"every {pre.audio_hop_sec:g} s (at most {pre.max_audio_windows}).",
        f"**Silence:** windows with RMS below {pre.silence_rms_threshold:g} are flagged; a fully silent clip returns *uncertain*.",
    ):
        st.markdown(f'<div class="step">{step}</div>', unsafe_allow_html=True)
    st.markdown("#### Sampled frames")
    if clip.has_video:
        cols = st.columns(8)
        for i, (f, t) in enumerate(zip(clip.frames, clip.frame_ts, strict=True)):
            cols[i % 8].image(f, caption=f"#{i} · {t:.1f}s", width="stretch")
    else:
        st.write("No frames decoded.")
    st.markdown("#### Audio")
    if len(a["wave"]):
        st.markdown("**Audio waveform**")
        st.caption("Higher amplitude means more audio energy at that moment; this is not a semantic label.")
        st.altair_chart(alt.Chart(a["wave"]).mark_area(opacity=.7, color="#5b7bd5").encode(
            x="time (s):Q", y=alt.Y("amplitude:Q", title="audio amplitude")).properties(height=160, title="Audio amplitude over time"),
            width="stretch")
    if len(clip.audio_ts):
        rms = np.sqrt((clip.audio_windows.astype(np.float64) ** 2).mean(1))
        win = pd.DataFrame({"window": [f"{s:.0f}-{e:.0f}s" for s, e in clip.audio_ts], "RMS loudness": rms,
                            "silent": clip.audio_silent.astype(bool)})
        st.altair_chart(alt.Chart(win).mark_bar().encode(
            x=alt.X("window:N", sort=list(win["window"]), title=f"{pre.audio_window_sec:g} s audio windows"),
            y="RMS loudness:Q", color=alt.Color("silent:N", scale=alt.Scale(domain=[False, True], range=["#5b7bd5", "#bbbbbb"])),
            tooltip=["window", alt.Tooltip("RMS loudness:Q", format=".4f"), "silent"]).properties(height=200, title="loudness per window"),
            width="stretch")
        st.caption("Blue = audio detected. Grey = window flagged as silent by the preprocessing threshold.")
    else:
        st.write("No audio track.")


def tab_features(a: dict[str, Any] | None, pred) -> None:
    if a is None:
        st.info("Upload and analyze a video to see its features.")
        return
    st.subheader("5. Feature patterns (technical)")
    st.caption("These plots show how the frozen encoders represent changes over time. They are diagnostic views, not direct explanations of the verdict.")
    m = a["models"][0]
    feats, clip, enc = m["feats"], m["clip"], pred.cfg.encoders
    video, audio = feats["video"].float().numpy(), feats["audio"].float().numpy()
    c = st.columns(4)
    card(c[0], "frame encoder", enc.video_model_name.split("/")[-1] if enc.video == "clip" else enc.video)
    card(c[1], "frame features", f"{video.shape[0]} × {video.shape[1]}")
    card(c[2], "audio encoder", enc.audio_model_name.split("/")[-1] if enc.audio == "clap" else enc.audio)
    card(c[3], "audio features", f"{audio.shape[0]} × {audio.shape[1]}")
    st.caption(f"Both encoders are frozen; only the small fusion model was trained. Decoding took "
               f"{m['res'].timing_ms.get('decode_ms', 0) / 1000:.1f} s, encoding {m['res'].timing_ms.get('encode_ms', 0) / 1000:.1f} s.")
    fl = [f"{t:.1f}s" for t in clip.frame_ts]
    al = [f"{s:.0f}-{e:.0f}s" for s, e in clip.audio_ts]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Frame similarity**")
        st.caption("Brighter blocks mean two sampled frames have more similar encoder representations.")
        if len(video):
            st.altair_chart(heatmap(cosine(video), fl, fl, "frame time", "frame time"), width="stretch")
    with c2:
        st.markdown("**Audio-window similarity**")
        st.caption("Brighter blocks mean two audio windows have more similar encoder representations.")
        if len(audio):
            st.altair_chart(heatmap(cosine(audio), al, al, "audio window", "audio window", "magma"), width="stretch")
    with st.expander("Technical diagnostics: embedding trajectories", expanded=False):
        st.caption("PCA compresses high-dimensional features into two axes. Nearby points are more similar in the compressed view; the axes themselves have no human meaning.")
        c1, c2 = st.columns(2)
        for col, x, times, name in ((c1, video, clip.frame_ts, "frames"),
                                    (c2, audio, clip.audio_ts.mean(1) if len(audio) else np.zeros(0), "audio windows")):
            if len(x) >= 2:
                p = pca2(x)
                df = pd.DataFrame({"PC1": p[:, 0], "PC2": p[:, 1], "time (s)": times})
                col.altair_chart(alt.Chart(df).mark_line(point=True).encode(
                    x="PC1:Q", y="PC2:Q", order="time (s):Q", color=alt.Color("time (s):Q", scale=alt.Scale(scheme="plasma")),
                    tooltip=[alt.Tooltip("time (s):Q", format=".1f")]).properties(height=260, title=name), width="stretch")


def render_text_mining(ta, note: str = "") -> None:
    """Zero-shot words for what the frames show and the audio sounds like, plus the transcript and keywords."""
    st.subheader("3. What is seen, heard, and said")
    st.caption("These are CLIP/CLAP similarity hints and an optional Whisper transcript. They help describe the video but do not change the model's verdict or prove that an object is present.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Seen in the frames · similarity hints**")
        st.caption("Relative similarity to the visual concept vocabulary; not object-detection confidence.")
        if ta.seen:
            st.altair_chart(tag_chart(ta.seen, "#2e9d5b"), width="stretch")
    with c2:
        st.markdown("**Heard in the audio · similarity hints**")
        st.caption("Relative similarity to the sound concept vocabulary; not a guaranteed sound identification.")
        if ta.heard:
            st.altair_chart(tag_chart(ta.heard, "#5b7bd5"), width="stretch")
        else:
            st.write("(silence)")
    st.markdown("**Speech transcript (Whisper)**")
    if ta.transcript is None:
        st.info(note or "Transcript not computed.")
    elif not ta.transcript:
        st.write("(no speech recognised)")
    else:
        words = {w for w, _ in ta.keywords}
        marked = " ".join(f"**{w}**" if w.lower().strip(".,!?;:'\"") in words else w for w in ta.transcript.split())
        st.markdown(f"> {marked}")
        if ta.keywords:
            kw = pd.DataFrame(ta.keywords, columns=["keyword", "count"])
            st.altair_chart(alt.Chart(kw).mark_bar(color="#e0a030").encode(
                x=alt.X("count:Q", axis=alt.Axis(tickMinStep=1)), y=alt.Y("keyword:N", sort="-x", title=None))
                .properties(height=30 * len(kw) + 20, title="keywords (stopwords removed)"), width="stretch")
    if ta.cross_checks:
        st.markdown("**Does the visual content resemble what was heard or said?**")
        st.dataframe(pd.DataFrame([{"phrase": c.phrase, "from": c.source, "visual rank": f"{c.visual_rank} of {c.n_concepts}",
                                    "visual match": visual_match(c.visual_rank)} for c in ta.cross_checks]),
                     hide_index=True, width="stretch")
        st.caption("Each phrase is compared with the frames using CLIP. This is a similarity heuristic: an off-screen sound or object can still be valid.")
    with st.expander("Seen and heard over time"):
        c1, c2 = st.columns(2)
        c1.dataframe(pd.DataFrame(ta.seen_timeline), hide_index=True, width="stretch")
        c2.dataframe(pd.DataFrame(ta.heard_timeline), hide_index=True, width="stretch")


def tab_text(a: dict[str, Any] | None) -> None:
    if a is None:
        st.info("Upload and analyze a video to see the text mining.")
    elif a["text"] is None:
        st.info(a["text_note"])
    else:
        render_text_mining(a["text"], a["text_note"])


def tab_model(a: dict[str, Any] | None, device: str) -> None:
    from av_contradiction.models.classifier import count_parameters

    if a is None:
        st.info("Upload and analyze a video to inspect the models on it.")
        return
    st.subheader("4. Model signals")
    st.caption("This section shows what the fusion model inspected while combining audio and video. Attention is an inspection aid, not a causal explanation.")
    for m in a["models"]:
        pred = load_predictor(m["ckpt"], device)
        mc, name = pred.cfg.model, Path(m["ckpt"]).parent.name
        enc = pred.cfg.encoders
        video_encoder = enc.video_model_name if enc.video == "clip" else enc.video
        audio_encoder = enc.audio_model_name if enc.audio == "clap" else enc.audio
        st.markdown("#### Architecture used for this video")
        pipeline = (
            f"frames → **{video_encoder}** (frozen) · audio windows → **{audio_encoder}** (frozen) → "
            "projection + time encoding"
        )
        if mc.use_cross_attention:
            pipeline += " → **audio↔video cross-attention**"
        pipeline += f" → **{mc.fusion} fusion** → classifier → verdict"
        st.markdown(pipeline)
        st.markdown(f"#### {name} · `{pred.ckpt.get('preset', '')}`")
        st.caption(role(m["ckpt"]))
        c = st.columns(5)
        card(c[0], "parameters", f"{count_parameters(pred.model) / 1e6:.2f} M")
        card(c[1], "cross-attention", "yes" if mc.use_cross_attention else "no")
        card(c[2], "fusion", mc.fusion)
        card(c[3], "time alignment", mc.alignment if mc.use_cross_attention else "–")
        card(c[4], "contradiction score", f"{m['res'].contradiction_probability:.2f}")
        att, clip = m["attention"], m["clip"]
        if att is not None:
            xl = ["no match"] + [f"{t:.1f}s" for t in clip.frame_ts][: att.shape[1] - 1]
            yl = [f"{s:.0f}-{e:.0f}s" for s, e in clip.audio_ts][: att.shape[0]]
            st.markdown("**Audio → video attention** · last layer")
            st.caption("Rows are audio windows. The first column represents the learned 'no matching visual evidence' token; brighter cells indicate more attention mass.")
            st.markdown("**Highest model attention signals**")
            st.table(least_supported_moments(m["res"]))
            st.altair_chart(heatmap(att[: len(yl), : len(xl)], xl, yl, "attended frame", "audio window", "blues"),
                            width="stretch")
        else:
            st.caption("This model has no cross-attention (it fuses pooled audio and video features), so there is no "
                       "attention map.")


def tab_evaluation() -> None:
    st.markdown("#### Released models (held-out test split)")
    rows = []
    for ckpt in find_checkpoints():
        rep = Path(ckpt).with_name("metrics_test.json")
        if rep.is_file():
            met = json.loads(rep.read_text(encoding="utf-8"))
            m = met["metrics"]
            rows.append({"model": Path(ckpt).parent.name, "role": role(ckpt), "preset": met.get("preset", ""), "test clips": m["n"],
                         "macro-F1": round(m["macro_f1"], 3), "accuracy": round(m["accuracy"], 3),
                         "ROC-AUC": round(m["roc_auc"], 3), "ECE": round(m["ece"], 3)})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    for title, folder, text in (("AVID-Bench subset (edited audio)", "avid",
                                 "Audio-only already reaches 0.96: the injected audio can be recognised without looking at the video."),):
        md = ROOT / "results" / folder / "ablation_summary.md"
        if not md.is_file():
            continue
        st.markdown(f"#### {title}")
        st.caption(text + " Mean ± std over 3 seeds; models selected on validation, reported once on test.")
        st.markdown(md.read_text(encoding="utf-8"))
        png = md.with_name("ablation_macro_f1.png")
        if png.is_file():
            st.image(str(png), width=640)
# ---------------------------------------------------------------- page


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    found = find_checkpoints()
    env_ckpt = os.environ.get("AVC_CHECKPOINT")
    if env_ckpt:
        env_ckpt = str(Path(env_ckpt).resolve())  # same form as the discovered paths, so it is listed once
        if env_ckpt not in found:
            found.insert(0, env_ckpt)
    with st.sidebar:
        st.markdown("## 🎬 AV Contradiction Analytics")
        st.caption("Does what you hear match what you see?")
        if not found:
            st.error("No trained model found under models/ or output/checkpoints/.")
            return
        with st.expander("Advanced settings"):
            main_ckpt = st.selectbox("Model", found, index=found.index(env_ckpt) if env_ckpt in found else 0,
                                     format_func=model_label)
            device = st.selectbox("Device", ["auto", "cuda", "cpu"])
        st.caption(f"Loaded model: `{Path(main_ckpt).parent.name}`")
        st.markdown("**How to use**")
        st.caption("1. Upload a short video\n2. Press Analyze\n3. Start with Overview\n4. Use the other tabs to inspect each stage")
        if st.button("Reset analysis", use_container_width=True):
            st.session_state.pop("analysis", None)
            st.session_state.pop("video_upload", None)
            st.rerun()
    try:
        pred = load_predictor(main_ckpt, device)
    except Exception as exc:
        st.error(f"Could not load {main_ckpt}: {exc}")
        return

    st.title("Audio-Visual Semantic Contradiction Detector")
    st.caption("Upload a video to inspect what is seen, heard, and predicted.")

    limits = pred.cfg.inference
    up_col, btn_col = st.columns([4, 1])
    up = up_col.file_uploader("Upload a short video", type=[e.lstrip(".") for e in limits.allowed_extensions], key="video_upload")
    analysis = None
    if up is not None:
        if up.size > limits.max_upload_mb * 1024 * 1024:
            st.error(f"File exceeds the {limits.max_upload_mb:.0f} MB limit (inference.max_upload_mb).")
        else:
            key = hashlib.sha1(up.getbuffer()).hexdigest() + main_ckpt + device
            btn_col.write("")
            btn_col.write("")
            if btn_col.button("Analyze", type="primary", width="stretch"):
                with tempfile.NamedTemporaryFile(suffix=Path(up.name).suffix.lower(), delete=False) as tmp:
                    tmp.write(up.getbuffer())
                    tmp_path = Path(tmp.name)
                try:
                    with st.status("Decoding, extracting features, predicting, transcribing...", expanded=False) as status:
                        result = analyse_upload(tmp_path, main_ckpt, device)
                        result["video_name"] = up.name
                        st.session_state["analysis"] = (key, result)
                        status.update(label="Done", state="complete")
                except Exception as exc:
                    st.error(f"Processing failed: {exc}")
                finally:
                    tmp_path.unlink(missing_ok=True)
            saved = st.session_state.get("analysis")
            if saved and saved[0] == key:
                analysis = saved[1]
            with st.expander("Uploaded video", expanded=analysis is None):
                st.video(up)

    if analysis is None:
        tabs = st.tabs(["Start here", "Dataset"])
        with tabs[0]:
            render_start()
        with tabs[1]:
            tab_dataset()
    else:
        tabs = st.tabs(["Overview", "Input & preprocessing", "What is seen/heard", "Model signals", "Feature patterns", "Dataset"])
        with tabs[0]:
            tab_overview(analysis)
        with tabs[1]:
            tab_preprocessing(analysis, pred)
        with tabs[2]:
            tab_text(analysis)
        with tabs[4]:
            tab_features(analysis, pred)
        with tabs[3]:
            tab_model(analysis, device)
        with tabs[5]:
            tab_dataset()


if __name__ == "__main__":  # streamlit runs the script as __main__
    main()
