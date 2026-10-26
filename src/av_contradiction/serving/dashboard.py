"""Streamlit dashboard for the AV contradiction detector.

Run from the project root: streamlit run src/av_contradiction/serving/dashboard.py
Loads a trained checkpoint in-process and predicts on an uploaded video.
All displayed numbers come from the loaded model or from saved evaluation reports; nothing is hardcoded.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import streamlit as st

from av_contradiction.inference.schemas import PredictionResult

# Checkpoints and reports are configured relative to the project root, which is where the app is launched.
ROOT = Path.cwd()

st.set_page_config(page_title="AV Contradiction Detector", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def load_predictor(checkpoint: str, device: str):
    from av_contradiction.inference.predictor import Predictor

    return Predictor(checkpoint, device)


def find_checkpoints() -> list[str]:
    """Released models (models/) first, then local training runs (output/checkpoints/)."""
    found: list[str] = []
    for base in (ROOT / "models", ROOT / "output" / "checkpoints"):
        if base.exists():
            found += sorted(str(p) for p in base.rglob("best.pt"))
    return found


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
    except ValueError:  # report outside the project folder
        return str(path)


def run_local(checkpoint: str, device: str, video: Path):
    return load_predictor(checkpoint, device).predict_clip(video)


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


def show_result(title: str, res: PredictionResult, clip) -> None:
    st.subheader(title)
    colour = {"contradictory": "red", "consistent": "green", "uncertain": "orange"}[res.label]
    st.markdown(f"### :{colour}[{res.label.upper()}]")
    st.metric("Contradiction score", f"{res.contradiction_probability:.3f}",
              help="Calibrated (temperature-scaled on validation)" if res.probability_calibrated else "Raw softmax score, not calibrated")
    st.progress(min(1.0, max(0.0, res.contradiction_probability)))
    st.caption(f"Confidence {res.confidence:.3f}. {res.confidence_interpretation}")
    for w in res.warnings:
        st.warning(w)

    if res.evidence.attention_links:
        st.markdown("**Moments with the least visual support**")
        st.table(least_supported_moments(res))
        st.caption("Audio windows whose attention went mostly to the model's 'no matching visual evidence' token. "
                   "A hint for where to look, not proof of where a contradiction is.")

    st.markdown("**Sampled frames** (model inputs, with timestamps)")
    if clip.has_video:
        cols = st.columns(min(8, len(clip.frames)))
        for i, (f, t) in enumerate(zip(clip.frames, clip.frame_ts, strict=True)):
            cols[i % len(cols)].image(f, caption=f"#{i} @ {t:.2f}s", width="stretch")
    else:
        st.write("No frames decoded.")

    with st.expander("Model", expanded=False):
        st.json(res.model.model_dump())
        st.write("Timing (ms):", res.timing_ms)
    with st.expander("Audio windows"):
        st.table([w.model_dump() for w in res.evidence.audio_windows])
    if res.evidence.attention_links:
        with st.expander("Cross-attention (audio window -> most-attended frame)"):
            st.info(res.evidence.note)
            st.table([a.model_dump() for a in res.evidence.attention_links])


def main() -> None:
    st.title("Audio-Visual Semantic Contradiction Detector")
    st.caption(
        "Research prototype. It predicts whether audio and visual content are consistent *on the benchmark it "
        "was trained on*. It is not a misinformation, deception or deepfake detector."
    )
    with st.sidebar:
        device = st.selectbox("Device", ["auto", "cpu", "cuda"])
        found = find_checkpoints()
        env_ckpt = os.environ.get("AVC_CHECKPOINT")
        if env_ckpt:
            env_ckpt = str(Path(env_ckpt).resolve())  # same form as the discovered paths, so it is listed once
        if env_ckpt and env_ckpt not in found:
            found.insert(0, env_ckpt)  # honour an explicitly configured checkpoint
        default = env_ckpt or (found[0] if found else "")
        ckpt = st.selectbox("Checkpoint", found, index=found.index(default) if default in found else 0) if found else st.text_input("Checkpoint path", default)
        compare = st.checkbox("Comparison mode (second model, e.g. a unimodal baseline)")
        ckpt2 = None
        if compare:
            ckpt2 = st.selectbox("Comparison checkpoint", found) if found else st.text_input("Comparison checkpoint path")

    if not ckpt or not Path(ckpt).is_file():
        st.error("No checkpoint found. Train one first (see README: python -m av_contradiction.cli.train) or enter a valid path.")
        return
    try:
        pred = load_predictor(ckpt, device)
    except Exception as exc:
        st.error(f"Could not load checkpoint: {exc}")
        return
    with st.expander("Model and dataset information", expanded=False):
        st.json(pred.info().model_dump())
        rep = find_report(pred, ckpt)
        if rep:
            path, metrics = rep
            m = metrics["metrics"]
            st.write(f"Held-out results from `{display_path(path)}` (split={metrics['split']}, n={m['n']}):")
            st.write({k: m[k] for k in ("macro_f1", "accuracy", "roc_auc", "pr_auc", "ece")})
        else:
            st.caption("No evaluation report found for this checkpoint (run python -m av_contradiction.cli.evaluate).")

    limits = pred.cfg.inference
    up = st.file_uploader("Upload a short video", type=[e.lstrip(".") for e in limits.allowed_extensions])
    if up is None:
        return
    if up.size > limits.max_upload_mb * 1024 * 1024:
        st.error(f"File exceeds the {limits.max_upload_mb:.0f} MB limit (inference.max_upload_mb).")
        return
    st.video(up)
    if not st.button("Analyze", type="primary"):
        return
    suffix = Path(up.name).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(up.getbuffer())
        tmp_path = Path(tmp.name)
    try:
        with st.status("Processing: decoding, encoding, predicting...", expanded=False) as status:
            res, clip = run_local(ckpt, device, tmp_path)
            results = [(f"Model: {Path(ckpt).parent.name}", res, clip)]
            if compare and ckpt2 and Path(ckpt2).is_file():
                res2, clip2 = run_local(ckpt2, device, tmp_path)
                results.append((f"Comparison: {Path(ckpt2).parent.name}", res2, clip2))
            status.update(label="Done", state="complete")
        cols = st.columns(len(results))
        for col, (title, r, c) in zip(cols, results, strict=True):
            with col:
                show_result(title, r, c)
    except Exception as exc:
        st.error(f"Processing failed: {exc}")
    finally:
        tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":  # streamlit runs the script as __main__
    main()
