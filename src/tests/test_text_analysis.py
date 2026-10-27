"""Text view: keyword extraction, zero-shot tags and the heard/said -> seen cross-check (no real models)."""

from pathlib import Path

import numpy as np
import pytest
import torch

from av_contradiction.inference.text_analysis import analyse, keywords, top_tags, visual_rank
from av_contradiction.inference.vocabulary import OFF_SCREEN_SOUNDS, SOUND_CONCEPTS, VISUAL_CONCEPTS

st_testing = pytest.importorskip("streamlit.testing.v1")
APP = str(Path(__file__).resolve().parents[1] / "av_contradiction" / "serving" / "dashboard.py")


class FakeText:
    """Stand-in for TextEncoders: concept i is the i-th basis vector; other phrases are hashed to one."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.visual_concepts = torch.eye(dim)[: len(VISUAL_CONCEPTS)]
        self.audio_concepts = torch.eye(dim)[: len(SOUND_CONCEPTS)]

    def visual(self, texts: list[str]) -> torch.Tensor:
        out = []
        for t in texts:
            name = t.removeprefix("a photo of ")
            i = VISUAL_CONCEPTS.index(name) if name in VISUAL_CONCEPTS else len(VISUAL_CONCEPTS) + hash(name) % 50
            out.append(torch.eye(self.dim)[i])
        return torch.stack(out)


def _feats(seen: str, heard: str, n_frames: int = 4, n_win: int = 3, silent: list[bool] | None = None) -> dict:
    text = FakeText()
    video = text.visual_concepts[VISUAL_CONCEPTS.index(seen)].repeat(n_frames, 1)
    audio = text.audio_concepts[SOUND_CONCEPTS.index(heard)].repeat(n_win, 1)
    return {"video": video, "audio": audio, "audio_silent": torch.tensor(silent or [False] * n_win)}


def test_keywords_drop_stopwords_and_count():
    kw = keywords("The dog is barking. The DOG wants a ball, and the ball is red!")
    assert kw[:2] == [("dog", 2), ("ball", 2)]
    assert all(w not in ("the", "is", "and", "a") for w, _ in kw)


def test_top_tags_and_visual_rank():
    emb = torch.eye(8)
    tags = top_tags(emb[[2, 2]], emb, [f"c{i}" for i in range(8)], 3)
    assert tags[0].concept == "c2" and tags[0].score > 0.9
    assert visual_rank(emb[[2]], emb[2], emb) == (1, 1.0)
    assert visual_rank(emb[[2]], emb[5], emb)[0] > 1


def test_analyse_tags_and_cross_check():
    ta = analyse(_feats("a whale", "a dog barking"), np.arange(4.0), np.array([[0, 2], [1, 3], [2, 4]], float), FakeText(),
                 transcript="A dog, a dog and a whale.")
    assert ta.seen[0].concept == "a whale" and ta.heard[0].concept == "a dog barking"
    checks = {c.phrase: c for c in ta.cross_checks}
    assert checks["a dog barking"].source == "heard" and checks["a dog barking"].visual_rank > 1  # heard, not seen
    assert checks["whale"].source == "said"
    assert ta.keywords[0] == ("dog", 2)
    assert len(ta.seen_timeline) == 4 and len(ta.heard_timeline) == 3


def test_analyse_skips_silence_and_off_screen_sounds():
    music = sorted(OFF_SCREEN_SOUNDS & set(SOUND_CONCEPTS))[0]
    ta = analyse(_feats("a kitchen", music, silent=[True, False, False]), np.arange(4.0),
                 np.array([[0, 2], [1, 3], [2, 4]], float), FakeText())
    assert ta.heard[0].concept == music
    assert all(c.phrase != music for c in ta.cross_checks)
    assert ta.heard_timeline[0]["heard"] == "(silence)"
    silent = analyse(_feats("a kitchen", music, silent=[True] * 3), np.arange(4.0), np.zeros((3, 2)), FakeText())
    assert silent.heard == [] and silent.transcript is None


def test_dashboard_renders_text_view():
    def script(app_path: str) -> None:
        import importlib.util

        import numpy as np

        from av_contradiction.inference.text_analysis import analyse
        from tests.test_text_analysis import FakeText, _feats

        spec = importlib.util.spec_from_file_location("dash", app_path)
        dash = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dash)
        ta = analyse(_feats("a whale", "a dog barking"), np.arange(4.0), np.array([[0, 2], [1, 3], [2, 4]], float),
                     FakeText(), transcript="a dog on a boat")
        dash.render_text_mining(ta)

    at = st_testing.AppTest.from_function(script, args=(APP,), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    assert any("What is seen, heard, and said" in s.value for s in at.subheader)
    assert any("Does the visual content resemble what was heard or said" in m.value for m in at.markdown)


def test_text_view_reports_unavailable_for_non_clip_models(trained_checkpoint, mini_dataset):
    """The tiny test model uses handcrafted encoders: the text mining must say so instead of failing."""

    def script(app_path: str, ckpt: str, video: str) -> None:
        import importlib.util
        from pathlib import Path

        import streamlit as st

        spec = importlib.util.spec_from_file_location("dash", app_path)
        dash = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dash)
        res, clip = dash.run_local(ckpt, "cpu", Path(video))
        feats = dash.load_predictor(ckpt, "cpu").last_features
        ta, note = dash.text_analysis_for(ckpt, "cpu", feats, clip, Path(video), transcribe=False)
        st.write("NONE" if ta is None else "TA")
        st.info(note)

    video = Path(mini_dataset["cfg"].data.root_dir) / mini_dataset["rows"][0]["video_path"]
    at = st_testing.AppTest.from_function(script, args=(APP, str(trained_checkpoint), str(video)), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    assert any(m.value == "NONE" for m in at.markdown)
    assert any("Text mining unavailable" in i.value for i in at.info)
