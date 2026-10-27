"""Headless dashboard smoke test via Streamlit's AppTest (no browser)."""

from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")

APP = str(Path(__file__).resolve().parents[1] / "av_contradiction" / "serving" / "dashboard.py")


def test_dashboard_loads_with_checkpoint(trained_checkpoint, monkeypatch):
    monkeypatch.setenv("AVC_CHECKPOINT", str(trained_checkpoint))
    at = st_testing.AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception
    assert any("Contradiction Detector" in t.value for t in at.title)


def test_dashboard_renders_prediction(trained_checkpoint, mini_dataset):
    """Render a real prediction through the dashboard's verdict view (AppTest cannot simulate uploads)."""

    def script(app_path: str, ckpt: str, video: str) -> None:
        import importlib.util
        from pathlib import Path

        spec = importlib.util.spec_from_file_location("dash", app_path)
        dash = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dash)
        res, clip = dash.run_local(ckpt, "cpu", Path(video))
        dash.render_verdict("test", res)

    video = Path(mini_dataset["cfg"].data.root_dir) / mini_dataset["rows"][0]["video_path"]
    at = st_testing.AppTest.from_function(script, args=(APP, str(trained_checkpoint), str(video)), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    assert any("Contradiction score" in m.label for m in at.metric)
    assert any("Model attention signals" in m.value for m in at.markdown)


def test_dashboard_all_tabs(trained_checkpoint, mini_dataset):
    """Analyse one clip and render every tab. The tiny test model uses handcrafted encoders,
    so the text-mining tab must explain that it is unavailable instead of failing."""

    def script(app_path: str, ckpt: str, video: str) -> None:
        import importlib.util
        from pathlib import Path

        import streamlit as st

        spec = importlib.util.spec_from_file_location("dash", app_path)
        dash = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dash)
        a = dash.analyse_upload(Path(video), ckpt, "cpu", transcribe=False)
        pred = dash.load_predictor(ckpt, "cpu")
        dash.tab_overview(a)
        dash.tab_dataset()
        dash.tab_preprocessing(a, pred)
        dash.tab_features(a, pred)
        dash.tab_text(a)
        dash.tab_model(a, "cpu")
        dash.tab_evaluation()
        st.write("DONE")

    video = Path(mini_dataset["cfg"].data.root_dir) / mini_dataset["rows"][0]["video_path"]
    at = st_testing.AppTest.from_function(script, args=(APP, str(trained_checkpoint), str(video)), default_timeout=180)
    at.run()
    assert not at.exception, at.exception
    assert sum("Contradiction score" in m.label for m in at.metric) == 1
    assert any("Text mining unavailable" in i.value for i in at.info)
    assert any("Audio → video attention" in m.value for m in at.markdown)
    assert any(m.value == "DONE" for m in at.markdown)
