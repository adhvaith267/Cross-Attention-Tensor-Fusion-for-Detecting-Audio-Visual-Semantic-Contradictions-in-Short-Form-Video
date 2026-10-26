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
    """Render a real prediction through the dashboard's result view (AppTest cannot simulate uploads)."""

    def script(app_path: str, ckpt: str, video: str) -> None:
        import importlib.util
        from pathlib import Path

        spec = importlib.util.spec_from_file_location("dash", app_path)
        dash = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dash)
        res, clip = dash.run_local(ckpt, "cpu", Path(video))
        dash.show_result("test", res, clip)

    video = Path(mini_dataset["cfg"].data.root_dir) / mini_dataset["rows"][0]["video_path"]
    at = st_testing.AppTest.from_function(script, args=(APP, str(trained_checkpoint), str(video)), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    assert any("Contradiction score" in m.label for m in at.metric)
    assert any("least visual support" in m.value for m in at.markdown)
