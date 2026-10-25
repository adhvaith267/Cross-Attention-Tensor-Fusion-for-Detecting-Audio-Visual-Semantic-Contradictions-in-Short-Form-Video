"""Integration tests with real pretrained encoders. Deselected by default (``-m integration`` to run).

Require CLIP/CLAP weights in the local Hugging Face cache; skipped otherwise. Nothing is downloaded.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from av_contradiction.config import build_config

pytestmark = pytest.mark.integration


def _cfg():
    return build_config({"encoders": {"video": "clip", "audio": "clap", "local_files_only": True},
                         "preprocess": {"num_frames": 4, "audio_sample_rate": 48000}, "hardware": {"profile": "none"}})


@pytest.fixture(scope="module")
def extractor():
    from av_contradiction.data.features import FeatureExtractor

    try:
        return FeatureExtractor(_cfg(), "cuda" if torch.cuda.is_available() else "cpu")
    except (OSError, ImportError) as exc:
        pytest.skip(f"Pretrained encoders unavailable: {exc}")


def test_clip_clap_shapes(extractor):
    from av_contradiction.data.preprocessing import DecodedClip

    clip = DecodedClip(
        duration_sec=4.0,
        frames=np.random.randint(0, 255, (4, 224, 224, 3), dtype=np.uint8),
        frame_ts=np.array([0.5, 1.5, 2.5, 3.5], np.float32),
        audio_windows=(0.1 * np.random.randn(3, 96000)).astype(np.float32),
        audio_ts=np.array([[0, 2], [1, 3], [2, 4]], np.float32),
        audio_silent=np.zeros(3, bool), sample_rate=48000,
    )
    feats = extractor.encode(clip)
    assert feats["video"].shape == (4, extractor.video_encoder.dim)
    assert feats["audio"].shape == (3, extractor.audio_encoder.dim)
    assert torch.isfinite(feats["video"]).all() and torch.isfinite(feats["audio"]).all()


def test_real_file_through_clip_clap(extractor, tmp_path):
    from av_contradiction.data.synthetic import render_clip

    path = Path(tmp_path) / "c.mp4"
    render_clip(path, ((200, 30, 30), (30, 30, 200)), (440.0, 880.0), seg_sec=2.0, size=128, sample_rate=48000)
    feats, clip = extractor.extract(path)
    assert feats["video"].shape[0] == 4 and feats["audio"].shape[0] >= 1
