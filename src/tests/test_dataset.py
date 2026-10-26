import subprocess

import numpy as np
import pytest
import torch

from av_contradiction.config import PreprocessConfig
from av_contradiction.data.dataset import collate_features
from av_contradiction.data.manifest import read_manifest, write_manifest
from av_contradiction.data.preprocessing import (
    VideoDecodeError,
    decode_clip,
    find_ffmpeg,
    probe,
    window_audio,
)
from av_contradiction.data.splits import (
    assign_groups_from_sources,
    check_group_leakage,
    group_stratified_split,
)
from av_contradiction.data.validation import audit_manifest


def _rows(n=20):
    return [
        {"sample_id": f"s{i}", "video_path": f"v{i}.mp4", "label": "consistent" if i % 2 else "contradictory", "group_id": f"g{i // 2}"}
        for i in range(n)
    ]


@pytest.mark.parametrize("ext", [".csv", ".jsonl"])
def test_manifest_roundtrip(tmp_path, ext):
    rows = _rows(4)
    rows[0]["is_synthetic"] = True
    write_manifest(rows, tmp_path / f"m{ext}")
    back = read_manifest(tmp_path / f"m{ext}")
    assert [r["sample_id"] for r in back] == [r["sample_id"] for r in rows]
    assert back[0]["is_synthetic"] is True and back[1]["is_synthetic"] is False


def test_group_split_no_leakage_and_stratified():
    rows = group_stratified_split(_rows(200), (0.7, 0.15, 0.15), seed=1)
    assert not check_group_leakage(rows)
    for s in ("train", "val", "test"):
        labs = [r["label"] for r in rows if r["split"] == s]
        assert labs and abs(labs.count("consistent") - labs.count("contradictory")) <= 4


def test_split_deterministic():
    a = [r["split"] for r in group_stratified_split(_rows(50), seed=7)]
    b = [r["split"] for r in group_stratified_split(_rows(50), seed=7)]
    assert a == b


def test_assign_groups_links_shared_sources():
    rows = [
        {"sample_id": "a", "video_source_id": "A", "audio_source_id": "A"},
        {"sample_id": "b", "video_source_id": "B", "audio_source_id": "B"},
        {"sample_id": "ab", "video_source_id": "A", "audio_source_id": "B"},  # swap links A and B
        {"sample_id": "c", "video_source_id": "C", "audio_source_id": "C"},
    ]
    g = {r["sample_id"]: r["group_id"] for r in assign_groups_from_sources(rows)}
    assert g["a"] == g["b"] == g["ab"] != g["c"]


def test_audit_detects_problems(tmp_path):
    (tmp_path / "ok.mp4").write_bytes(b"x" * 10)
    (tmp_path / "empty.mp4").write_bytes(b"")
    rows = [
        {"sample_id": "1", "video_path": "ok.mp4", "label": "consistent", "split": "train", "group_id": "g"},
        {"sample_id": "1", "video_path": "missing.mp4", "label": "consistent", "split": "train"},
        {"sample_id": "2", "video_path": "empty.mp4", "label": "weird", "split": "test"},
        {"sample_id": "3", "video_path": "ok.mp4", "label": "contradictory", "split": "test", "group_id": "g"},
    ]
    rep = audit_manifest(rows, tmp_path)
    assert not rep["ok"]
    assert rep["duplicate_sample_ids"] == ["1"]
    assert rep["invalid_label_ids"] == ["2"]
    assert rep["label_conflict_paths"] == ["ok.mp4"]
    assert "g" in rep["leaking_groups"]
    assert rep["missing_files"] == ["1"] and rep["empty_files"] == ["2"]


def test_probe_missing_and_corrupt(tmp_path):
    with pytest.raises(VideoDecodeError):
        probe(tmp_path / "nope.mp4")
    (tmp_path / "bad.mp4").write_bytes(b"\x00" * 1000)
    with pytest.raises(VideoDecodeError):
        probe(tmp_path / "bad.mp4")


def test_window_audio_edge_cases():
    w, ts, silent = window_audio(np.zeros(0, np.float32), 16000, 1.0, 0.5, 10, 1e-3)
    assert w.shape == (0, 16000) and ts.shape == (0, 2)
    w, ts, silent = window_audio(np.ones(4000, np.float32), 16000, 1.0, 0.5, 10, 1e-3)
    assert w.shape == (1, 16000) and ts[0, 1] == pytest.approx(0.25) and not silent[0]
    w, ts, _ = window_audio(np.ones(16000 * 10, np.float32), 16000, 1.0, 0.5, 5, 1e-3)
    assert len(w) == 5 and ts[0, 0] == 0 and ts[-1, 1] == pytest.approx(10.0)


def test_decode_video_without_audio(tmp_path):
    ff = find_ffmpeg()
    path = tmp_path / "silent.mp4"
    subprocess.run([ff, "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=red:s=32x32:d=1", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(path)], check=True, timeout=60)
    clip = decode_clip(path, PreprocessConfig(num_frames=3, frame_size=16, audio_sample_rate=16000))
    assert clip.has_video and not clip.has_audio
    assert clip.frames.shape == (3, 16, 16, 3)
    assert np.all(np.diff(clip.frame_ts) > 0)
    assert any("audio modality missing" in w for w in clip.warnings)


def test_duration_limits(tmp_path):
    ff = find_ffmpeg()
    path = tmp_path / "long.mp4"
    subprocess.run([ff, "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=red:s=32x32:d=3", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(path)], check=True, timeout=60)
    with pytest.raises(VideoDecodeError, match="too long"):
        decode_clip(path, PreprocessConfig(max_duration_sec=1.0, audio_sample_rate=16000))


def test_collate_padding_masks():
    items = [
        {"audio": torch.ones(3, 5), "audio_ts": torch.zeros(3, 2), "video": torch.ones(2, 4), "video_ts": torch.tensor([0.1, 0.2]), "label": 0, "sample_id": "a"},
        {"audio": torch.zeros(0, 5), "audio_ts": torch.zeros(0, 2), "video": torch.ones(4, 4), "video_ts": torch.arange(4.0), "label": 1, "sample_id": "b"},
    ]
    b = collate_features(items)
    assert b["audio"].shape == (2, 3, 5) and b["video"].shape == (2, 4, 4)
    assert b["audio_mask"].tolist() == [[True] * 3, [False] * 3]
    assert b["video_mask"].tolist() == [[True, True, False, False], [True] * 4]
    assert b["video"][0, 2:].abs().sum() == 0


def test_extraction_records_failures_and_keeps_going(mini_dataset, tmp_path):
    from av_contradiction.config import replace_section
    from av_contradiction.data.features import cache_path, extract_manifest

    cfg = replace_section(mini_dataset["cfg"], data={"feature_cache_dir": str(tmp_path / "cache")})
    rows = mini_dataset["rows"][:3] + [{"sample_id": "missing", "video_path": "nope.mp4", "label": "consistent"}]
    report = extract_manifest(cfg, rows, "cpu", decode_workers=2)
    assert report["extracted"] == 3 and list(report["failed"]) == ["missing"]
    assert all(cache_path(cfg, r["sample_id"]).exists() for r in rows[:3])
    assert extract_manifest(cfg, rows, "cpu")["skipped"] == 3  # cached clips are not re-extracted
