"""Dataset adapters on mock inputs in the published formats (no real dataset needed)."""

import json
import shutil
from pathlib import Path

from av_contradiction.data.adapters import avhbench_av_matching, vggsound_conflict_swaps
from av_contradiction.data.preprocessing import probe
from av_contradiction.data.splits import (
    assign_groups_from_sources,
    check_group_leakage,
    group_stratified_split,
)


def test_avhbench_av_matching(tmp_path):
    qa = [
        {"video_id": "abc", "task": "AV Matching", "text": "Are the audio and video matched?", "label": "Yes"},
        {"video_id": "abc_swap1", "task": "AV Matching", "text": "Are the audio and video matched?", "label": "No"},
        {"video_id": "xyz", "task": "Audio-driven Video Hallucination", "text": "Is there a dog?", "label": "Yes"},
        {"video_id": "q", "task": "AV Captioning", "text": "Describe", "label": "A man talks"},
    ]
    p = tmp_path / "qa.json"
    p.write_text(json.dumps(qa))
    rows = avhbench_av_matching(p, tmp_path / "videos", group_regex=r"^(.*?)(?:_swap\d+)?$")
    assert [r["label"] for r in rows] == ["consistent", "contradictory"]
    assert rows[0]["group_id"] == rows[1]["group_id"] == "abc"
    assert rows[0]["video_path"].endswith("videos/abc.mp4")


def test_vggsound_conflict_swaps(tmp_path, mini_dataset):
    src = Path(mini_dataset["cfg"].data.root_dir) / "videos"
    clips = tmp_path / "clips"
    clips.mkdir()
    vids = sorted(src.glob("*.mp4"))[:4]
    meta = [("aaa", 0, "playing piano"), ("bbb", 10, "playing violin, fiddle"), ("ccc", 20, "dog barking"), ("ddd", 30, "cat meowing")]
    for v, (ytid, start, _) in zip(vids, meta, strict=True):
        shutil.copy(v, clips / f"{ytid}_{start:06d}.mp4")
    csv = tmp_path / "vggsound.csv"
    csv.write_text("\n".join(f'{y},{s},"{lab}",train' for y, s, lab in meta) + "\n")
    groups = tmp_path / "groups.json"
    groups.write_text(json.dumps({"instruments": ["playing piano", "playing violin, fiddle"], "pets": ["dog barking", "cat meowing", "not a class"]}))

    rows = vggsound_conflict_swaps(csv, clips, groups, tmp_path / "work", seed=0)
    orig = [r for r in rows if r["label"] == "consistent"]
    swaps = [r for r in rows if r["label"] == "contradictory"]
    assert len(orig) == 4 and len(swaps) == 4
    for s in swaps:
        assert s["is_synthetic"] and s["audio_source_id"] != s["video_source_id"]
        info = probe(s["video_path"])
        assert info.has_audio and info.has_video
    # piano video must only receive violin audio (same group), never pet audio
    piano_swap = next(s for s in swaps if s["video_source_id"] == "aaa_000000")
    assert piano_swap["audio_source_id"] == "bbb_000010"

    rows = group_stratified_split(assign_groups_from_sources(rows), (0.5, 0.25, 0.25))
    assert not check_group_leakage(rows)
