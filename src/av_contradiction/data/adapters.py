"""Converters from public datasets into the project manifest format.

* AVHBench "AV Matching" (Sung-Bin et al., ICLR 2025): yes/no audio-visual correspondence
  questions over real and audio-swapped clips. Yes -> consistent, No -> contradictory.
  This is a *correspondence* label, the closest public proxy found for semantic contradiction.
  No license is stated in the repository at the time of writing: verify terms before use.

* VGGSound controlled audio swaps: given locally downloaded VGGSound clips (CC BY 4.0 annotations;
  video copyright stays with uploaders) and a hand-curated *conflict group* file, build
  contradictory samples by muxing the audio of class B onto the video of class A where A and B are
  in the same conflict group (e.g. two instruments: the sound contradicts the visible instrument).
  Cross-group pairs are deliberately NOT generated: they are "unrelated", not contradictory.
  All such negatives are flagged ``is_synthetic=True`` and should be spot-checked by a human.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

from av_contradiction.data.manifest import Row
from av_contradiction.data.preprocessing import find_ffmpeg
from av_contradiction.utils.logging import get_logger

log = get_logger(__name__)

AVH_LICENSE_NOTE = "UNVERIFIED: AVHBench repo states no license; videos derive from VALOR/AudioCaps (YouTube). Check terms."
VGGSOUND_LICENSE_NOTE = "VGGSound annotations CC BY 4.0; video/audio copyright remains with original YouTube uploaders."


def avhbench_av_matching(qa_json: str | Path, video_dir: str | Path, group_regex: str | None = None) -> list[Row]:
    """Rows for the AV-matching task. ``group_regex`` (first capture group) maps a video_id to its
    source group, e.g. r"^(.*?)(?:_swap.*)?$" if swapped clips share the original id as a prefix.
    Inspect the downloaded ids before choosing it; without it each video is its own group."""
    with open(qa_json, encoding="utf-8") as fh:
        items = json.load(fh)
    pat = re.compile(group_regex) if group_regex else None
    rows: list[Row] = []
    for i, it in enumerate(items):
        if "match" not in str(it.get("task", "")).lower():
            continue
        ans = str(it.get("label", "")).strip().lower()
        if ans not in ("yes", "no"):
            continue
        vid = str(it["video_id"])
        group = vid
        if pat and (m := pat.match(vid)):
            group = m.group(1)
        rows.append({
            "sample_id": f"avh_{i:05d}_{vid}", "video_path": (Path(video_dir) / f"{vid}.mp4").as_posix(),
            "label": "consistent" if ans == "yes" else "contradictory",
            "group_id": group, "source_dataset": "AVHBench", "source_id": vid,
            "license": AVH_LICENSE_NOTE, "is_synthetic": "", "generation_method": "AVHBench (see paper)",
            "video_source_id": group, "notes": str(it.get("text", ""))[:200],
        })
    log.info("AVHBench: %d AV-matching rows", len(rows))
    return rows


def read_vggsound_csv(path: str | Path) -> list[dict[str, str]]:
    """VGGSound's vggsound.csv has no header: youtube_id, start_seconds, label, split."""
    out = []
    with open(path, encoding="utf-8", newline="") as fh:
        for rec in csv.reader(fh):
            if len(rec) >= 4:
                out.append({"ytid": rec[0], "start": int(float(rec[1])), "label": rec[2].strip(), "split": rec[3].strip()})
    return out


def mux_audio_swap(video_src: Path, audio_src: Path, dest: Path, ffmpeg: str | None = None, timeout: float = 120) -> None:
    """Video stream from ``video_src`` + audio stream from ``audio_src`` (trimmed to the shorter)."""
    ffmpeg = ffmpeg or find_ffmpeg()
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y", "-v", "error", "-nostdin", "-i", str(video_src), "-i", str(audio_src),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-shortest", str(dest),
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode(errors="replace")[-300:])


def vggsound_conflict_swaps(
    vggsound_csv: str | Path,
    clip_dir: str | Path,
    conflict_groups: str | Path,
    out_dir: str | Path,
    filename_pattern: str = "{ytid}_{start:06d}.mp4",
    max_per_class: int | None = None,
    seed: int = 0,
    ffmpeg: str | None = None,
) -> list[Row]:
    """One original (consistent) + one within-group audio swap (contradictory) per available clip.

    ``video_path`` values are absolute: originals point into ``clip_dir``, swaps into ``out_dir/swaps``.
    """
    rng = np.random.default_rng(seed)
    with open(conflict_groups, encoding="utf-8") as fh:
        groups: dict[str, list[str]] = {k: v for k, v in json.load(fh).items() if not k.startswith("_")}
    meta = read_vggsound_csv(vggsound_csv)
    known = {m["label"] for m in meta}
    class_to_group = {}
    for g, classes in groups.items():
        for c in classes:
            if c not in known:
                log.warning("Conflict-group class '%s' not found in %s (ignored)", c, vggsound_csv)
            else:
                class_to_group[c] = g
    clip_dir, out_dir = Path(clip_dir), Path(out_dir)
    by_class: dict[str, list[dict]] = defaultdict(list)
    for m in meta:
        p = clip_dir / filename_pattern.format(**m)
        if m["label"] in class_to_group and p.is_file():
            m["path"] = p
            by_class[m["label"]].append(m)
    if max_per_class:
        for c in by_class:
            rng.shuffle(by_class[c])
            by_class[c] = by_class[c][:max_per_class]
    rows: list[Row] = []
    for cls, items in sorted(by_class.items()):
        partners = [c for c in groups[class_to_group[cls]] if c != cls and by_class.get(c)]
        for m in items:
            sid = f"{m['ytid']}_{m['start']:06d}"
            rows.append({
                "sample_id": f"vgg_{sid}_orig", "video_path": m["path"].resolve().as_posix(), "label": "consistent",
                "source_dataset": "VGGSound", "source_id": sid, "license": VGGSOUND_LICENSE_NOTE,
                "is_synthetic": False, "generation_method": "original", "video_source_id": sid, "audio_source_id": sid,
                "notes": f"class={cls}",
            })
            if not partners:
                continue
            pcls = str(rng.choice(partners))
            donor = by_class[pcls][int(rng.integers(len(by_class[pcls])))]
            dsid = f"{donor['ytid']}_{donor['start']:06d}"
            dest = out_dir / "swaps" / f"{sid}__audio_{dsid}.mp4"
            try:
                if not dest.exists():
                    mux_audio_swap(m["path"], donor["path"], dest, ffmpeg)
            except Exception as exc:
                log.warning("Swap %s failed: %s", dest.name, exc)
                continue
            rows.append({
                "sample_id": f"vgg_{sid}_swap_{dsid}", "video_path": dest.resolve().as_posix(), "label": "contradictory",
                "source_dataset": "VGGSound", "source_id": sid, "license": VGGSOUND_LICENSE_NOTE, "is_synthetic": True,
                "generation_method": f"within-group audio swap ({class_to_group[cls]}): video={cls}, audio={pcls}",
                "video_source_id": sid, "audio_source_id": dsid, "notes": "needs human spot-check",
            })
    log.info("VGGSound swaps: %d rows", len(rows))
    return rows
