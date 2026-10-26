"""Fully synthetic audio-visual sanity benchmark ("ColorTone").

THIS IS NOT A SEMANTIC-CONTRADICTION DATASET. It is a controlled mechanism check with a known
ground-truth rule, used to verify that the pipeline and the cross-modal components work:

* A fixed bijection maps K colours to K tone frequencies (the "semantics").
* Each clip has two segments; each segment shows one colour and plays one tone (+ noise).
* consistent:            both segments' tones match their colours.
* contradictory/swap:    the two tones are exchanged in time. The clip-level *bag* of colours and
                         tones equals that of a consistent clip, so it is only detectable by
                         temporally aligned cross-modal reasoning.
* contradictory/replace: one segment plays the tone of a colour absent from the clip.

Each scene (pair of colours) yields one consistent and one contradictory clip sharing the same
video track, so they share a ``group_id`` and always land in the same split.
Each modality alone is uninformative of the label by construction (chance-level for unimodal models).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from av_contradiction.data.manifest import Row
from av_contradiction.data.preprocessing import find_ffmpeg

COLORS = {"red": (220, 40, 40), "green": (40, 190, 60), "blue": (40, 70, 220), "yellow": (230, 210, 40)}
TONES_HZ = {"red": 330.0, "green": 550.0, "blue": 880.0, "yellow": 1400.0}


def _hex(rgb: tuple[int, int, int]) -> str:
    return "0x" + "".join(f"{int(np.clip(c, 0, 255)):02X}" for c in rgb)


def render_clip(
    path: Path,
    colors: tuple[tuple[int, int, int], tuple[int, int, int]],
    tones: tuple[float, float],
    seg_sec: float = 1.5,
    size: int = 96,
    noise_amp: float = 0.05,
    tone_volume: float = 0.5,
    ffmpeg: str | None = None,
    sample_rate: int = 16000,
) -> None:
    ffmpeg = ffmpeg or find_ffmpeg()
    total = 2 * seg_sec
    cmd = [ffmpeg, "-y", "-v", "error", "-nostdin"]
    for c in colors:
        cmd += ["-f", "lavfi", "-i", f"color=c={_hex(c)}:s={size}x{size}:r=12:d={seg_sec}"]
    for f in tones:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency={f:.1f}:sample_rate={sample_rate}:duration={seg_sec}"]
    cmd += ["-f", "lavfi", "-i", f"anoisesrc=color=white:amplitude={noise_amp:.4f}:sample_rate={sample_rate}:duration={total}"]
    graph = (
        "[0:v][1:v]concat=n=2:v=1:a=0,noise=alls=10:allf=t[v];"
        f"[2:a][3:a]concat=n=2:v=0:a=1,volume={tone_volume:.3f}[s];"
        "[s][4:a]amix=inputs=2:normalize=0:duration=first[a]"
    )
    cmd += [
        "-filter_complex", graph, "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k", str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=60, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg failed rendering {path.name}: {proc.stderr.decode(errors='replace')[-400:]}")


def generate_colortone(out_dir: str | Path, n_scenes: int = 100, seed: int = 0, seg_sec: float = 1.5, ffmpeg: str | None = None) -> list[Row]:
    """Render ``2 * n_scenes`` clips into ``out_dir/videos`` and return manifest rows (no splits yet)."""
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    (out_dir / "videos").mkdir(parents=True, exist_ok=True)
    names = list(COLORS)
    rows: list[Row] = []
    for s in range(n_scenes):
        c1, c2 = rng.choice(names, size=2, replace=False)
        jitter = rng.integers(-25, 26, size=(2, 3))
        vis = (tuple(np.add(COLORS[c1], jitter[0])), tuple(np.add(COLORS[c2], jitter[1])))
        detune = rng.uniform(0.97, 1.03, size=2)
        good = (TONES_HZ[c1] * detune[0], TONES_HZ[c2] * detune[1])
        if rng.random() < 0.5:
            method, bad = "swap", (good[1], good[0])
        else:
            absent = [n for n in names if n not in (c1, c2)]
            k = int(rng.integers(0, 2))
            wrong = TONES_HZ[str(rng.choice(absent))] * rng.uniform(0.97, 1.03)
            bad = (wrong, good[1]) if k == 0 else (good[0], wrong)
            method = f"replace_seg{k}"
        noise = float(rng.uniform(0.01, 0.08))
        vol = float(rng.uniform(0.3, 0.7))
        for label, tones, gen in (("consistent", good, "original"), ("contradictory", bad, method)):
            sid = f"ct{s:04d}_{label[:4]}"
            rel = Path("videos") / f"{sid}.mp4"
            render_clip(out_dir / rel, vis, tones, seg_sec, noise_amp=noise, tone_volume=vol, ffmpeg=ffmpeg)
            rows.append({
                "sample_id": sid, "video_path": rel.as_posix(), "label": label, "group_id": f"scene{s:04d}",
                "source_dataset": "synthetic_colortone", "source_id": f"scene{s:04d}", "license": "generated (CC0)",
                "is_synthetic": True, "generation_method": f"colortone:{gen}",
                "video_source_id": f"scene{s:04d}", "audio_source_id": f"scene{s:04d}_{label[:4]}",
                "notes": f"colors={c1},{c2}",
            })
    return rows
