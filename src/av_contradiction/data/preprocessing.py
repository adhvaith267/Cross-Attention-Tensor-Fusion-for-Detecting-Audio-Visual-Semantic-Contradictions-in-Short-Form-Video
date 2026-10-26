"""Video/audio decoding with FFmpeg.

All FFmpeg calls use argument lists (never ``shell=True``) and a timeout, so user-supplied
file paths cannot inject shell commands. Frames are decoded one timestamp at a time via
input seeking, so a whole video is never held in memory.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from av_contradiction.config import PreprocessConfig


class VideoDecodeError(RuntimeError):
    """Raised when a file cannot be decoded or violates configured limits."""


@dataclass
class ProbeResult:
    duration_sec: float
    has_video: bool
    has_audio: bool


@dataclass
class DecodedClip:
    """Decoded, time-stamped modalities of one clip.

    frames:        uint8 [T_v, S, S, 3] (RGB, resized+center-cropped to ``frame_size``)
    frame_ts:      float32 [T_v] seconds, the exact seek time of each frame
    audio_windows: float32 [T_a, window_samples] mono waveform windows
    audio_ts:      float32 [T_a, 2] (start, end) seconds of each window
    audio_silent:  bool [T_a]
    """

    duration_sec: float
    frames: np.ndarray
    frame_ts: np.ndarray
    audio_windows: np.ndarray
    audio_ts: np.ndarray
    audio_silent: np.ndarray
    sample_rate: int
    warnings: list[str] = field(default_factory=list)

    @property
    def has_video(self) -> bool:
        return len(self.frame_ts) > 0

    @property
    def has_audio(self) -> bool:
        return len(self.audio_ts) > 0


def find_ffmpeg(explicit: str | None = None) -> str:
    """Locate an FFmpeg binary: explicit path, $AVC_FFMPEG, PATH, then imageio-ffmpeg's bundled binary."""
    for candidate in (explicit, os.environ.get("AVC_FFMPEG")):
        if candidate:
            if Path(candidate).is_file():
                return candidate
            raise FileNotFoundError(f"Configured FFmpeg binary not found: {candidate}")
    on_path = shutil.which("ffmpeg")
    if on_path:
        return on_path
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover - depends on environment
        raise FileNotFoundError("FFmpeg not found. Install it or `pip install imageio-ffmpeg`.") from exc


def _run(cmd: list[str], timeout: float) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise VideoDecodeError(f"FFmpeg timed out after {timeout:.0f}s") from exc


_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")


def probe(path: str | Path, ffmpeg: str | None = None, timeout: float = 30.0) -> ProbeResult:
    """Read container metadata by parsing ``ffmpeg -i`` output (imageio-ffmpeg ships no ffprobe)."""
    path = Path(path)
    if not path.is_file():
        raise VideoDecodeError(f"File does not exist: {path.name}")
    if path.stat().st_size == 0:
        raise VideoDecodeError(f"File is empty: {path.name}")
    ffmpeg = ffmpeg or find_ffmpeg()
    proc = _run([ffmpeg, "-hide_banner", "-nostdin", "-i", str(path)], timeout)
    text = proc.stderr.decode("utf-8", errors="replace")
    if "Invalid data found" in text or "moov atom not found" in text:
        raise VideoDecodeError(f"Corrupt or unsupported file: {path.name}")
    has_video = bool(re.search(r"Stream #\S+.*?: Video:", text)) and "attached pic" not in text
    has_audio = bool(re.search(r"Stream #\S+.*?: Audio:", text))
    if not (has_video or has_audio):
        raise VideoDecodeError(f"No audio or video streams found in {path.name}")
    dm = _DURATION_RE.search(text)
    if dm:
        h, m, s = dm.groups()
        duration = int(h) * 3600 + int(m) * 60 + float(s)
    else:
        duration = _decode_duration(path, ffmpeg, timeout)
    return ProbeResult(duration, has_video, has_audio)


def _decode_duration(path: Path, ffmpeg: str, timeout: float) -> float:
    """Fallback when the container reports no duration: fully decode and read the final timestamp."""
    proc = _run([ffmpeg, "-hide_banner", "-nostdin", "-i", str(path), "-f", "null", "-"], timeout)
    times = re.findall(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)", proc.stderr.decode("utf-8", errors="replace"))
    if not times:
        raise VideoDecodeError(f"Could not determine duration of {path.name}")
    h, m, s = times[-1]
    return int(h) * 3600 + int(m) * 60 + float(s)


def frame_timestamps(duration: float, num_frames: int) -> np.ndarray:
    """Uniform segment-centre timestamps: t_i = (i + 0.5) * duration / N."""
    return ((np.arange(num_frames) + 0.5) * duration / num_frames).astype(np.float32)


def extract_frames(
    path: str | Path,
    timestamps: np.ndarray,
    size: int,
    ffmpeg: str | None = None,
    timeout: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode one RGB frame per timestamp, resized (shorter side) and center-cropped to ``size``.

    Returns (frames uint8 [T,size,size,3], kept_timestamps). Timestamps that yield no frame
    (e.g. seeking past the last keyframe) are dropped rather than filled with fake data.
    """
    ffmpeg = ffmpeg or find_ffmpeg()
    frame_bytes = size * size * 3
    vf = f"scale={size}:{size}:force_original_aspect_ratio=increase,crop={size}:{size}"
    frames, kept = [], []
    for t in timestamps:
        cmd = [
            ffmpeg, "-v", "error", "-nostdin", "-ss", f"{float(t):.3f}", "-i", str(path),
            "-frames:v", "1", "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
        ]
        out = _run(cmd, timeout).stdout
        if len(out) >= frame_bytes:
            frames.append(np.frombuffer(out[:frame_bytes], dtype=np.uint8).reshape(size, size, 3))
            kept.append(t)
    if not frames:
        return np.zeros((0, size, size, 3), np.uint8), np.zeros((0,), np.float32)
    return np.stack(frames), np.asarray(kept, np.float32)


def extract_audio(path: str | Path, sample_rate: int, ffmpeg: str | None = None, timeout: float = 60.0) -> np.ndarray:
    """Decode the first audio stream to mono float32 at ``sample_rate``. Empty array if none."""
    ffmpeg = ffmpeg or find_ffmpeg()
    cmd = [
        ffmpeg, "-v", "error", "-nostdin", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sample_rate),
        "-f", "f32le", "pipe:1",
    ]
    proc = _run(cmd, timeout)
    if proc.returncode != 0 and not proc.stdout:
        return np.zeros((0,), np.float32)
    usable = len(proc.stdout) - len(proc.stdout) % 4
    wav = np.frombuffer(proc.stdout[:usable], dtype=np.float32).copy()
    return np.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)


def window_audio(
    wav: np.ndarray,
    sample_rate: int,
    window_sec: float,
    hop_sec: float,
    max_windows: int,
    silence_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a waveform into fixed-length windows with (start, end) times.

    Clips shorter than one window yield a single zero-padded window. When there are more than
    ``max_windows`` windows, an evenly spaced subset is kept (timestamps stay exact).
    """
    win = max(1, int(round(window_sec * sample_rate)))
    hop = max(1, int(round(hop_sec * sample_rate)))
    if wav.size == 0:
        return np.zeros((0, win), np.float32), np.zeros((0, 2), np.float32), np.zeros((0,), bool)
    if wav.size <= win:
        starts = np.array([0])
    else:
        starts = np.arange(0, wav.size - win + 1, hop)
        if starts[-1] + win < wav.size:  # cover the tail
            starts = np.append(starts, wav.size - win)
    if len(starts) > max_windows:
        starts = starts[np.linspace(0, len(starts) - 1, max_windows).round().astype(int)]
    windows = np.zeros((len(starts), win), np.float32)
    for i, s in enumerate(starts):
        chunk = wav[s : s + win]
        windows[i, : len(chunk)] = chunk
    ts = np.stack([starts / sample_rate, np.minimum(starts + win, wav.size) / sample_rate], axis=1).astype(np.float32)
    rms = np.sqrt(np.mean(windows**2, axis=1))
    return windows, ts, rms < silence_threshold


def decode_clip(path: str | Path, cfg: PreprocessConfig) -> DecodedClip:
    """Validate and decode a clip into time-stamped frames and audio windows."""
    ffmpeg = find_ffmpeg(cfg.ffmpeg_path)
    info = probe(path, ffmpeg, timeout=cfg.ffmpeg_timeout_sec)
    warnings: list[str] = []
    if info.duration_sec < cfg.min_duration_sec:
        raise VideoDecodeError(f"Clip too short ({info.duration_sec:.2f}s < {cfg.min_duration_sec}s)")
    if info.duration_sec > cfg.max_duration_sec:
        raise VideoDecodeError(f"Clip too long ({info.duration_sec:.1f}s > {cfg.max_duration_sec}s)")

    if info.has_video:
        frames, frame_ts = extract_frames(
            path, frame_timestamps(info.duration_sec, cfg.num_frames), cfg.frame_size, ffmpeg, cfg.ffmpeg_timeout_sec
        )
        if len(frame_ts) == 0:
            warnings.append("Video stream present but no frames could be decoded.")
        elif len(frame_ts) < cfg.num_frames:
            warnings.append(f"Only {len(frame_ts)}/{cfg.num_frames} frames could be decoded.")
    else:
        frames = np.zeros((0, cfg.frame_size, cfg.frame_size, 3), np.uint8)
        frame_ts = np.zeros((0,), np.float32)
        warnings.append("No video stream: visual modality missing.")

    if info.has_audio:
        wav = extract_audio(path, cfg.audio_sample_rate, ffmpeg, cfg.ffmpeg_timeout_sec)
        if wav.size == 0:
            warnings.append("Audio stream present but could not be decoded.")
    else:
        wav = np.zeros((0,), np.float32)
        warnings.append("No audio stream: audio modality missing.")
    windows, audio_ts, silent = window_audio(
        wav, cfg.audio_sample_rate, cfg.audio_window_sec, cfg.audio_hop_sec, cfg.max_audio_windows,
        cfg.silence_rms_threshold,
    )
    if len(silent) and silent.mean() > 0.9:
        warnings.append("Audio is (almost) entirely silent; audio evidence is weak.")
    return DecodedClip(info.duration_sec, frames, frame_ts, windows, audio_ts, silent, cfg.audio_sample_rate, warnings)
