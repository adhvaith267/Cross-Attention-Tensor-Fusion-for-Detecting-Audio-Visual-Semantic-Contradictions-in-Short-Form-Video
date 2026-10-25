"""Optional Whisper transcription (evidence only; never used as the audio representation)."""

from __future__ import annotations

from pathlib import Path

from av_contradiction.data.preprocessing import extract_audio, find_ffmpeg


class WhisperTranscriber:
    def __init__(self, model_name: str, local_files_only: bool = True, device: str = "cpu") -> None:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError("Transcription needs `pip install -e .[encoders]`") from exc
        try:
            self.pipe = pipeline(
                "automatic-speech-recognition", model=model_name, device=device,
                model_kwargs={"local_files_only": local_files_only},
            )
        except OSError as exc:
            raise OSError(f"Whisper weights '{model_name}' not available locally") from exc

    def transcribe(self, path: str | Path, ffmpeg: str | None = None) -> list[dict]:
        wav = extract_audio(path, 16000, find_ffmpeg(ffmpeg))
        if wav.size == 0:
            return []
        out = self.pipe({"raw": wav, "sampling_rate": 16000}, return_timestamps=True)
        chunks = out.get("chunks") or [{"timestamp": (None, None), "text": out.get("text", "")}]
        return [
            {"start_sec": c["timestamp"][0], "end_sec": c["timestamp"][1], "text": c["text"].strip()}
            for c in chunks if c.get("text", "").strip()
        ]
