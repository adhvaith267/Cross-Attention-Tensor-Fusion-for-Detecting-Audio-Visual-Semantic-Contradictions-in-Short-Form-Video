"""Dataset manifest I/O.

A manifest is a CSV or JSONL file with one row per sample. Columns:

required: sample_id, video_path, label
optional: split, group_id, source_dataset, source_id, source_url, license, is_synthetic,
          generation_method, audio_source_id, video_source_id, annotator, notes,
          video_feature_id, audio_feature_id (build the row from other rows' cached features)

``group_id`` identifies everything that must stay in the same split (original video, speaker,
or source clip). Synthetic negatives built from clip X and clip Y must carry a group that
covers both sources; ``prepare_dataset.py`` handles that via union-find over source ids.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

REQUIRED_COLUMNS = ("sample_id", "video_path", "label")
OPTIONAL_COLUMNS = (
    "split", "group_id", "source_dataset", "source_id", "source_url", "license", "is_synthetic",
    "generation_method", "audio_source_id", "video_source_id", "annotator", "notes",
    "video_feature_id", "audio_feature_id",
)
ALL_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

Row = dict[str, Any]


def _normalise(row: Row) -> Row:
    out = {k: ("" if v is None else v) for k, v in row.items()}
    syn = str(out.get("is_synthetic", "")).strip().lower()
    out["is_synthetic"] = syn in ("1", "true", "yes")
    for key in ("sample_id", "video_path", "label", "split", "group_id"):
        if key in out:
            out[key] = str(out[key]).strip()
    return out


def read_manifest(path: str | Path) -> list[Row]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    if path.suffix.lower() == ".jsonl":
        with open(path, encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
    elif path.suffix.lower() == ".csv":
        with open(path, encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
    else:
        raise ValueError(f"Manifest must be .csv or .jsonl, got {path.suffix}")
    return [_normalise(r) for r in rows]


def write_manifest(rows: list[Row], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = sorted({k for r in rows for k in r} - set(ALL_COLUMNS))
    columns = [c for c in ALL_COLUMNS if any(c in r for r in rows) or c in REQUIRED_COLUMNS] + extra
    if path.suffix.lower() == ".jsonl":
        with open(path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps({c: r.get(c, "") for c in columns}) + "\n")
    else:
        with open(path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns)
            writer.writeheader()
            for r in rows:
                writer.writerow({c: r.get(c, "") for c in columns})


def manifest_hash(path: str | Path) -> str:
    """SHA-256 of the manifest bytes; stored in checkpoints to tie models to an exact dataset version."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_video_path(row: Row, root_dir: str | Path) -> Path:
    p = Path(row["video_path"])
    return p if p.is_absolute() else Path(root_dir) / p


def filter_split(rows: list[Row], split: str) -> list[Row]:
    return [r for r in rows if r.get("split") == split]
