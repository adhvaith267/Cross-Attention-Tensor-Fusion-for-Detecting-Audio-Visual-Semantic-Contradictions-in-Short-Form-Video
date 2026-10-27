"""Manifest auditing: schema, labels, files, duplicates, leakage, class balance."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from av_contradiction.data.manifest import REQUIRED_COLUMNS, Row, resolve_video_path
from av_contradiction.data.preprocessing import VideoDecodeError, extract_frames, find_ffmpeg, probe
from av_contradiction.data.splits import check_group_leakage

NEAR_DUP_MAX_FILES = 5000


def fast_file_hash(path: Path, chunk: int = 1 << 20) -> str:
    """Size + SHA-1 of the first and last MiB. Cheap exact-duplicate detector for large videos."""
    size = path.stat().st_size
    h = hashlib.sha1(str(size).encode())
    with open(path, "rb") as fh:
        h.update(fh.read(chunk))
        if size > 2 * chunk:
            fh.seek(-chunk, 2)
            h.update(fh.read(chunk))
    return h.hexdigest()


def average_hash(frame: np.ndarray) -> np.ndarray:
    """64-bit perceptual average hash of an 8x8 RGB frame."""
    gray = frame.astype(np.float32).mean(axis=2).ravel()
    return gray > gray.mean()


def audit_manifest(
    rows: list[Row],
    root_dir: str | Path = ".",
    label_names: tuple[str, ...] | list[str] = ("consistent", "contradictory"),
    check_files: bool = True,
    probe_files: bool = False,
    near_duplicates: bool = False,
    near_dup_hamming: int = 4,
    ffmpeg: str | None = None,
) -> dict[str, Any]:
    """Audit a manifest and return a JSON-serialisable report.

    ``report["invalid_sample_ids"]`` lists samples that should be excluded from training
    (missing/corrupt files, invalid labels, duplicate ids, conflicting labels).
    """
    report: dict[str, Any] = {"n_rows": len(rows), "errors": [], "warnings": []}
    invalid: set[str] = set()

    present = set().union(*(r.keys() for r in rows)) if rows else set()
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in present]
    if missing_cols:
        report["errors"].append(f"Missing required columns: {missing_cols}")
        report["invalid_sample_ids"] = []
        return report

    ids = Counter(r["sample_id"] for r in rows)
    dup_ids = sorted(i for i, n in ids.items() if n > 1 or not i)
    if dup_ids:
        report["errors"].append(f"{len(dup_ids)} duplicate or empty sample_ids")
        invalid.update(dup_ids)
    report["duplicate_sample_ids"] = dup_ids

    bad_labels = [r["sample_id"] for r in rows if r["label"] not in label_names]
    if bad_labels:
        report["errors"].append(f"{len(bad_labels)} rows have labels outside {list(label_names)}")
        invalid.update(bad_labels)
    report["invalid_label_ids"] = bad_labels

    # Same file labelled differently -> annotation inconsistency. Rows whose audio comes from another
    # row's features (audio swaps) legitimately share their video file with a consistent row.
    by_path: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        if not r.get("audio_feature_id"):
            by_path[r["video_path"]].add(r["label"])
    conflicts = sorted(p for p, labs in by_path.items() if len(labs) > 1)
    if conflicts:
        report["errors"].append(f"{len(conflicts)} video files carry conflicting labels")
        invalid.update(r["sample_id"] for r in rows if r["video_path"] in conflicts)
    report["label_conflict_paths"] = conflicts[:100]

    splits = Counter(r.get("split", "") for r in rows)
    if "" in splits:
        report["warnings"].append(f"{splits['']} rows have no split assigned")
    report["class_distribution"] = dict(Counter(r["label"] for r in rows))
    report["class_distribution_by_split"] = {
        s: dict(Counter(r["label"] for r in rows if r.get("split", "") == s)) for s in sorted(splits)
    }
    report["synthetic_by_split"] = {
        s: dict(Counter(r["label"] for r in rows if r.get("split", "") == s and r.get("is_synthetic")))
        for s in sorted(splits)
    }
    for s, dist in report["class_distribution_by_split"].items():
        if dist and min(dist.get(lab, 0) for lab in label_names) == 0:
            report["warnings"].append(f"Split '{s or '<none>'}' is missing at least one class: {dist}")
        elif dist:
            ratio = max(dist.values()) / max(1, min(dist.values()))
            if ratio > 1.5:
                report["warnings"].append(f"Split '{s}' imbalance ratio {ratio:.2f}; prefer macro-F1.")

    leakage = check_group_leakage(rows)
    for col in ("audio_source_id", "video_source_id", "source_id"):
        tmp = [{"group_id": f"{col}:{r[col]}", "split": r.get("split", "")} for r in rows if r.get(col)]
        leakage.update(check_group_leakage(tmp))
    if leakage:
        report["errors"].append(f"{len(leakage)} groups/sources appear in more than one split (data leakage)")
    report["leaking_groups"] = dict(list(leakage.items())[:100])
    report["n_leaking_groups"] = len(leakage)

    no_license = sum(1 for r in rows if not r.get("license"))
    if no_license:
        report["warnings"].append(f"{no_license} rows have no license recorded")

    missing, empty, corrupt, no_audio, no_video = [], [], [], [], []
    hashes: dict[str, list[str]] = defaultdict(list)
    ahashes: dict[str, np.ndarray] = {}
    if check_files:
        ff = find_ffmpeg(ffmpeg) if (probe_files or near_duplicates) else None
        for r in rows:
            p = resolve_video_path(r, root_dir)
            sid = r["sample_id"]
            if not p.is_file():
                missing.append(sid)
                continue
            if p.stat().st_size == 0:
                empty.append(sid)
                continue
            hashes[fast_file_hash(p)].append(sid)
            if probe_files:
                try:
                    info = probe(p, ff)
                    if not info.has_audio:
                        no_audio.append(sid)
                    if not info.has_video:
                        no_video.append(sid)
                    if near_duplicates and info.has_video and len(ahashes) < NEAR_DUP_MAX_FILES:
                        frames, _ = extract_frames(p, np.array([info.duration_sec / 2]), 8, ff)
                        if len(frames):
                            ahashes[sid] = average_hash(frames[0])
                except VideoDecodeError:
                    corrupt.append(sid)
        invalid.update(missing, empty, corrupt)
        for name, lst in (("missing", missing), ("empty", empty), ("corrupt", corrupt)):
            if lst:
                report["errors"].append(f"{len(lst)} {name} video files")
    report.update(
        missing_files=missing, empty_files=empty, corrupt_files=corrupt,
        no_audio_stream=no_audio, no_video_stream=no_video,
    )

    # Identical file bytes reused by several samples are expected for swapped negatives, so flag
    # them only when the copies land in different splits.
    split_of = {r["sample_id"]: r.get("split", "") for r in rows}
    exact_dups = [ids_ for ids_ in hashes.values() if len(ids_) > 1]
    cross_split_dups = [d for d in exact_dups if len({split_of[i] for i in d}) > 1]
    report["exact_duplicate_groups"] = len(exact_dups)
    report["cross_split_exact_duplicates"] = cross_split_dups[:100]
    if cross_split_dups:
        report["errors"].append(f"{len(cross_split_dups)} byte-identical files span multiple splits")

    if near_duplicates:
        report["near_duplicate_pairs_cross_split"] = _near_duplicates(ahashes, split_of, near_dup_hamming)
        if report["near_duplicate_pairs_cross_split"]:
            report["warnings"].append(
                f"{len(report['near_duplicate_pairs_cross_split'])} perceptually similar pairs span splits (review manually)"
            )
    report["invalid_sample_ids"] = sorted(invalid)
    report["ok"] = not report["errors"]
    return report


def _near_duplicates(ahashes: dict[str, np.ndarray], split_of: dict[str, str], max_dist: int) -> list[list[str]]:
    if len(ahashes) < 2:
        return []
    ids = list(ahashes)
    bits = np.stack([ahashes[i] for i in ids]).astype(np.uint8)
    dist = (bits[:, None, :] != bits[None, :, :]).sum(-1)
    pairs = []
    for a, b in zip(*np.where(np.triu(dist <= max_dist, k=1)), strict=True):
        if split_of[ids[a]] != split_of[ids[b]]:
            pairs.append([ids[a], ids[b]])
    return pairs[:200]
