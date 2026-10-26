"""AVID-Bench (Chen et al., 2026) segment subset: selection, resumable download and verification.

Source: ModelScope dataset ``czx13927247324/AVID`` (the download host documented in
https://github.com/czx1220/AVID-bench; the Hugging Face card ``Chen-15/AVID`` hosts annotations only).
Media are individual pre-cut segment MP4s, so only selected files are ever downloaded.

Label semantics (verified in ``annotations/*_segments.jsonl``; folder names are NOT our labels):

* ``*_segments/negative/<vid>_<k>.mp4``: ``ground_truth.exists == "Yes"``, audio was *injected* with an
  inconsistency of ``ground_truth.injection_type``  -> label ``contradictory``
* ``*_segments/positive/<vid>_<k>.mp4``: ``ground_truth.exists == "No"``, original audio -> ``consistent``

A positive and a negative file with the same name show the same video clip, so the subset is built from
such *pairs*: the label then depends only on how audio relates to video, never on the visuals alone.
Every pair of one source video ``<vid>`` stays in one split. AVID has no validation split; ours is
carved from official-train source videos and never from the official test split.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from av_contradiction.data.manifest import Row
from av_contradiction.utils.logging import get_logger

log = get_logger(__name__)

DATASET_ID = "czx13927247324/AVID"
SEGMENT_DIRS = (
    "train_7239_segments/positive", "train_7239_segments/negative",
    "test_1171_segments/positive", "test_1171_segments/negative",
)
ANNOTATIONS = {"train": "annotations/train_segments.jsonl", "test": "annotations/test_segments.jsonl"}
# Injection types whose audio contradicts what is shown. Excluded by default: VOLUME_FLUCTUATION
# (a loudness glitch), TEMPORAL_SHIFT and LIP_SYNC (timing offsets, not semantic content; also below
# the resolution of 2 s audio windows and ~1.6 fps frame sampling).
SEMANTIC_TYPES = ("SEMANTIC_DIVERGENCE", "EMOTION_MISMATCH", "BACKGROUND_CONFLICT", "BACKGROUND_SOUND", "VOICE_IDENTITY")
LICENSE_NOTE = "AVID-Bench, Apache-2.0 (dataset card); clips derive from YouTube videos whose copyright stays with uploaders."
_PAGE = 1000


def list_remote(cache_dir: str | Path, dirs: tuple[str, ...] = SEGMENT_DIRS) -> dict[str, dict[str, Any]]:
    """``{path: {"size", "sha256"}}`` for every file in ``dirs``, via the ModelScope listing API.
    Pages are cached in ``cache_dir`` so an interrupted listing resumes where it stopped."""
    from modelscope.hub.api import HubApi

    api, cache_dir = HubApi(), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, Any]] = {}
    for d in dirs:
        page = 1
        while True:
            cached = cache_dir / f"{d.replace('/', '__')}__{page:03d}.json"
            if cached.is_file():
                entries = json.loads(cached.read_text(encoding="utf-8"))
            else:
                entries = _with_retries(lambda d=d, page=page: api.get_dataset_files(
                    DATASET_ID, root_path=d, recursive=False, page_number=page, page_size=_PAGE))
                entries = [{"path": e["Path"], "size": int(e["Size"]), "sha256": e["Sha256"]} for e in entries if e["Type"] == "blob"]
                cached.write_text(json.dumps(entries), encoding="utf-8")
                log.info("listed %s page %d (%d files)", d, page, len(entries))
            files.update({e["path"]: {"size": e["size"], "sha256": e["sha256"]} for e in entries})
            if len(entries) < _PAGE:
                break
            page += 1
    return files


def _with_retries(fn, tries: int = 4, base_delay: float = 5.0):
    for attempt in range(tries):
        try:
            return fn()
        except Exception as exc:  # network errors surface as several exception types
            if attempt == tries - 1:
                raise
            log.warning("attempt %d failed (%s); retrying", attempt + 1, exc)
            time.sleep(base_delay * 2**attempt)


def source_of(path: str) -> str:
    """``.../<youtube id>_<k>.mp4`` -> ``<youtube id>`` (ids may contain ``_`` and ``-``)."""
    return re.sub(r"_\d+\.mp4$", "", path.rsplit("/", 1)[-1])


def load_pairs(annotation_file: str | Path, official_split: str) -> tuple[list[dict[str, Any]], Counter]:
    """Pair each consistent segment with the injected version of the same clip.
    Returns (pairs, counts of unpaired records)."""
    by_seg: dict[str, dict[str, dict]] = defaultdict(dict)
    with open(annotation_file, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            gt = rec["ground_truth"]
            kind = {"Yes": "neg", "No": "pos"}.get(str(gt.get("exists")))
            folder = rec["video_path"].split("/")[1]
            if kind is None or (kind == "neg") != (folder == "negative"):
                raise ValueError(f"Unexpected label/folder combination in {rec['qa_id']}: {gt.get('exists')} / {folder}")
            by_seg[rec["video_path"].rsplit("/", 1)[-1]][kind] = rec
    pairs, unpaired = [], Counter()
    for seg, recs in by_seg.items():
        if set(recs) != {"pos", "neg"}:
            unpaired[f"only_{next(iter(recs))}"] += 1
            continue
        gt = recs["neg"]["ground_truth"]
        pairs.append({
            "segment": seg, "source": source_of(seg), "official_split": official_split,
            "pos": recs["pos"], "neg": recs["neg"], "injection_type": gt.get("injection_type", ""),
            "videoclass": gt.get("videoclass", ""), "annotation_file": str(Path(annotation_file).as_posix()),
        })
    return pairs, unpaired


def _pick(pairs: list[dict], n: int, rng: random.Random, exclude_sources: set[str]) -> list[dict]:
    """``n`` pairs, at most one per source video, round-robin over injection types (most diverse first)."""
    by_type: dict[str, list[dict]] = defaultdict(list)
    seen: set[str] = set(exclude_sources)
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    for p in shuffled:
        by_type[p["injection_type"]].append(p)
    chosen: list[dict] = []
    types = sorted(by_type)
    while len(chosen) < n and any(by_type.values()):
        for t in types:
            while by_type[t]:
                p = by_type[t].pop()
                if p["source"] not in seen:
                    seen.add(p["source"])
                    chosen.append(p)
                    break
            if len(chosen) >= n:
                break
    return chosen


def pair_bytes(p: dict, remote: dict[str, dict]) -> int:
    return remote[p["pos"]["video_path"]]["size"] + remote[p["neg"]["video_path"]]["size"]


def select_subset(
    root: str | Path,
    remote: dict[str, dict],
    n_pairs: dict[str, int],
    budget_gb: float,
    categories: tuple[str, ...] = SEMANTIC_TYPES,
    max_pair_mb: float | None = None,
    seed: int = 0,
) -> tuple[list[Row], dict[str, Any]]:
    """Choose pairs for train/val (official train) and test (official test) within ``budget_gb``.

    Returns manifest rows in the project format plus a report of everything that was excluded and why.
    """
    rng = random.Random(seed)
    root = Path(root)
    report: dict[str, Any] = {"excluded": {}, "categories": list(categories), "seed": seed}
    pools: dict[str, list[dict]] = {}
    for off, ann in ANNOTATIONS.items():
        pairs, unpaired = load_pairs(root / ann, off)
        excl: Counter = Counter({f"unpaired_{k}": v for k, v in unpaired.items()})
        kept = []
        for p in pairs:
            if p["injection_type"] not in categories:
                excl[f"category_{p['injection_type']}"] += 1
            elif p["pos"]["video_path"] not in remote or p["neg"]["video_path"] not in remote:
                excl["file_not_on_host"] += 1
            elif max_pair_mb and pair_bytes(p, remote) > max_pair_mb * 1e6:
                excl[f"pair_larger_than_{max_pair_mb:g}MB"] += 1
            else:
                kept.append(p)
        report["excluded"][f"official_{off}"] = dict(excl)
        pools[off] = kept

    test = _pick(pools["test"], n_pairs["test"], rng, set())
    val = _pick(pools["train"], n_pairs["val"], rng, set())
    train = _pick(pools["train"], n_pairs["train"], rng, {p["source"] for p in val})
    chosen = {"train": train, "val": val, "test": test}

    # Enforce the byte budget by dropping pairs proportionally across splits (never by duplicating).
    budget = budget_gb * 1e9
    total = sum(pair_bytes(p, remote) for ps in chosen.values() for p in ps)
    while total > budget:
        split = max(chosen, key=lambda s: len(chosen[s]) / max(1, n_pairs[s]))
        total -= pair_bytes(chosen[split].pop(), remote)
    report["selected_pairs"] = {s: len(ps) for s, ps in chosen.items()}
    report["selected_bytes"] = int(total)

    rows: list[Row] = []
    for split, ps in chosen.items():
        for p in ps:
            for kind, label in (("pos", "consistent"), ("neg", "contradictory")):
                rec, path = p[kind], p[kind]["video_path"]
                gt = rec["ground_truth"]
                rows.append({
                    "sample_id": rec["qa_id"], "video_path": path, "label": label, "split": split, "group_id": p["source"],
                    "source_dataset": "AVID-Bench", "source_id": p["segment"], "license": LICENSE_NOTE,
                    "is_synthetic": kind == "neg",
                    "generation_method": f"AVID audio injection: {p['injection_type']}" if kind == "neg" else "original",
                    "video_source_id": p["segment"], "audio_source_id": p["segment"] if kind == "pos" else f"injected:{p['segment']}",
                    "notes": str(gt.get("inconsistency_point", ""))[:300],
                    "official_split": p["official_split"], "official_exists": gt.get("exists"),
                    "injection_type": p["injection_type"] if kind == "neg" else "", "videoclass": p["videoclass"],
                    "annotation_file": p["annotation_file"], "size_bytes": remote[path]["size"], "sha256": remote[path]["sha256"],
                })
    return rows, report


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def disk_usage(root: Path) -> int:
    return sum(f.stat().st_size for f in root.rglob("*") if f.is_file())


def _fetch(rel: str, size: int, sha: str, root: Path, tries: int) -> str | None:
    """Download one file, verify size + SHA-256; returns an error message or None."""
    from modelscope.hub.file_download import dataset_file_download

    dest = root / rel
    err = "unknown"
    for attempt in range(tries):
        try:
            dest.unlink(missing_ok=True)
            dataset_file_download(DATASET_ID, rel, local_dir=str(root))
            if dest.stat().st_size == size and sha256_of(dest) == sha:
                return None
            err = "size or checksum mismatch"
        except Exception as exc:  # network errors surface as several exception types
            err = str(exc)[:200]
        dest.unlink(missing_ok=True)  # never leave a partial or corrupt file behind
        if attempt < tries - 1:
            time.sleep(3 * 2**attempt)
    return err


def download_subset(
    rows: list[Row], root: str | Path, budget_gb: float, log_path: str | Path, workers: int = 8, tries: int = 4
) -> dict[str, Any]:
    """Download each manifest file individually with the official ModelScope SDK.

    * resumable: files whose size and SHA-256 already match are skipped (hashes cached in .verified.json)
    * bounded retries with backoff; mismatching or partial files are deleted
    * hard budget: a file starts only if bytes on disk + bytes in flight + its size stay within budget
    """
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    root, budget = Path(root), budget_gb * 1e9
    verified_log = root / ".verified.json"
    verified: dict[str, str] = json.loads(verified_log.read_text(encoding="utf-8")) if verified_log.is_file() else {}
    stats: Counter = Counter()
    todo = []
    for rel, r in {r["video_path"]: r for r in rows}.items():
        dest = root / rel
        if dest.is_file() and dest.stat().st_size == r["size_bytes"] and (
            verified.get(rel) == r["sha256"] or sha256_of(dest) == r["sha256"]
        ):
            verified[rel] = r["sha256"]
            stats["skipped_valid"] += 1
        else:
            todo.append(r)
    committed = disk_usage(root) - sum(f.stat().st_size for r in todo if (f := root / r["video_path"]).is_file())
    log.info("%d files to download, %d already valid, %.2f GB on disk", len(todo), stats["skipped_valid"], committed / 1e9)

    with open(log_path, "a", encoding="utf-8") as logf, ThreadPoolExecutor(workers) as pool:
        pending: dict = {}
        queue = list(reversed(todo))
        while queue or pending:
            while queue and len(pending) < workers:
                r = queue[-1]
                if committed + r["size_bytes"] > budget:
                    log.error("Budget: %s (%d B) would exceed %.1f GB; not starting it or anything after it",
                              r["video_path"], r["size_bytes"], budget_gb)
                    stats["not_started_budget"] = len(queue)
                    queue.clear()
                    break
                queue.pop()
                committed += r["size_bytes"]  # reserve before starting so parallel downloads cannot overshoot
                pending[pool.submit(_fetch, r["video_path"], r["size_bytes"], r["sha256"], root, tries)] = r
            if not pending:
                break
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                r = pending.pop(fut)
                err = fut.result()
                entry = {"path": r["video_path"], "bytes": r["size_bytes"], "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
                if err:
                    committed -= r["size_bytes"]
                    stats["failed"] += 1
                    logf.write(json.dumps({**entry, "status": "failed", "error": err}) + "\n")
                else:
                    verified[r["video_path"]] = r["sha256"]
                    stats["downloaded"] += 1
                    logf.write(json.dumps({**entry, "status": "ok"}) + "\n")
                logf.flush()
                if (stats["downloaded"] + stats["failed"]) % 50 == 0:
                    verified_log.write_text(json.dumps(verified), encoding="utf-8")
                    log.info("%d/%d done (%d failed) | %.2f GB committed", stats["downloaded"] + stats["failed"],
                             len(todo), stats["failed"], committed / 1e9)
    verified_log.write_text(json.dumps(verified), encoding="utf-8")
    return {**stats, "bytes_on_disk": disk_usage(root)}


def _check_media(path: Path, ffmpeg: str) -> dict[str, Any]:
    """Header probe + one decoded middle frame + full audio decode (16 kHz mono) of one file."""
    import numpy as np

    from av_contradiction.data.preprocessing import VideoDecodeError, extract_audio, extract_frames, probe

    try:
        info = probe(path, ffmpeg)
    except VideoDecodeError as exc:
        return {"ok": False, "reason": f"corrupt: {exc}"}
    out: dict[str, Any] = {"duration_sec": round(info.duration_sec, 2), "has_video": info.has_video, "has_audio": info.has_audio}
    if info.duration_sec <= 0:
        return {**out, "ok": False, "reason": "invalid duration"}
    frames, _ = extract_frames(path, np.array([info.duration_sec / 2]), 32, ffmpeg) if info.has_video else ([], None)
    wav = extract_audio(path, 16000, ffmpeg) if info.has_audio else np.zeros(0)
    out["frame_decoded"] = len(frames) > 0
    out["audio_sec"] = round(len(wav) / 16000, 2)
    out["audio_rms"] = float(np.sqrt(np.mean(wav**2))) if len(wav) else 0.0
    problems = [p for p, bad in (("no video stream", not info.has_video), ("no audio stream", not info.has_audio),
                                 ("frame not decodable", not out["frame_decoded"]), ("audio not decodable", len(wav) == 0)) if bad]
    return {**out, "ok": not problems, "reason": "; ".join(problems)}


def verify_subset(rows: list[Row], root: str | Path, workers: int = 4) -> dict[str, Any]:
    """Files present with the expected bytes, decodable audio + video, labels equal to the official
    annotation records, no source video in two splits, class/category distribution, disk usage."""
    from concurrent.futures import ThreadPoolExecutor

    from av_contradiction.data.preprocessing import find_ffmpeg
    from av_contradiction.data.validation import audit_manifest

    root = Path(root)
    ffmpeg = find_ffmpeg()
    rep: dict[str, Any] = {"n_rows": len(rows), "problems": {}}
    official = {}
    for ann in ANNOTATIONS.values():
        with open(root / ann, encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                official[rec["qa_id"]] = rec
    label_errors = []
    for r in rows:
        rec = official.get(r["sample_id"])
        expected = {"Yes": "contradictory", "No": "consistent"}.get(rec["ground_truth"].get("exists")) if rec else None
        if rec is None or rec["video_path"] != r["video_path"] or expected != r["label"]:
            label_errors.append(r["sample_id"])
    rep["label_mismatches_vs_official"] = label_errors

    missing = [r["sample_id"] for r in rows if not (root / r["video_path"]).is_file()]
    wrong_size = [r["sample_id"] for r in rows if (root / r["video_path"]).is_file()
                  and (root / r["video_path"]).stat().st_size != int(r["size_bytes"])]
    rep["missing_files"], rep["wrong_size_files"] = missing, wrong_size
    present = [r for r in rows if r["sample_id"] not in set(missing) | set(wrong_size)]
    with ThreadPoolExecutor(workers) as pool:
        media = dict(zip((r["sample_id"] for r in present), pool.map(lambda r: _check_media(root / r["video_path"], ffmpeg), present), strict=True))
    rep["problems"] = {sid: m["reason"] for sid, m in media.items() if not m["ok"]}
    durations = sorted(m["duration_sec"] for m in media.values() if "duration_sec" in m)
    if durations:
        rep["duration_sec"] = {"min": durations[0], "median": durations[len(durations) // 2], "max": durations[-1]}
    rep["near_silent_files"] = sorted(sid for sid, m in media.items() if m.get("audio_rms", 1) < 1e-3)

    audit = audit_manifest(rows, root, check_files=True)
    rep["leakage"] = {"leaking_groups": audit["n_leaking_groups"], "cross_split_exact_duplicates": len(audit["cross_split_exact_duplicates"]),
                      "duplicate_sample_ids": len(audit["duplicate_sample_ids"]),
                      "duplicate_paths": len(rows) - len({r["video_path"] for r in rows})}
    sources = defaultdict(set)
    for r in rows:
        sources[r["group_id"]].add(r["split"])
    rep["leakage"]["source_videos_in_multiple_splits"] = sum(len(s) > 1 for s in sources.values())
    rep["leakage"]["test_rows_from_official_train"] = sum(r["split"] == "test" and r["official_split"] != "test" for r in rows)
    rep["leakage"]["train_val_rows_from_official_test"] = sum(r["split"] != "test" and r["official_split"] == "test" for r in rows)

    rep["class_distribution"] = {s: dict(Counter(r["label"] for r in rows if r["split"] == s)) for s in ("train", "val", "test")}
    rep["injection_types"] = {s: dict(Counter(r["injection_type"] for r in rows if r["split"] == s and r["injection_type"]))
                              for s in ("train", "val", "test")}
    rep["videoclass_test"] = dict(Counter(r["videoclass"] for r in rows if r["split"] == "test" and r["label"] == "consistent"))
    rep["source_videos"] = {s: len({r["group_id"] for r in rows if r["split"] == s}) for s in ("train", "val", "test")}
    rep["bytes_on_disk_total"] = disk_usage(root)
    rep["bytes_media"] = sum((root / r["video_path"]).stat().st_size for r in rows if (root / r["video_path"]).is_file())
    rep["ok"] = not (label_errors or missing or wrong_size or rep["problems"] or any(rep["leakage"].values()))
    return rep
