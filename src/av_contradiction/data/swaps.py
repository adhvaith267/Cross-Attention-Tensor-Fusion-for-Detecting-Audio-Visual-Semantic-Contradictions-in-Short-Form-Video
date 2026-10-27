"""Audio-swap benchmark built from cached features: natural audio, no editing artefacts.

AVID-Bench's contradictions are edited audio, so an audio-only model can find them without looking
at the video. Here every clip is real and unedited:

* consistent:    clip A's video with A's own audio (an existing row)
* contradictory: clip A's video with clip B's audio (a new row that points at both feature files)

Within each split, the swaps form a derangement: every audio track and every video appears exactly
once as consistent and once as contradictory, so neither modality alone carries any label
information. A model has to compare the two.

Partners are matched only within a split (no leakage) and only between clips with the same number
of frames and audio windows. A swapped row keeps the video source's timeline (``video_ts``,
``audio_ts``), so clip length and window timing are identical to the consistent row and cannot give
the label away. Clips whose audio is mostly silent are dropped: silence over a scene is not a
semantic contradiction.

Pairing:
  random  partners drawn at random within each length bucket
  hard    within each bucket, the assignment that maximises the CLAP similarity between the partner's
          audio and the clip's own audio (e.g. speech replaced by other speech). Harder, but some
          swaps may plausibly still fit the video.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from av_contradiction.config import Config
from av_contradiction.data.features import cache_path
from av_contradiction.data.manifest import Row

MAX_SILENT_FRACTION = 0.5


def _profile(cfg: Config, row: Row) -> dict[str, Any] | None:
    p = cache_path(cfg, row["sample_id"])
    if not p.exists():
        return None
    f = torch.load(p, map_location="cpu", weights_only=True)
    audio = f["audio"].float()
    return {
        "n_audio": int(audio.shape[0]),
        "n_video": int(f["video"].shape[0]),
        "silent": float(f["audio_silent"].float().mean()) if audio.shape[0] else 1.0,
        "audio_mean": torch.nn.functional.normalize(audio.mean(0), dim=0).numpy() if audio.shape[0] else None,
    }


def _derangement(n: int, rng: random.Random) -> list[int]:
    """A random cyclic shift of a shuffled order: perm[i] != i for every i (n >= 2)."""
    order = list(range(n))
    rng.shuffle(order)
    perm = [0] * n
    for k, i in enumerate(order):
        perm[i] = order[(k + 1) % n]
    return perm


def _hard_assignment(emb: np.ndarray) -> list[int]:
    """Assignment maximising audio similarity to each clip's own audio, never a clip's own audio."""
    sim = emb @ emb.T
    np.fill_diagonal(sim, -1e9)
    _, cols = linear_sum_assignment(-sim)
    return [int(c) for c in cols]


def build_audio_swaps(
    cfg: Config, rows: list[Row], pairing: str = "random", seed: int = 0, consistent_label: str = "consistent",
    contradictory_label: str = "contradictory",
) -> tuple[list[Row], dict[str, Any]]:
    """Return (manifest rows, report). Uses only the ``consistent_label`` rows of ``rows``; their
    features must already be cached."""
    if pairing not in ("random", "hard"):
        raise ValueError("pairing must be 'random' or 'hard'")
    rng = random.Random(seed)
    sources = [r for r in rows if r["label"] == consistent_label]
    report: dict[str, Any] = {"pairing": pairing, "seed": seed, "candidates": len(sources),
                              "dropped": Counter(), "by_split": {}}
    buckets: dict[tuple[str, int, int], list[tuple[Row, dict[str, Any]]]] = defaultdict(list)
    for r in sources:
        prof = _profile(cfg, r)
        if prof is None:
            report["dropped"]["no cached features"] += 1
        elif prof["n_audio"] == 0 or prof["n_video"] == 0:
            report["dropped"]["missing audio or video"] += 1
        elif prof["silent"] > MAX_SILENT_FRACTION:
            report["dropped"]["mostly silent audio"] += 1
        else:
            buckets[(r.get("split", ""), prof["n_video"], prof["n_audio"])].append((r, prof))

    out: list[Row] = []
    for key in sorted(buckets):
        members = buckets[key]
        if len(members) < 2:
            report["dropped"]["no partner of the same length"] += len(members)
            continue
        if pairing == "random":
            perm = _derangement(len(members), rng)
        else:
            perm = _hard_assignment(np.stack([p["audio_mean"] for _, p in members]))
        for i, (row, _) in enumerate(members):
            partner, _ = members[perm[i]]
            vid, aid = row["sample_id"], partner["sample_id"]
            out.append({**row, "is_synthetic": False})
            out.append({
                "sample_id": f"swap:{vid}+{aid}", "video_path": row["video_path"], "label": contradictory_label,
                "split": row.get("split", ""), "group_id": row.get("group_id", ""),
                "source_dataset": row.get("source_dataset", ""), "source_id": row.get("source_id", ""),
                "license": row.get("license", ""), "is_synthetic": True,
                "generation_method": f"audio swap ({pairing}): video of {vid}, audio of {aid}",
                "video_source_id": row.get("video_source_id", vid), "audio_source_id": partner.get("audio_source_id", aid),
                "video_feature_id": vid, "audio_feature_id": aid,
                "notes": "Natural audio from another clip of the same length",
            })
    for split in sorted({r["split"] for r in out}):
        rs = [r for r in out if r["split"] == split]
        report["by_split"][split] = dict(Counter(r["label"] for r in rs))
    report["dropped"] = dict(report["dropped"])
    report["rows"] = len(out)
    report["balance_check"] = check_swap_balance(out, consistent_label, contradictory_label)
    return out, report


def check_swap_balance(rows: list[Row], consistent_label: str = "consistent", contradictory_label: str = "contradictory") -> dict[str, Any]:
    """Verify that each video and each audio appears once per class and partners share a split."""
    split_of = {r["sample_id"]: r.get("split", "") for r in rows if r["label"] == consistent_label}
    video_uses: Counter = Counter()
    audio_uses: Counter = Counter()
    problems: list[str] = []
    for r in rows:
        vid = r.get("video_feature_id") or r["sample_id"]
        aid = r.get("audio_feature_id") or r["sample_id"]
        video_uses[(vid, r["label"])] += 1
        audio_uses[(aid, r["label"])] += 1
        if r["label"] == contradictory_label:
            if vid == aid:
                problems.append(f"{r['sample_id']}: audio and video from the same clip")
            if split_of.get(vid) != r.get("split") or split_of.get(aid) != r.get("split"):
                problems.append(f"{r['sample_id']}: partner from another split")
    for sid in split_of:
        for uses, name in ((video_uses, "video"), (audio_uses, "audio")):
            if uses[(sid, consistent_label)] != 1 or uses[(sid, contradictory_label)] != 1:
                problems.append(f"{sid}: {name} not used exactly once per class")
    return {"ok": not problems, "problems": problems[:20], "n_problems": len(problems)}
