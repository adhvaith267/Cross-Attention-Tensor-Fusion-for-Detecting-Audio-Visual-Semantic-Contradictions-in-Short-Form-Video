"""Leakage-safe, group-aware, label-stratified splitting."""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from av_contradiction.data.manifest import Row


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def assign_groups_from_sources(rows: list[Row]) -> list[Row]:
    """Set ``group_id`` so samples sharing any source (audio, video or original id) share a group.

    A swapped negative made from audio of clip A and video of clip B links A and B, so every
    sample derived from either ends up in the same split.
    """
    uf = UnionFind()
    for r in rows:
        keys = [f"{k}:{r[k]}" for k in ("source_id", "audio_source_id", "video_source_id") if r.get(k)]
        if r.get("group_id"):
            keys.append(f"group:{r['group_id']}")
        keys = keys or [f"sample:{r['sample_id']}"]
        for k in keys[1:]:
            uf.union(keys[0], k)
        r["_group_key"] = keys[0]
    for r in rows:
        r["group_id"] = uf.find(r.pop("_group_key"))
    return rows


def group_stratified_split(
    rows: list[Row],
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    names: tuple[str, str, str] = ("train", "val", "test"),
) -> list[Row]:
    """Assign whole groups to splits, greedily balancing both split size and label distribution.

    Groups are shuffled deterministically, then largest-first each group goes to the split with
    the largest remaining deficit (target count minus current count, summed over the group's labels).
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1")
    groups: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        groups[r.get("group_id") or r["sample_id"]].append(r)
    label_totals = Counter(r["label"] for r in rows)
    targets = {s: {lab: n * ratio for lab, n in label_totals.items()} for s, ratio in zip(names, ratios, strict=True)}
    counts: dict[str, Counter] = {s: Counter() for s in names}

    rng = np.random.default_rng(seed)
    keys = list(groups)
    rng.shuffle(keys)
    keys.sort(key=lambda k: -len(groups[k]))  # stable sort keeps the shuffled order within equal sizes
    for key in keys:
        g_labels = Counter(r["label"] for r in groups[key])
        best = max(
            names,
            key=lambda s: sum((targets[s][lab] - counts[s][lab]) * n for lab, n in g_labels.items()),
        )
        counts[best].update(g_labels)
        for r in groups[key]:
            r["split"] = best
    return rows


def check_group_leakage(rows: list[Row]) -> dict[str, list[str]]:
    """Return {group_id: [splits]} for groups that appear in more than one split."""
    seen: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        if r.get("group_id"):
            seen[r["group_id"]].add(r.get("split", ""))
    return {g: sorted(s) for g, s in seen.items() if len(s) > 1}
