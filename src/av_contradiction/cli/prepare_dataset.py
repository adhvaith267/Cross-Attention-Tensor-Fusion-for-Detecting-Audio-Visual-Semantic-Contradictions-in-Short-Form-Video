"""Build a split manifest from a supported source.

Subcommands:
  avhbench      AVHBench AV-matching QA json + video folder
  vggsound-swap VGGSound csv + local clips + conflict-group json -> originals + within-group audio swaps
  split         (re)assign leakage-safe group-stratified splits to an existing manifest

Examples:
  python -m av_contradiction.cli.prepare_dataset avhbench --qa-json data/raw/avhbench/QA.json --video-dir data/raw/avhbench/videos --out data/manifests/avhbench.csv
  python -m av_contradiction.cli.prepare_dataset vggsound-swap --csv data/raw/vggsound/vggsound.csv --clip-dir data/raw/vggsound/clips \
      --groups data/conflict_groups.json --work-dir data/derived/vggsound --out data/manifests/vggsound_swap.csv
  python -m av_contradiction.cli.prepare_dataset split --manifest my.csv --out data/manifests/manifest.csv
"""

from __future__ import annotations

import argparse
from collections import Counter

from av_contradiction.data.adapters import avhbench_av_matching, vggsound_conflict_swaps
from av_contradiction.data.manifest import read_manifest, write_manifest
from av_contradiction.data.splits import (
    assign_groups_from_sources,
    check_group_leakage,
    group_stratified_split,
)


def finalize(rows, out: str, ratios: tuple[float, float, float], seed: int, keep_splits: bool) -> None:
    rows = assign_groups_from_sources(rows)
    if not keep_splits or any(not r.get("split") for r in rows):
        rows = group_stratified_split(rows, ratios, seed)
    leaks = check_group_leakage(rows)
    if leaks:
        raise SystemExit(f"Refusing to write manifest: {len(leaks)} groups leak across splits")
    write_manifest(rows, out)
    print(f"Wrote {len(rows)} rows -> {out}")
    print(dict(sorted(Counter((r["split"], r["label"]) for r in rows).items())))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ratios", type=float, nargs=3, default=(0.7, 0.15, 0.15), metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--seed", type=int, default=42)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("avhbench")
    a.add_argument("--qa-json", required=True)
    a.add_argument("--video-dir", required=True)
    a.add_argument("--group-regex", default=None, help="Regex whose first group maps video_id -> source group")
    a.add_argument("--out", required=True)

    v = sub.add_parser("vggsound-swap")
    v.add_argument("--csv", required=True)
    v.add_argument("--clip-dir", required=True)
    v.add_argument("--groups", required=True, help="Conflict-group JSON {group: [class, ...]}")
    v.add_argument("--work-dir", required=True, help="Where swapped clips are written")
    v.add_argument("--pattern", default="{ytid}_{start:06d}.mp4")
    v.add_argument("--max-per-class", type=int, default=None)
    v.add_argument("--out", required=True)

    s = sub.add_parser("split")
    s.add_argument("--manifest", required=True)
    s.add_argument("--keep-existing", action="store_true", help="Keep splits already present in the manifest")
    s.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.cmd == "avhbench":
        rows = avhbench_av_matching(args.qa_json, args.video_dir, args.group_regex)
        finalize(rows, args.out, tuple(args.ratios), args.seed, False)
    elif args.cmd == "vggsound-swap":
        rows = vggsound_conflict_swaps(args.csv, args.clip_dir, args.groups, args.work_dir, args.pattern, args.max_per_class, args.seed)
        finalize(rows, args.out, tuple(args.ratios), args.seed, False)
    else:
        finalize(read_manifest(args.manifest), args.out, tuple(args.ratios), args.seed, args.keep_existing)


if __name__ == "__main__":
    main()
