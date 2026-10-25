"""Render the synthetic ColorTone sanity benchmark and write a split manifest.

This is a mechanism check with a known rule (colour<->tone), NOT a semantic-contradiction dataset.

Usage: python -m av_contradiction.cli.make_synthetic_benchmark --out data/synthetic_colortone --scenes 150
"""

from __future__ import annotations

import argparse
from collections import Counter

from av_contradiction.data.manifest import write_manifest
from av_contradiction.data.splits import check_group_leakage, group_stratified_split
from av_contradiction.data.synthetic import generate_colortone


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/synthetic_colortone")
    ap.add_argument("--manifest", default="data/manifests/synthetic_colortone.csv")
    ap.add_argument("--scenes", type=int, default=150, help="Each scene yields 1 consistent + 1 contradictory clip")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = generate_colortone(args.out, args.scenes, args.seed)
    rows = group_stratified_split(rows, (0.6, 0.2, 0.2), seed=args.seed)
    assert not check_group_leakage(rows)
    write_manifest(rows, args.manifest)
    print(f"Wrote {len(rows)} clips to {args.out} and manifest {args.manifest}")
    print(dict(Counter((r["split"], r["label"]) for r in rows)))


if __name__ == "__main__":
    main()
