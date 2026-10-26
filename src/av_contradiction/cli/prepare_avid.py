"""Build, download and verify a small AVID-Bench segment subset (official ModelScope host).

Steps (each is resumable and safe to rerun):
  select    download the official annotations (63 MB), list remote file sizes, choose matched
            consistent/injected pairs within the budget and write the manifest + selection report
  download  fetch only the files in the manifest, verifying size and SHA-256, under a hard budget
  verify    decode every file, re-check labels against the official records, check split leakage

Example:
  python -m av_contradiction.cli.prepare_avid select          # defaults reproduce manifests/avid_subset.csv
  python -m av_contradiction.cli.prepare_avid download
  python -m av_contradiction.cli.prepare_avid verify
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from av_contradiction.data.avid import (
    ANNOTATIONS,
    DATASET_ID,
    SEMANTIC_TYPES,
    disk_usage,
    download_subset,
    list_remote,
    select_subset,
    verify_subset,
)
from av_contradiction.data.manifest import read_manifest, write_manifest


def _annotations(root: Path) -> None:
    from modelscope.hub.file_download import dataset_file_download

    for rel in ANNOTATIONS.values():
        if not (root / rel).is_file():
            dataset_file_download(DATASET_ID, rel, local_dir=str(root))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/avid", help="Dataset folder (mirrors the official layout)")
    ap.add_argument("--manifest", default="manifests/avid_subset.csv", help="Tracked in git so the subset is reproducible")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--train-pairs", type=int, default=1000)
    s.add_argument("--val-pairs", type=int, default=150)
    s.add_argument("--test-pairs", type=int, default=150)
    s.add_argument("--budget-gb", type=float, default=8.0, help="Media budget for the selection")
    s.add_argument("--max-pair-mb", type=float, default=None, help="Skip pairs larger than this (MB)")
    s.add_argument("--categories", nargs="+", default=list(SEMANTIC_TYPES))
    s.add_argument("--seed", type=int, default=0)
    d = sub.add_parser("download")
    d.add_argument("--budget-gb", type=float, default=10.0, help="Hard limit for everything under --root")
    d.add_argument("--workers", type=int, default=8)
    d.add_argument("--limit", type=int, default=None, help="Only the first N files (pilot run)")
    sub.add_parser("verify")
    args = ap.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    if args.cmd == "select":
        _annotations(root)
        remote = list_remote(root / "remote_listing")
        rows, report = select_subset(
            root, remote, {"train": args.train_pairs, "val": args.val_pairs, "test": args.test_pairs},
            args.budget_gb, tuple(args.categories), args.max_pair_mb, args.seed,
        )
        write_manifest(rows, args.manifest)
        report["manifest"] = args.manifest
        (root / "selection_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
    elif args.cmd == "download":
        rows = read_manifest(args.manifest)
        rows = rows[: args.limit] if args.limit else rows
        for r in rows:
            r["size_bytes"] = int(r["size_bytes"])
        stats = download_subset(rows, root, args.budget_gb, root / "download_log.jsonl", args.workers)
        print(json.dumps(stats, indent=2), f"\n{disk_usage(root) / 1e9:.2f} GB under {root}")
    else:
        rep = verify_subset(read_manifest(args.manifest), root)
        (root / "verification_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
        short = {k: (v if not isinstance(v, list) else len(v)) for k, v in rep.items()}
        short["problems"] = len(rep["problems"])
        print(json.dumps(short, indent=2))


if __name__ == "__main__":
    main()
