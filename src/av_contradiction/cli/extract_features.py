"""Decode videos and cache frozen-encoder features for every manifest row.

Usage: python -m av_contradiction.cli.extract_features [--split train] [--overwrite] [--device cuda]
Failures (corrupt files, decode errors) are listed in <cache>/extraction_report.json and skipped.
"""

from __future__ import annotations

import argparse
import json

from av_contradiction.cli import add_config_args
from av_contradiction.config import load_config
from av_contradiction.data.features import cache_dir_for, extract_manifest
from av_contradiction.data.manifest import read_manifest
from av_contradiction.utils.device import select_device


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--split", default=None, help="Only this split (default: all rows)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=None, help="Only the first N rows (smoke tests)")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    rows = read_manifest(cfg.data.manifest)
    if args.split:
        rows = [r for r in rows if r.get("split") == args.split]
    rows = rows[: args.limit] if args.limit else rows
    device = select_device(args.device)
    print(f"Extracting {len(rows)} clips with video={cfg.encoders.video}, audio={cfg.encoders.audio} on {device} -> {cache_dir_for(cfg)}")
    report = extract_manifest(cfg, rows, device, args.overwrite)
    path = cache_dir_for(cfg) / "extraction_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"extracted={report['extracted']} skipped(existing)={report['skipped']} failed={len(report['failed'])} in {report['seconds']}s")
    for sid, err in list(report["failed"].items())[:10]:
        print(f"  FAILED {sid}: {err}")


if __name__ == "__main__":
    main()
