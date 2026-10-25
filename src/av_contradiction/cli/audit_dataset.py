"""Audit a manifest: schema, labels, missing/corrupt files, duplicates, split leakage, class balance.

Usage:
  python -m av_contradiction.cli.audit_dataset --manifest data/manifests/manifest.csv --root-dir data/raw [--probe] [--near-duplicates]
Exit code 1 if errors were found.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from av_contradiction.data.manifest import manifest_hash, read_manifest
from av_contradiction.data.validation import audit_manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root-dir", default=".", help="Directory relative video paths are resolved against")
    ap.add_argument("--labels", nargs="+", default=["consistent", "contradictory"])
    ap.add_argument("--no-files", action="store_true", help="Skip file existence checks")
    ap.add_argument("--probe", action="store_true", help="Decode headers with FFmpeg to find corrupt files / missing streams")
    ap.add_argument("--near-duplicates", action="store_true", help="Perceptual-hash middle frames (implies --probe)")
    ap.add_argument("--out", default=None, help="Write the JSON report here")
    args = ap.parse_args()

    rows = read_manifest(args.manifest)
    report = audit_manifest(
        rows, args.root_dir, args.labels, check_files=not args.no_files,
        probe_files=args.probe or args.near_duplicates, near_duplicates=args.near_duplicates,
    )
    report["manifest"] = args.manifest
    report["manifest_sha256"] = manifest_hash(args.manifest)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"rows: {report['n_rows']} | sha256: {report['manifest_sha256'][:16]}")
    print("class distribution by split:", json.dumps(report.get("class_distribution_by_split", {})))
    print("synthetic by split:", json.dumps(report.get("synthetic_by_split", {})))
    for e in report["errors"]:
        print("ERROR:", e)
    for w in report["warnings"]:
        print("WARNING:", w)
    print("OK" if report.get("ok") else f"{len(report.get('invalid_sample_ids', []))} invalid samples")
    sys.exit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
