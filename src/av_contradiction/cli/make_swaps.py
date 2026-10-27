"""Build an audio-swap manifest from the consistent clips of a manifest (natural audio, no edits).

Every clip appears once with its own audio (consistent) and once with the audio of another clip of the
same length from the same split (contradictory), so neither modality alone predicts the label. Needs
the consistent clips' features in the cache (run extract_features first); writes no videos.

Usage:
  python -m av_contradiction.cli.make_swaps --source manifests/avid_subset.csv --out output/avid_swap.csv \
      [--pairing random|hard] [--seed 0] [--set data.feature_cache_dir=...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from av_contradiction.cli import add_config_args
from av_contradiction.config import load_config
from av_contradiction.data.manifest import read_manifest, write_manifest
from av_contradiction.data.swaps import build_audio_swaps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--source", required=True, help="Manifest whose consistent rows are the clips to swap")
    ap.add_argument("--out", required=True, help="Output manifest (.csv or .jsonl); a _report.json is written next to it")
    ap.add_argument("--pairing", choices=["random", "hard"], default="random")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    rows, report = build_audio_swaps(cfg, read_manifest(args.source), args.pairing, args.seed)
    if not report["balance_check"]["ok"]:
        raise SystemExit(f"Balance check failed: {report['balance_check']}")
    write_manifest(rows, args.out)
    out = Path(args.out)
    out.with_name(out.stem + "_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "balance_check"} | {"balance_ok": True}, indent=2))
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
