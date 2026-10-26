"""Predict consistency/contradiction for one video.

Usage: python -m av_contradiction.cli.predict --video sample.mp4 --checkpoint output/checkpoints/default/best.pt [--out pred.json]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from av_contradiction.inference.predictor import Predictor


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None, help="Also write the JSON result here")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    pred = Predictor(args.checkpoint, args.device, args.overrides).predict(args.video)
    text = pred.model_dump_json(indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
