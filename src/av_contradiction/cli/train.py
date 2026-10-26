"""Train a model (proposed or a baseline preset) on cached features.

Usage:
  python -m av_contradiction.cli.train
  python -m av_contradiction.cli.train --preset concat
  python -m av_contradiction.cli.train --resume output/checkpoints/default/last.pt
Checkpoints: <training.checkpoint_dir>/<experiment_name>/{best,last}.pt
"""

from __future__ import annotations

import argparse

from av_contradiction.cli import add_config_args
from av_contradiction.config import load_config, replace_section
from av_contradiction.evaluation.ablations import PRESETS
from av_contradiction.training.trainer import train_from_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--preset", choices=sorted(PRESETS), default=None, help="Apply a baseline/ablation preset")
    ap.add_argument("--resume", default=None, help="Checkpoint to resume from (e.g. .../last.pt)")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    if args.preset:
        cfg = replace_section(cfg, model=PRESETS[args.preset])
        cfg.experiment_name = f"{cfg.experiment_name}/{args.preset}"
    result = train_from_config(cfg, resume=args.resume, preset=args.preset or "custom")
    print(f"stopped: {result.stopped_reason} | epochs: {result.epochs_run} | best val macro-F1: {result.best_metric} "
          f"(epoch {result.best_epoch}) | best checkpoint: {result.best_checkpoint}")


if __name__ == "__main__":
    main()
