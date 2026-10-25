"""Run baselines + ablations on identical splits and features, over one or more seeds.

Usage: python -m av_contradiction.cli.run_ablations --seeds 1 2 3
Outputs: <evaluation.output_dir>/<experiment>/ablation_{runs,summary}.json, ablation_summary.md, ablation_macro_f1.png,
and best_model/{best.pt, calibration.json, best_model.json}: the run with the highest validation macro-F1.
"""

from __future__ import annotations

import argparse

from av_contradiction.cli import add_config_args
from av_contradiction.config import load_config
from av_contradiction.evaluation.ablations import DEFAULT_SUITE, PRESETS, run_suite


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--presets", nargs="+", default=list(DEFAULT_SUITE), choices=sorted(PRESETS) + ["majority", "similarity"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    res = run_suite(cfg, args.presets, args.seeds)
    print(res["markdown"])
    print("Saved to", res["output_dir"])
    if b := res["best"]:
        print(f"Best model (chosen on validation): {b['preset']} seed {b['seed']} | val macro-F1 {b['val_macro_f1']:.3f} "
              f"| test macro-F1 {b['test_macro_f1']:.3f} -> {res['output_dir']}/best_model/best.pt")


if __name__ == "__main__":
    main()
