"""Evaluate a checkpoint on a held-out split (default: test).

Usage: python -m av_contradiction.cli.evaluate --checkpoint output/checkpoints/default/best.pt [--split test] [--set data.manifest=...]
Writes metrics JSON, predictions CSV, confusion matrix and reliability plots to <evaluation.output_dir>/<experiment>/,
and calibration.json (temperature fitted on val) next to the checkpoint.
"""

from __future__ import annotations

import argparse
import json

from av_contradiction.evaluation.evaluate import evaluate_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-calibration-file", action="store_true", help="Do not write calibration.json")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    res = evaluate_checkpoint(args.checkpoint, args.overrides, args.split, args.output_dir, args.device, not args.no_calibration_file)
    m = res["metrics"]
    print(json.dumps({
        "split": res["split"], "n": m["n"], "macro_f1": m["macro_f1"], "macro_f1_ci95": res["macro_f1_ci95"],
        "accuracy": m["accuracy"], "roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"], "ece": m["ece"], "brier": m["brier"],
        "temperature": res.get("temperature"), "confusion_matrix": m["confusion_matrix"],
        "params_head": res["params_trainable_head"], "latency_ms_fusion_only": res["latency_ms_per_clip_fusion_only"],
        "peak_gpu_memory_mb": res["peak_gpu_memory_mb"], "warnings": res["warnings"],
    }, indent=2))


if __name__ == "__main__":
    main()
