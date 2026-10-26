"""Baseline/ablation suite: same splits, same cached features, multiple seeds.

Every neural preset is trained with checkpoint selection on validation macro-F1 and then
evaluated once on the test split. No preset or hyperparameter is chosen using test results:
the overall best model is the neural run with the highest *validation* macro-F1, copied to
``<output_dir>/<experiment>/best_model/``.
"""

from __future__ import annotations

import copy
import shutil
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from av_contradiction.config import Config, replace_section
from av_contradiction.data.dataset import CachedFeatureDataset
from av_contradiction.data.manifest import filter_split, read_manifest
from av_contradiction.evaluation.evaluate import evaluate_checkpoint
from av_contradiction.evaluation.reporting import aggregate, markdown_table, plot_comparison, save_json
from av_contradiction.models.baselines import MajorityClassBaseline, ProjectedSimilarityBaseline
from av_contradiction.training.checkpointing import load_checkpoint, save_checkpoint
from av_contradiction.training.metrics import compute_metrics
from av_contradiction.training.trainer import train_from_config
from av_contradiction.utils.logging import get_logger

log = get_logger(__name__)

# name -> ModelConfig overrides
PRESETS: dict[str, dict[str, Any]] = {
    "audio_only": {"use_audio": True, "use_video": False, "use_cross_attention": False, "fusion": "concat"},
    "video_only": {"use_audio": False, "use_video": True, "use_cross_attention": False, "fusion": "concat"},
    "concat": {"use_audio": True, "use_video": True, "use_cross_attention": False, "fusion": "concat"},
    "tensor_no_xattn": {"use_audio": True, "use_video": True, "use_cross_attention": False, "fusion": "tensor"},
    "xattn_concat": {"use_audio": True, "use_video": True, "use_cross_attention": True, "fusion": "concat"},
    "xattn_tensor_global": {"use_audio": True, "use_video": True, "use_cross_attention": True, "fusion": "tensor", "alignment": "global"},
    "xattn_tensor": {"use_audio": True, "use_video": True, "use_cross_attention": True, "fusion": "tensor", "alignment": "timestamp"},
}
DEFAULT_SUITE = ("majority", "similarity", "audio_only", "video_only", "concat", "xattn_concat", "xattn_tensor", "xattn_tensor_global", "tensor_no_xattn")


def preset_config(cfg: Config, preset: str, seed: int) -> Config:
    if preset not in PRESETS:
        raise KeyError(f"Unknown preset '{preset}'. Options: {sorted(PRESETS)}")
    new = replace_section(cfg, model=PRESETS[preset], training={"seed": seed})
    new.experiment_name = f"{cfg.experiment_name}/{preset}/seed{seed}"
    return new


def _pooled(ds: CachedFeatureDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a, v = [], []
    for i in range(len(ds)):
        it = ds[i]
        a.append(it["audio"].mean(0).numpy() if it["audio"].shape[0] else np.zeros(it["audio"].shape[-1], np.float32))
        v.append(it["video"].mean(0).numpy() if it["video"].shape[0] else np.zeros(it["video"].shape[-1], np.float32))
    return np.stack(a), np.stack(v), ds.labels


def run_non_neural(cfg: Config, name: str) -> dict[str, Any]:
    rows = read_manifest(cfg.data.manifest)
    tr, va, te = (CachedFeatureDataset(cfg, filter_split(rows, s)) for s in ("train", "val", cfg.evaluation.split))
    labels_names = list(cfg.data.label_names)
    if name == "majority":
        model = MajorityClassBaseline().fit(tr.labels, len(labels_names))
        probs = model.predict_proba(len(te))
        note = "Predicts the most frequent training class."
    elif name == "similarity":
        atr, vtr, ltr = _pooled(tr)
        ava, vva, lva = _pooled(va)
        ate, vte, _ = _pooled(te)
        model = ProjectedSimilarityBaseline().fit(atr, vtr, ltr).tune_threshold(ava, vva, lva)
        probs = model.predict_proba(ate, vte)
        note = "Ridge audio->video map fitted on consistent train clips; 1-cosine score; threshold tuned on val. Scores are uncalibrated."
    else:
        raise KeyError(name)
    return {"preset": name, "seed": cfg.training.seed, "status": "ok", "note": note, "params": 0, "latency_ms": None,
            "metrics": compute_metrics(te.labels, probs, labels_names, cfg.evaluation.ece_bins)}


def run_suite(cfg: Config, presets: list[str] | tuple[str, ...] = DEFAULT_SUITE, seeds: list[int] | tuple[int, ...] = (42,), output_dir: str | Path | None = None) -> dict[str, Any]:
    out = Path(output_dir or cfg.evaluation.output_dir) / cfg.experiment_name
    results: list[dict[str, Any]] = []
    for preset in presets:
        for seed in seeds if preset in PRESETS else seeds[:1]:
            try:
                if preset in PRESETS:
                    pcfg = preset_config(cfg, preset, seed)
                    train = train_from_config(pcfg, preset=preset)
                    if train.best_checkpoint is None:
                        raise RuntimeError("Training produced no best checkpoint")
                    ev = evaluate_checkpoint(train.best_checkpoint, overrides=None, output_dir=cfg.evaluation.output_dir)
                    results.append({
                        "preset": preset, "seed": seed, "status": "ok", "val_macro_f1": train.best_metric, "metrics": ev["metrics"],
                        "metrics_uncalibrated": ev["metrics_uncalibrated"], "params": ev["params"],
                        "latency_ms": ev["latency_ms_per_clip_fusion_only"], "best_epoch": train.best_epoch,
                        "stopped_reason": train.stopped_reason, "checkpoint": train.best_checkpoint,
                        "macro_f1_ci95": ev["macro_f1_ci95"], "temperature": ev.get("temperature"), "report_dir": ev["report_dir"],
                    })
                else:
                    results.append(run_non_neural(copy.deepcopy(cfg), preset))
            except Exception as exc:  # record failures honestly instead of dropping them
                log.error("Preset %s seed %d failed: %s", preset, seed, exc)
                results.append({"preset": preset, "seed": seed, "status": "failed", "error": str(exc), "traceback": traceback.format_exc()})
            save_json(results, out / "ablation_runs.json")
    table = aggregate(results)
    save_json(table, out / "ablation_summary.json")
    md = markdown_table(table)
    (out / "ablation_summary.md").write_text(md + "\n", encoding="utf-8")
    plot_comparison(table, out / "ablation_macro_f1.png")
    best = save_best_model(results, out / "best_model")
    log.info("Ablation summary:\n%s", md)
    return {"runs": results, "summary": table, "markdown": md, "output_dir": str(out), "best": best}


def save_best_model(results: list[dict[str, Any]], dest: Path) -> dict[str, Any] | None:
    """Copy the neural run with the highest validation macro-F1 (checkpoint, calibration and its test
    report, so the dashboard can show the results on another machine) to ``dest``."""
    neural = [r for r in results if r.get("status") == "ok" and r.get("val_macro_f1") is not None]
    if not neural:
        return None
    best = max(neural, key=lambda r: r["val_macro_f1"])
    ckpt = Path(best["checkpoint"])
    dest.mkdir(parents=True, exist_ok=True)
    # Inference needs no optimizer/scheduler state; dropping it keeps the released file small (~13 MB).
    state = {k: v for k, v in load_checkpoint(ckpt).items() if k not in ("optimizer_state", "scheduler_state", "scaler_state")}
    save_checkpoint(dest / "best.pt", state)
    if (cal := ckpt.with_name("calibration.json")).is_file():
        shutil.copy2(cal, dest / "calibration.json")
    if (report := Path(best["report_dir"]) / "metrics_test.json").is_file():
        shutil.copy2(report, dest / "metrics_test.json")
    info = {"preset": best["preset"], "seed": best["seed"], "selected_by": "validation macro_f1",
            "val_macro_f1": best["val_macro_f1"], "test_macro_f1": best["metrics"]["macro_f1"],
            "test_macro_f1_ci95": best["macro_f1_ci95"], "test_accuracy": best["metrics"]["accuracy"],
            "source_checkpoint": str(ckpt)}
    save_json(info, dest / "best_model.json")
    log.info("Best model (by val macro-F1): %s seed %s -> %s", best["preset"], best["seed"], dest)
    return info
