"""Held-out evaluation of a trained checkpoint.

The checkpoint was selected on validation metrics during training; this module only *reports*
test metrics. Optional temperature scaling is fitted on the validation split and applied to the
test split; both raw and calibrated metrics are reported.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from av_contradiction.config import Config, build_config, parse_overrides
from av_contradiction.data.dataset import (
    CachedFeatureDataset,
    RawVideoDataset,
    collate_features,
    collate_raw,
    move_batch,
)
from av_contradiction.data.manifest import filter_split, manifest_hash, read_manifest
from av_contradiction.evaluation.reporting import (
    plot_confusion,
    plot_reliability,
    save_json,
    write_predictions,
)
from av_contradiction.models.audio_encoder import build_audio_encoder
from av_contradiction.models.classifier import EndToEndModel, build_model, count_parameters
from av_contradiction.models.video_encoder import build_video_encoder
from av_contradiction.training.checkpointing import load_checkpoint
from av_contradiction.training.metrics import (
    TEMP_BOUNDS,
    bootstrap_ci,
    compute_metrics,
    fit_temperature,
    softmax,
)
from av_contradiction.utils.device import peak_memory_mb, reset_peak_memory, select_device
from av_contradiction.utils.logging import get_logger

log = get_logger(__name__)


def config_from_checkpoint(ckpt: dict[str, Any], overrides: list[str] | None = None) -> Config:
    raw = ckpt["config"]
    raw.setdefault("hardware", {})["profile"] = "none"
    return build_config(raw, overrides, strict=False)


def load_model(ckpt: dict[str, Any], cfg: Config, device: torch.device) -> nn.Module:
    model: nn.Module = build_model(cfg.model, ckpt["audio_dim"], ckpt["video_dim"])
    model.load_state_dict(ckpt["model_state"])
    if "encoder_state" in ckpt:
        e2e = EndToEndModel(build_video_encoder(cfg.encoders, cfg.preprocess), build_audio_encoder(cfg.encoders, cfg.preprocess), model)
        e2e.load_state_dict(ckpt["encoder_state"], strict=False)
        model = e2e
    return model.to(device).eval()


def _dataset(cfg: Config, rows, model: nn.Module):
    if isinstance(model, EndToEndModel):
        return RawVideoDataset(cfg, rows, model.video_encoder, model.audio_encoder), collate_raw
    ds = CachedFeatureDataset(cfg, rows)
    if ds.missing:
        log.warning("%d samples lack cached features and are excluded from evaluation", len(ds.missing))
    return ds, collate_features


@torch.no_grad()
def collect_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, list[str]]:
    logits, labels, ids = [], [], []
    for batch in loader:
        batch = move_batch(batch, device)
        logits.append(model(batch)["logits"].float().cpu().numpy())
        labels.append(batch["label"].cpu().numpy())
        ids += batch["sample_id"]
    if not logits:
        return np.zeros((0, 2)), np.zeros(0, int), []
    return np.concatenate(logits), np.concatenate(labels), ids


@torch.no_grad()
def measure_latency(model: nn.Module, dataset, collate, device: torch.device, warmup: int, iters: int) -> float | None:
    """Mean forward latency (ms) for a single clip on pre-extracted features (excludes decoding/encoding)."""
    if len(dataset) == 0:
        return None
    batch = move_batch(collate([dataset[0]]), device)
    for _ in range(warmup):
        model(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        model(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / max(1, iters) * 1000


def evaluate_checkpoint(
    checkpoint: str | Path,
    overrides: list[str] | None = None,
    split: str | None = None,
    output_dir: str | Path | None = None,
    device: str = "auto",
    write_calibration: bool = True,
) -> dict[str, Any]:
    ckpt = load_checkpoint(checkpoint, "cpu")
    cfg = config_from_checkpoint(ckpt, overrides)
    split = split or cfg.evaluation.split
    dev = select_device(device)
    model = load_model(ckpt, cfg, dev)
    labels_names = list(ckpt["label_names"])

    rows = read_manifest(cfg.data.manifest)
    mh = manifest_hash(cfg.data.manifest)
    warnings: list[str] = []
    if mh != ckpt.get("manifest_hash"):
        warnings.append("Manifest hash differs from the one used in training; splits may not be comparable.")
    test_ds, collate = _dataset(cfg, filter_split(rows, split), model)
    if len(test_ds) == 0:
        raise ValueError(f"No evaluable samples in split '{split}'")
    loader = DataLoader(test_ds, batch_size=cfg.training.batch_size, shuffle=False, collate_fn=collate, num_workers=cfg.training.num_workers)
    reset_peak_memory(dev)
    logits, labels, ids = collect_logits(model, loader, dev)
    peak = peak_memory_mb(dev)
    probs = softmax(logits)

    result: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "experiment_name": cfg.experiment_name,
        "preset": ckpt.get("preset", ""),
        "split": split,
        "seed": ckpt.get("seed"),
        "manifest_hash": mh,
        "feature_hash": ckpt.get("feature_hash"),
        "best_val_metric": ckpt.get("best_metric"),
        "best_epoch": ckpt.get("best_epoch"),
        "metrics_uncalibrated": compute_metrics(labels, probs, labels_names, cfg.evaluation.ece_bins),
        "macro_f1_ci95": bootstrap_ci(labels, probs, "macro_f1", cfg.evaluation.n_bootstrap, seed=cfg.training.seed),
        "params_total": count_parameters(model),
        "params_trainable_head": count_parameters(getattr(model, "fusion_model", model)),
        "latency_ms_per_clip_fusion_only": measure_latency(
            model, test_ds, collate, dev, cfg.evaluation.latency_warmup, cfg.evaluation.latency_iters
        ),
        "peak_gpu_memory_mb": peak,
        "device": str(dev),
        "warnings": warnings,
    }

    final_probs = probs
    if cfg.evaluation.calibrate:
        val_ds, _ = _dataset(cfg, filter_split(rows, "val"), model)
        if len(val_ds):
            vl, vlab, _ = collect_logits(model, DataLoader(val_ds, batch_size=cfg.training.batch_size, collate_fn=collate), dev)
            temp = fit_temperature(vl, vlab)
            at_bound = temp <= TEMP_BOUNDS[0] * 1.01 or temp >= TEMP_BOUNDS[1] * 0.99
            if at_bound:
                warnings.append(
                    f"Temperature hit its bound (T={temp:.3f}); validation logits are (near-)separable, so the "
                    "calibration is degenerate and probabilities will be extreme. Treat them as uncalibrated."
                )
            final_probs = softmax(logits, temp)
            result["temperature"] = temp
            result["temperature_at_bound"] = at_bound
            result["calibration_fit_split"] = "val"
            result["metrics_calibrated"] = compute_metrics(labels, final_probs, labels_names, cfg.evaluation.ece_bins)
            if write_calibration:
                save_json(
                    {"temperature": temp, "at_bound": at_bound, "fit_split": "val", "n_val": int(len(vlab)), "manifest_hash": mh},
                    Path(checkpoint).with_name("calibration.json"),
                )
        else:
            warnings.append("No validation samples: calibration skipped.")
    result["metrics"] = result.get("metrics_calibrated", result["metrics_uncalibrated"])

    out = Path(output_dir or cfg.evaluation.output_dir) / cfg.experiment_name
    out.mkdir(parents=True, exist_ok=True)
    result["report_dir"] = str(out)
    save_json(result, out / f"metrics_{split}.json")
    write_predictions(out / f"predictions_{split}.csv", ids, labels, final_probs, labels_names)
    plot_confusion(result["metrics"]["confusion_matrix"], labels_names, out / f"confusion_{split}.png", cfg.experiment_name)
    plot_reliability(final_probs, labels, out / f"reliability_{split}.png", cfg.evaluation.ece_bins, cfg.experiment_name)
    log.info("[%s] %s macro-F1 %.4f acc %.4f -> %s", cfg.experiment_name, split, result["metrics"]["macro_f1"], result["metrics"]["accuracy"], out)
    return result


__all__ = ["evaluate_checkpoint", "load_model", "config_from_checkpoint", "collect_logits", "parse_overrides"]
