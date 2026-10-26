"""Training loop on cached features: AMP, gradient clipping, warmup + cosine schedule, early stopping
on validation macro-F1, best/last checkpoints, resume, NaN guards and graceful interruption."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from av_contradiction.config import Config, save_config
from av_contradiction.data.dataset import (
    CachedFeatureDataset,
    collate_features,
    feature_statistics,
    move_batch,
)
from av_contradiction.data.features import cache_dir_for
from av_contradiction.data.manifest import filter_split, manifest_hash, read_manifest
from av_contradiction.models.classifier import AVContradictionModel, count_parameters
from av_contradiction.training.checkpointing import load_checkpoint, save_checkpoint
from av_contradiction.training.losses import build_loss, compute_class_weights
from av_contradiction.training.metrics import compute_metrics, softmax
from av_contradiction.utils.device import amp_supported, select_device
from av_contradiction.utils.logging import get_logger
from av_contradiction.utils.reproducibility import package_versions, seed_worker, set_seed

log = get_logger(__name__)


class NonFiniteLossError(RuntimeError):
    pass


@dataclass
class TrainResult:
    best_metric: float | None  # best validation macro-F1
    best_epoch: int | None
    best_checkpoint: str | None
    last_checkpoint: str | None
    epochs_run: int
    global_step: int
    stopped_reason: str
    history: list[dict[str, Any]] = field(default_factory=list)


def warmup_cosine(total_steps: int, warmup_steps: int):
    def fn(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return 0.5 * (1 + math.cos(math.pi * progress))

    return fn


class Trainer:
    def __init__(
        self,
        cfg: Config,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        run_dir: str | Path,
        meta: dict[str, Any],
        class_weights: torch.Tensor | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.cfg, self.tc = cfg, cfg.training
        self.device = device or select_device(self.tc.device)
        self.model = model.to(self.device)
        self.train_loader, self.val_loader = train_loader, val_loader
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.meta = meta
        self.criterion = build_loss(class_weights.to(self.device) if class_weights is not None else None)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.tc.lr, weight_decay=self.tc.weight_decay)
        total_steps = max(1, len(train_loader)) * self.tc.epochs
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, warmup_cosine(total_steps, int(total_steps * self.tc.warmup_ratio))
        )
        self.use_amp = self.tc.amp and amp_supported(self.device)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.epoch = 0
        self.global_step = 0
        self.best_metric: float | None = None
        self.best_epoch: int | None = None
        self.patience = 0
        self.nan_steps = 0
        self.history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ checkpoints
    def state(self) -> dict[str, Any]:
        state = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "patience": self.patience,
            "history": self.history,
        }
        state.update(self.meta)
        return state

    def load_state(self, path: str | Path) -> None:
        ckpt = load_checkpoint(path, self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.scaler.load_state_dict(ckpt["scaler_state"])
        self.epoch, self.global_step = ckpt["epoch"], ckpt["global_step"]
        self.best_metric, self.best_epoch = ckpt["best_metric"], ckpt["best_epoch"]
        self.patience, self.history = ckpt["patience"], list(ckpt["history"])
        if ckpt.get("manifest_hash") != self.meta.get("manifest_hash"):
            log.warning("Resuming with a different manifest hash than the checkpoint was trained on")
        log.info("Resumed from %s at epoch %d step %d", path, self.epoch, self.global_step)

    # ------------------------------------------------------------------ loops
    def _forward_loss(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
            logits = self.model(batch)["logits"]
        return self.criterion(logits.float(), batch["label"]), logits

    def train_epoch(self) -> float:
        self.model.train()
        losses = []
        for batch in self.train_loader:
            batch = move_batch(batch, self.device)
            loss, _ = self._forward_loss(batch)
            self.optimizer.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                self.nan_steps += 1
                log.warning("Non-finite loss at step %d (%d so far); skipping batch", self.global_step, self.nan_steps)
                if self.nan_steps > self.tc.max_nan_steps:
                    raise NonFiniteLossError(f"More than {self.tc.max_nan_steps} non-finite losses")
                continue
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.tc.grad_clip or float("inf"))
            if self.use_amp or torch.isfinite(norm):
                self.scaler.step(self.optimizer)  # under AMP, GradScaler itself skips steps with inf/nan gradients
            else:
                log.warning("Non-finite gradient norm; skipping optimizer step")
            self.scaler.update()
            self.scheduler.step()
            self.global_step += 1
            losses.append(loss.item())
        return float(np.mean(losses)) if losses else float("nan")

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray, float]:
        self.model.eval()
        all_logits, all_labels, losses = [], [], []
        for batch in loader:
            batch = move_batch(batch, self.device)
            loss, logits = self._forward_loss(batch)
            losses.append(loss.item() * len(batch["label"]))
            all_logits.append(logits.float().cpu().numpy())
            all_labels.append(batch["label"].cpu().numpy())
        labels = np.concatenate(all_labels) if all_labels else np.zeros(0, int)
        logits = np.concatenate(all_logits) if all_logits else np.zeros((0, 2))
        return logits, labels, float(np.sum(losses) / max(1, len(labels)))

    def fit(self, resume_from: str | Path | None = None) -> TrainResult:
        if resume_from:
            self.load_state(resume_from)
        stopped = "completed"
        last_path = self.run_dir / "last.pt"
        try:
            while self.epoch < self.tc.epochs:
                t0 = time.time()
                train_loss = self.train_epoch()
                logits, labels, val_loss = self.predict(self.val_loader)
                val = compute_metrics(labels, softmax(logits), self.cfg.data.label_names)
                self.epoch += 1
                entry = {
                    "epoch": self.epoch, "step": self.global_step, "train_loss": train_loss, "val_loss": val_loss,
                    "val_macro_f1": val["macro_f1"], "val_accuracy": val["accuracy"], "val_roc_auc": val["roc_auc"],
                    "lr": self.scheduler.get_last_lr()[0], "seconds": round(time.time() - t0, 2),
                }
                self.history.append(entry)
                log.info("epoch %d | train_loss %.4f | val_loss %.4f | val_macro_f1 %.4f | val_acc %.4f",
                         self.epoch, train_loss, val_loss, val["macro_f1"], val["accuracy"])
                if self.best_metric is None or val["macro_f1"] > self.best_metric:
                    self.best_metric, self.best_epoch, self.patience = float(val["macro_f1"]), self.epoch, 0
                    save_checkpoint(self.run_dir / "best.pt", self.state())
                else:
                    self.patience += 1
                save_checkpoint(last_path, self.state())
                if self.patience >= self.tc.early_stopping_patience:
                    stopped = "early_stopping"
                    break
        except KeyboardInterrupt:
            stopped = "interrupted"
            log.warning("Interrupted: saving %s", self.run_dir / "interrupted.pt")
            save_checkpoint(self.run_dir / "interrupted.pt", self.state())
        save_checkpoint(last_path, self.state())
        best = self.run_dir / "best.pt"
        return TrainResult(
            self.best_metric, self.best_epoch, str(best) if best.exists() else None, str(last_path),
            self.epoch, self.global_step, stopped, self.history,
        )


# ---------------------------------------------------------------------- high-level entry point
def make_loader(dataset: CachedFeatureDataset, cfg: Config, shuffle: bool) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(cfg.training.seed)
    return DataLoader(
        dataset, batch_size=cfg.training.batch_size, shuffle=shuffle, num_workers=cfg.training.num_workers,
        collate_fn=collate_features, worker_init_fn=seed_worker, generator=g, pin_memory=torch.cuda.is_available(),
    )


def train_from_config(cfg: Config, resume: str | Path | None = None, preset: str | None = None) -> TrainResult:
    """Build datasets/model from ``cfg`` and train. Checkpoints go to <checkpoint_dir>/<experiment_name>/."""
    set_seed(cfg.training.seed)
    device = select_device(cfg.training.device)
    rows = read_manifest(cfg.data.manifest)
    train_rows, val_rows = filter_split(rows, "train"), filter_split(rows, "val")
    if not train_rows or not val_rows:
        raise ValueError("Manifest needs non-empty 'train' and 'val' splits (run python -m av_contradiction.cli.prepare_dataset)")
    train_ds, val_ds = CachedFeatureDataset(cfg, train_rows), CachedFeatureDataset(cfg, val_rows)
    for name, ds in (("train", train_ds), ("val", val_ds)):
        if ds.missing:
            log.warning("%d %s samples have no cached features in %s (skipped)", len(ds.missing), name, cache_dir_for(cfg))
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(f"No cached features found in {cache_dir_for(cfg)}; run python -m av_contradiction.cli.extract_features first")
    sample = train_ds[0]
    audio_dim, video_dim = sample["audio"].shape[-1], sample["video"].shape[-1]
    model = AVContradictionModel(cfg.model, audio_dim, video_dim)
    class_weights = compute_class_weights(
        train_ds.labels, len(cfg.data.label_names), cfg.training.class_weighting, cfg.training.imbalance_threshold
    )
    log.info(
        "train=%d val=%d | class counts %s | class weights %s | params %d | device %s",
        len(train_ds), len(val_ds), np.bincount(train_ds.labels).tolist(),
        None if class_weights is None else class_weights.tolist(), count_parameters(model), device,
    )
    run_dir = Path(cfg.training.checkpoint_dir) / cfg.experiment_name
    meta = {
        "config": cfg.to_dict(),
        "label_names": list(cfg.data.label_names),
        "audio_dim": int(audio_dim),
        "video_dim": int(video_dim),
        "seed": cfg.training.seed,
        "manifest_hash": manifest_hash(cfg.data.manifest),
        "feature_hash": cfg.feature_hash(),
        "versions": package_versions(),
        "feature_stats": feature_statistics(train_ds),
        "preset": preset or "",
        "class_weights": None if class_weights is None else class_weights.tolist(),
    }
    trainer = Trainer(cfg, model, make_loader(train_ds, cfg, True), make_loader(val_ds, cfg, False), run_dir, meta, class_weights, device)
    save_config(cfg, run_dir / "config.yaml")
    result = trainer.fit(resume)
    with open(run_dir / "train_summary.json", "w", encoding="utf-8") as fh:
        json.dump(asdict(result), fh, indent=2)
    log.info("Training finished (%s). Best val macro-F1 %s at epoch %s", result.stopped_reason, result.best_metric, result.best_epoch)
    return result
