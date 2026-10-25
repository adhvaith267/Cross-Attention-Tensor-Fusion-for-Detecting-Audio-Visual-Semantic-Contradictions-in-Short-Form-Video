"""Classification, ranking and calibration metrics.

Positive class for ROC/PR-AUC is ``contradictory`` (id 1).
ECE: equal-width confidence bins over max-probability; ECE = sum_b (n_b / N) |acc_b - conf_b|.
Brier: mean squared error of the contradiction probability against the 0/1 label.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    conf = probs.max(1)
    correct = (probs.argmax(1) == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        sel = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if sel.any():
            ece += sel.mean() * abs(correct[sel].mean() - conf[sel].mean())
    return float(ece)


def compute_metrics(labels: np.ndarray, probs: np.ndarray, label_names: list[str] | tuple[str, ...], ece_bins: int = 10) -> dict[str, Any]:
    labels = np.asarray(labels)
    preds = probs.argmax(1)
    k = len(label_names)
    p, r, f, s = precision_recall_fscore_support(labels, preds, labels=list(range(k)), zero_division=0)
    out: dict[str, Any] = {
        "n": int(len(labels)),
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_precision": float(p.mean()),
        "macro_recall": float(r.mean()),
        "macro_f1": float(f1_score(labels, preds, labels=list(range(k)), average="macro", zero_division=0)),
        "per_class": {
            name: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i, name in enumerate(label_names)
        },
        "confusion_matrix": confusion_matrix(labels, preds, labels=list(range(k))).tolist(),
        "brier": float(np.mean((probs[:, 1] - (labels == 1)) ** 2)) if k == 2 else None,
        "ece": expected_calibration_error(probs, labels, ece_bins),
    }
    if k == 2 and len(np.unique(labels)) == 2:
        out["roc_auc"] = float(roc_auc_score(labels, probs[:, 1]))
        out["pr_auc"] = float(average_precision_score(labels, probs[:, 1]))
    else:
        out["roc_auc"] = out["pr_auc"] = None  # undefined with a single class present
    return out


def bootstrap_ci(
    labels: np.ndarray, probs: np.ndarray, metric: str = "macro_f1", n: int = 1000, seed: int = 0, alpha: float = 0.05
) -> dict[str, float] | None:
    """Percentile bootstrap CI over test samples (resampling with replacement)."""
    if n <= 0 or len(labels) < 2:
        return None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(labels), len(labels))
        preds = probs[idx].argmax(1)
        if metric == "macro_f1":
            vals.append(f1_score(labels[idx], preds, average="macro", zero_division=0))
        elif metric == "accuracy":
            vals.append(float((preds == labels[idx]).mean()))
        else:
            raise ValueError(metric)
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return {"low": float(lo), "high": float(hi), "level": 1 - alpha, "n_boot": n}


TEMP_BOUNDS = (0.05, 20.0)


def fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iter: int = 200) -> float:
    """Temperature scaling (Guo et al., 2017): minimise validation NLL over a single scalar T > 0."""
    lg = torch.tensor(logits, dtype=torch.float64)
    lb = torch.tensor(labels, dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(lg / log_t.exp(), lb)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp().clamp(*TEMP_BOUNDS))
