"""Loss construction with data-driven class weighting."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def compute_class_weights(labels: np.ndarray, num_classes: int, mode: str = "auto", threshold: float = 1.5) -> torch.Tensor | None:
    """Inverse-frequency weights normalised to mean 1.

    mode="auto" applies them only when majority/minority count ratio exceeds ``threshold``
    (so balanced data is trained with plain cross-entropy); "balanced" always applies; "none" never.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    if mode == "none" or (counts == 0).any():
        return None
    if mode == "auto" and counts.max() / counts.min() <= threshold:
        return None
    w = counts.sum() / (num_classes * counts)
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def build_loss(class_weights: torch.Tensor | None = None, label_smoothing: float = 0.0) -> nn.Module:
    return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
