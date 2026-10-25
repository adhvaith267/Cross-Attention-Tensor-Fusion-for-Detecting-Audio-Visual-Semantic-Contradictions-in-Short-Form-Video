"""Atomic checkpoint save/load."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: str | Path, state: dict[str, Any]) -> Path:
    """Write to a temp file then rename, so an interrupted save never corrupts the previous checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)
    return path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    # Checkpoints contain only tensors and plain Python containers -> safe weights_only loading.
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    for key in ("model_state", "config", "label_names", "audio_dim", "video_dim"):
        if key not in ckpt:
            raise KeyError(f"Checkpoint {path.name} is missing '{key}'")
    return ckpt
