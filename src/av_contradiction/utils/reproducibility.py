"""Seeding and environment capture for reproducible runs."""

from __future__ import annotations

import importlib.metadata
import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` deriving per-worker seeds from the torch seed."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def package_versions(names: tuple[str, ...] = ("torch", "numpy", "scikit-learn", "transformers", "imageio-ffmpeg")) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions
