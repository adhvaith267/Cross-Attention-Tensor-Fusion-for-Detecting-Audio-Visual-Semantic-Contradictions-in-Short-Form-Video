"""Hardware detection and device selection."""

from __future__ import annotations

import platform
from typing import Any

import torch


def select_device(preference: str = "auto") -> torch.device:
    """Return the requested device, falling back to CPU when CUDA is unavailable."""
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preference.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Device '{preference}' requested but CUDA is not available")
    return torch.device(preference)


def gpu_memory_gb(device_index: int = 0) -> float | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_properties(device_index).total_memory / 1024**3


def recommend_profile() -> str:
    """Pick a hardware profile from detected VRAM. CPU-only machines get ``low_memory``."""
    mem = gpu_memory_gb()
    if mem is None or mem < 10:
        return "low_memory"
    if mem < 24:
        return "standard_gpu"
    return "experimental"


def amp_supported(device: torch.device) -> bool:
    return device.type == "cuda"


def environment_report() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpus": [],
        "recommended_profile": recommend_profile(),
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["gpus"].append({"index": i, "name": props.name, "memory_gb": round(props.total_memory / 1024**3, 2)})
        info["bf16_supported"] = torch.cuda.is_bf16_supported()
    return info


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / 1024**2
