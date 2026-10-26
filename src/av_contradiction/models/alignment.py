"""Temporal alignment between audio windows and video frames.

Two mechanisms, both driven by real timestamps (never by array index):

* ``sinusoidal_time_encoding`` - absolute-time features added to projected embeddings.
* ``TemporalProximityBias``   - additive attention bias  b_ij = -(t_i - t_j)^2 / (2 sigma^2),
  i.e. a Gaussian log-prior that makes a query prefer keys close to it in time. sigma is learnable.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def window_centers(audio_ts: torch.Tensor) -> torch.Tensor:
    """(start, end) window times ``[..., 2]`` -> window centres ``[...]``."""
    return audio_ts.mean(-1)


def sinusoidal_time_encoding(ts: torch.Tensor, dim: int, max_period_sec: float = 1000.0, min_period_sec: float = 0.25) -> torch.Tensor:
    """Encode seconds ``[B, T]`` as ``[B, T, dim]`` with geometric periods in [min_period, max_period]."""
    half = dim // 2
    periods = torch.exp(
        torch.linspace(math.log(min_period_sec), math.log(max_period_sec), half, device=ts.device, dtype=torch.float32)
    )
    angles = 2 * math.pi * ts.float().unsqueeze(-1) / periods
    enc = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if dim % 2:
        enc = torch.nn.functional.pad(enc, (0, 1))
    return enc


class TemporalProximityBias(nn.Module):
    def __init__(self, init_sigma_sec: float = 1.0) -> None:
        super().__init__()
        self.log_sigma = nn.Parameter(torch.tensor(math.log(init_sigma_sec)))

    @property
    def sigma(self) -> torch.Tensor:
        return self.log_sigma.exp().clamp(min=1e-2)

    def forward(self, q_ts: torch.Tensor, k_ts: torch.Tensor) -> torch.Tensor:
        """q_ts [B, Tq], k_ts [B, Tk] (seconds) -> bias [B, Tq, Tk]."""
        diff = q_ts.float().unsqueeze(-1) - k_ts.float().unsqueeze(-2)
        return -(diff**2) / (2 * self.sigma**2)

