r"""Low-rank multimodal tensor fusion (after Liu et al., ACL 2018, "Efficient Low-rank Multimodal
Fusion with Modality-Specific Factors").

Full tensor fusion appends a constant 1 to each of M input vectors z_m \in R^{d_m} and takes the
outer product  Z = \bigotimes_m [z_m; 1],  then maps it linearly to d_out:

    h = W \cdot vec(Z),   W \in R^{d_out \times \prod_m (d_m + 1)}.

The appended 1 keeps every lower-order (unimodal, bimodal, ...) interaction term. With M = 4
inputs of size 128 that is ~2.8e8 * d_out weights, which is infeasible. The low-rank version
factorises W into R rank-1 terms per output unit:

    h = \sum_{r=1}^{R} \lambda_r \bigodot_{m=1}^{M} ( [z_m; 1]^T W_m^{(r)} ) + b,
    W_m^{(r)} \in R^{(d_m + 1) \times d_out}

so the fused vector is never materialised.

Parameters:   R * \sum_m (d_m + 1) * d_out + R + d_out
Compute:      O(B * R * \sum_m (d_m + 1) * d_out)     (vs. O(B * \prod_m (d_m+1) * d_out))

Inputs are LayerNorm-ed first so the M-way elementwise product stays numerically well scaled.
"""

from __future__ import annotations

import torch
from torch import nn


class LowRankTensorFusion(nn.Module):
    def __init__(self, input_dims: list[int], out_dim: int, rank: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        if not input_dims:
            raise ValueError("Need at least one input")
        self.input_dims, self.out_dim, self.rank = list(input_dims), out_dim, rank
        self.norms = nn.ModuleList(nn.LayerNorm(d) for d in input_dims)
        self.factors = nn.ParameterList()
        for d in input_dims:
            w = torch.empty(rank, d + 1, out_dim)
            nn.init.xavier_normal_(w)
            # Bias row (multiplies the appended 1) starts near 1 so the product does not collapse to 0.
            with torch.no_grad():
                w[:, -1, :] = 1.0 + 0.01 * torch.randn(rank, out_dim)
            self.factors.append(nn.Parameter(w))
        self.rank_weights = nn.Parameter(torch.full((rank,), 1.0 / rank))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.out_norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """inputs: list of [B, d_m] tensors in the order given by ``input_dims`` -> [B, out_dim]."""
        if len(inputs) != len(self.factors):
            raise ValueError(f"Expected {len(self.factors)} inputs, got {len(inputs)}")
        fused = None
        for x, norm, w in zip(inputs, self.norms, self.factors, strict=True):
            z = torch.cat([norm(x), x.new_ones(x.shape[0], 1)], dim=1)  # [B, d+1]
            proj = torch.einsum("bd,rdo->rbo", z, w.to(z.dtype))  # [R, B, out]
            fused = proj if fused is None else fused * proj
        out = torch.einsum("r,rbo->bo", self.rank_weights.to(fused.dtype), fused) + self.bias
        return self.dropout(self.out_norm(out))


class ConcatFusion(nn.Module):
    """Baseline: MLP over the concatenation of the inputs."""

    def __init__(self, input_dims: list[int], out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        total = sum(input_dims)
        self.net = nn.Sequential(
            nn.LayerNorm(total), nn.Linear(total, out_dim), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(out_dim)
        )

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        return self.net(torch.cat(inputs, dim=1))
