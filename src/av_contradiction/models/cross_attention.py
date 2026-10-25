"""Bidirectional audio<->video cross-attention.

Shapes (B batch, Ta audio windows, Tv video frames, D hidden):

    audio  a: [B, Ta, D]   mask_a: [B, Ta] (True = valid)
    video  v: [B, Tv, D]   mask_v: [B, Tv]

Per layer, computed simultaneously from the previous layer's states:

    a' = a + Attn(Q = LN(a), K = V = LN([null_v; v]))   # audio queries attend to video
    v' = v + Attn(Q = LN(v), K = V = LN([null_a; a]))   # video queries attend to audio
    x'' = x' + FFN(LN(x'))

A learned *null* key/value is prepended to each context so that (i) a query always has at least
one valid key, avoiding NaN softmax rows when a modality is missing or fully padded, and
(ii) the model can explicitly express "no matching evidence in the other modality".
Padded keys get -inf before the softmax; the temporal-proximity bias (alignment.py) is added
to real keys only (the null token is time-independent).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.h, self.dh = num_heads, dim // num_heads
        self.q, self.k, self.v, self.o = (nn.Linear(dim, dim) for _ in range(4))
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        ctx: torch.Tensor,
        key_mask: torch.Tensor,
        bias: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """x [B,Tq,D], ctx [B,Tk,D], key_mask [B,Tk] bool, bias [B,Tq,Tk] -> ([B,Tq,D], weights [B,Tq,Tk] | None)."""
        b, tq, d = x.shape
        tk = ctx.shape[1]
        q = self.q(x).view(b, tq, self.h, self.dh).transpose(1, 2)
        k = self.k(ctx).view(b, tk, self.h, self.dh).transpose(1, 2)
        v = self.v(ctx).view(b, tk, self.h, self.dh).transpose(1, 2)
        mask = torch.zeros(b, 1, tq, tk, device=x.device, dtype=q.dtype)
        if bias is not None:
            mask = mask + bias.unsqueeze(1).to(q.dtype)
        mask = mask.masked_fill(~key_mask[:, None, None, :], float("-inf"))
        if need_weights:
            scores = (q @ k.transpose(-1, -2)) / self.dh**0.5 + mask
            attn = scores.softmax(-1)
            weights = attn.mean(1)
            out = F.dropout(attn, self.dropout, self.training) @ v
        else:
            weights = None
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0)
        return self.o(out.transpose(1, 2).reshape(b, tq, d)), weights


class CrossAttentionLayer(nn.Module):
    """Pre-LN residual cross-attention followed by a residual feed-forward block."""

    def __init__(self, dim: int, num_heads: int, ff_mult: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm_q, self.norm_kv, self.norm_ff = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = MultiHeadCrossAttention(dim, num_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * ff_mult, dim)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, ctx, key_mask, bias=None, need_weights=False):
        h, w = self.attn(self.norm_q(x), self.norm_kv(ctx), key_mask, bias, need_weights)
        x = x + self.drop(h)
        x = x + self.drop(self.ff(self.norm_ff(x)))
        return x, w


class BidirectionalCrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_layers: int, ff_mult: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.a_from_v = nn.ModuleList(CrossAttentionLayer(dim, num_heads, ff_mult, dropout) for _ in range(num_layers))
        self.v_from_a = nn.ModuleList(CrossAttentionLayer(dim, num_heads, ff_mult, dropout) for _ in range(num_layers))
        self.null_a = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.null_v = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.out_norm_a, self.out_norm_v = nn.LayerNorm(dim), nn.LayerNorm(dim)

    @staticmethod
    def _with_null(x: torch.Tensor, mask: torch.Tensor, null: torch.Tensor, bias: torch.Tensor | None):
        b = x.shape[0]
        ctx = torch.cat([null.expand(b, -1, -1).to(x.dtype), x], dim=1)
        m = torch.cat([mask.new_ones(b, 1), mask], dim=1)
        if bias is not None:
            bias = F.pad(bias, (1, 0), value=0.0)
        return ctx, m, bias

    def forward(
        self,
        a: torch.Tensor,
        mask_a: torch.Tensor,
        v: torch.Tensor,
        mask_v: torch.Tensor,
        bias_av: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> dict[str, object]:
        """bias_av [B, Ta, Tv] (audio-query x video-key); its transpose is used for the other direction.

        Returns {"audio": [B,Ta,D], "video": [B,Tv,D], "attn_a2v": list of [B,Ta,1+Tv], "attn_v2a": list of [B,Tv,1+Ta]}.
        Column 0 of each attention map is the null token.
        """
        bias_va = bias_av.transpose(1, 2) if bias_av is not None else None
        maps_av, maps_va = [], []
        for lay_av, lay_va in zip(self.a_from_v, self.v_from_a, strict=True):
            ctx_v, m_v, b_av = self._with_null(v, mask_v, self.null_v, bias_av)
            ctx_a, m_a, b_va = self._with_null(a, mask_a, self.null_a, bias_va)
            a_new, w_av = lay_av(a, ctx_v, m_v, b_av, need_weights)
            v_new, w_va = lay_va(v, ctx_a, m_a, b_va, need_weights)
            a, v = a_new, v_new
            if need_weights:
                maps_av.append(w_av)
                maps_va.append(w_va)
        return {"audio": self.out_norm_a(a), "video": self.out_norm_v(v), "attn_a2v": maps_av, "attn_v2a": maps_va}
