"""Sequence-context classification head (EXP-0027).

Every published sleep-staging SOTA model classifies an epoch from a WINDOW of
neighbouring epoch features (SeqSleepNet/SleepTransformer/SleePyCo: 10-35 epochs,
bidirectional). Our harnesses so far used a position-wise linear head — deliberately
minimal, but it concedes that context. This module adds the missing sequence stage
on TOP of the (frozen-or-finetuned) causal encoder's per-epoch features:

  ContextHead: sinusoidal PE -> N-layer bidirectional TransformerEncoder
               (with key-padding mask) -> linear logits.

`lookahead` bounds how far attention may look into the future:
  None  = unrestricted bidirectional (offline operating point);
  k >= 0 = token i attends to j <= i+k (k=0 keeps the head causal, so the streaming
           claim survives; intermediate k gives a latency-accuracy curve).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _sinusoidal_pe(max_len: int, d: int) -> torch.Tensor:
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe = torch.zeros(max_len, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class ContextHead(nn.Module):
    def __init__(self, d: int, n_classes: int, layers: int = 2, heads: int = 8,
                 ffn: int = 512, dropout: float = 0.1, lookahead: int | None = None,
                 window: int | None = None, max_len: int = 4096):
        super().__init__()
        self.lookahead = lookahead
        self.window = window          # symmetric band |i-j| <= window (ladder-matched context)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=ffn, dropout=dropout,
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.out = nn.Linear(d, n_classes)
        self.register_buffer("pe", _sinusoidal_pe(max_len, d), persistent=False)

    def _attn_mask(self, p: int, device) -> torch.Tensor | None:
        if self.lookahead is None and self.window is None:
            return None
        i = torch.arange(p, device=device)
        # True = MASKED
        m = torch.zeros(p, p, dtype=torch.bool, device=device)
        if self.lookahead is not None:      # token i may attend to j <= i + lookahead
            m |= i.unsqueeze(0) > (i.unsqueeze(1) + self.lookahead)
        if self.window is not None:         # |i - j| <= window
            m |= (i.unsqueeze(0) - i.unsqueeze(1)).abs() > self.window
        return m

    def forward(self, h: torch.Tensor, lengths=None) -> torch.Tensor:
        """h: (B, P, d) per-epoch features; lengths: list/tensor of valid lengths."""
        b, p, d = h.shape
        x = h + self.pe[:p].to(h.dtype)
        pad = None
        if lengths is not None:
            ar = torch.arange(p, device=h.device)
            lt = torch.as_tensor(list(lengths), device=h.device)
            pad = ar.unsqueeze(0) >= lt.unsqueeze(1)          # True = padding
        x = self.encoder(x, mask=self._attn_mask(p, h.device), src_key_padding_mask=pad)
        return self.out(x)


def build_head(kind: str, d: int, n_classes: int, lookahead: int | None = None,
               window: int | None = None) -> nn.Module:
    if kind == "linear":
        return nn.Linear(d, n_classes)
    if kind == "context":
        return ContextHead(d, n_classes, lookahead=lookahead, window=window)
    raise ValueError(kind)
