"""PhysioFM-S: structured-patch, decoder-only, predictive-coding transformer.

The proposal's Phase 2: replace TimesFM's univariate scalar patch with a
structured ``(C x B)`` DE patch, keep the decoder-only causal attention, and
predict the next DE window(s) (the PC-SSL objective inside a transformer).

Two variants (two co-primary experiments):
  * ``scratch``  : a right-sized transformer trained from random init.
  * ``timesfm``  : the *actual* TimesFM-2.5 decoder stack (pretrained weights),
                   with fresh structured input/output blocks. Tests whether
                   TimesFM's temporal priors transfer to structured DE.

Crucially we do NOT use TimesFM's ``model.forward`` (which applies per-series
RevIN instance normalization — the Phase 1 signal killer). We feed our own
structured-patch embeddings straight into the decoder layers, so absolute
band-power structure is preserved. Input normalization is a fixed per-(channel,
band) standardization fit on the pretraining corpus (the same kind StandardScaler
applies in the raw-DE ceiling), not per-series-over-time instance norm.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from transformers import TimesFm2_5Config
from transformers.masking_utils import create_causal_mask
from transformers.models.timesfm2_5.modeling_timesfm2_5 import (
    TimesFm2_5DecoderLayer,
    TimesFm2_5RMSNorm,
    TimesFm2_5RotaryEmbedding,
)

MODEL_ID = "google/timesfm-2.5-200m-transformers"


class _BandAttention(nn.Module):
    """Squeeze-excitation band attention (PC-SSL). x: (N, C, B)."""

    def __init__(self, num_bands: int = 5):
        super().__init__()
        self.fc1 = nn.Linear(num_bands * 2, max(num_bands // 2, 1))
        self.fc2 = nn.Linear(max(num_bands // 2, 1), num_bands)

    def forward(self, x):
        import torch.nn.functional as F

        context = torch.cat([x.mean(dim=1), x.max(dim=1).values], dim=-1)
        w = torch.sigmoid(self.fc2(F.relu(self.fc1(context))))
        return x * w.unsqueeze(1)


class _ChannelAttention(nn.Module):
    """Squeeze-excitation channel attention (PC-SSL). x: (N, C, B)."""

    def __init__(self, num_channels: int = 62):
        super().__init__()
        self.fc1 = nn.Linear(num_channels * 2, max(num_channels // 2, 1))
        self.fc2 = nn.Linear(max(num_channels // 2, 1), num_channels)

    def forward(self, x):
        import torch.nn.functional as F

        context = torch.cat([x.mean(dim=2), x.max(dim=2).values], dim=-1)
        w = torch.sigmoid(self.fc2(F.relu(self.fc1(context))))
        return x * w.unsqueeze(2)


class CBPatchEmbed(nn.Module):
    """Channel/band-attention structured patch embedder."""

    def __init__(self, n_ch: int, n_band: int, p_in: int, d: int):
        super().__init__()
        self.n_ch, self.n_band, self.p_in = n_ch, n_band, p_in
        self.band_attn = _BandAttention(n_band)
        self.chan_attn = _ChannelAttention(n_ch)
        self.proj = nn.Linear(p_in * n_ch * n_band, d)
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, patches):  # (B, P, p_in*n_ch*n_band)
        b, p, _ = patches.shape
        x = patches.reshape(b * p * self.p_in, self.n_ch, self.n_band)
        x = self.chan_attn(self.band_attn(x))
        x = x.reshape(b, p, self.p_in * self.n_ch * self.n_band)
        return self.proj(x)


def small_config(hidden: int = 256, layers: int = 6, heads: int = 8) -> TimesFm2_5Config:
    return TimesFm2_5Config(
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=heads,
        head_dim=hidden // heads,
        intermediate_size=hidden,
        attention_bias=False,
        use_bias=False,
        activation="swish",
        rms_norm_eps=1e-6,
        max_position_embeddings=4096,
        attention_dropout=0.0,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
    )


class PhysioFMS(nn.Module):
    def __init__(
        self,
        n_cb: int = 310,
        p_in: int = 1,
        p_out: int = 16,
        variant: str = "scratch",
        hidden: int = 256,
        layers: int = 6,
        heads: int = 8,
        embedder: str = "linear",
        n_ch: int = 62,
        n_band: int = 5,
    ) -> None:
        super().__init__()
        self.n_cb = n_cb
        self.p_in = p_in
        self.p_out = p_out
        self.variant = variant
        self.embedder = embedder

        if variant == "timesfm":
            from transformers import TimesFm2_5ModelForPrediction

            base = TimesFm2_5ModelForPrediction.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
            self.config = base.config
            self.layers = base.model.layers
            self.rotary_emb = base.model.rotary_emb
            d = self.config.hidden_size
        elif variant == "timesfm_rand":
            # F3: a stack with TimesFM-2.5's EXACT shape (d, layers, heads,
            # intermediate_size, head_dim) but RANDOM weights — to disentangle
            # "TimesFM priors transfer" from "a big fixed nonlinear mixer".
            cfg = TimesFm2_5Config.from_pretrained(MODEL_ID)
            self.config = cfg
            self.layers = nn.ModuleList(
                [TimesFm2_5DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)]
            )
            self.rotary_emb = TimesFm2_5RotaryEmbedding(cfg)
            d = cfg.hidden_size
            self._init_scratch()
        elif variant == "scratch":
            cfg = small_config(hidden, layers, heads)
            self.config = cfg
            self.layers = nn.ModuleList(
                [TimesFm2_5DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)]
            )
            self.rotary_emb = TimesFm2_5RotaryEmbedding(cfg)
            d = cfg.hidden_size
            self._init_scratch()
        else:
            raise ValueError(f"unknown variant {variant!r}")

        self.config._attn_implementation = "eager"
        self.d = d
        if embedder == "attn":
            self.patch_in = CBPatchEmbed(n_ch, n_band, p_in, d)
        else:
            self.patch_in = nn.Linear(p_in * n_cb, d, bias=True)
            nn.init.normal_(self.patch_in.weight, std=0.02)
            nn.init.zeros_(self.patch_in.bias)
        self.out_norm = TimesFm2_5RMSNorm(d, eps=self.config.rms_norm_eps)
        self.head = nn.Linear(d, p_out * n_cb, bias=True)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def _init_scratch(self) -> None:
        for layer in self.layers:
            for m in layer.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            # softplus(0)*1.442695 = 1/sqrt(head_dim) -> standard attention scaling
            nn.init.zeros_(layer.self_attn.scaling)

    def freeze_backbone(self) -> None:
        for p in self.layers.parameters():
            p.requires_grad_(False)

    def encode(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, T, n_cb) standardized -> hidden (B, P, d), P = T // p_in."""
        b, t, _ = x.shape
        p = t // self.p_in
        patches = x[:, : p * self.p_in].reshape(b, p, self.p_in * self.n_cb)
        emb = self.patch_in(patches)

        if pad_mask is None:
            patch_valid = torch.ones(b, p, dtype=torch.long, device=x.device)
        else:
            pv = pad_mask[:, : p * self.p_in].reshape(b, p, self.p_in)
            patch_valid = (pv.sum(-1) > 0).long()

        position_ids = torch.arange(p, device=x.device).unsqueeze(0).expand(b, -1)
        attn_mask = create_causal_mask(self.config, emb, patch_valid, past_key_values=None)
        pos_emb = self.rotary_emb(emb, position_ids)

        h = emb
        for layer in self.layers:
            h = layer(
                h,
                position_embeddings=pos_emb,
                attention_mask=attn_mask,
                position_ids=position_ids,
            )
        return self.out_norm(h)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Predictive-coding forward: returns predicted future windows.

        Output (B, P, p_out, n_cb): at patch position j, predicts the next
        ``p_out`` DE windows after the windows that patch covers.
        """
        h = self.encode(x, pad_mask)
        b, p, _ = h.shape
        return self.head(h).reshape(b, p, self.p_out, self.n_cb)
