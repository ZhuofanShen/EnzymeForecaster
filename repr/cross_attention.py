from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionFusion(nn.Module):
    """
    Fuse LigandMPNN and ESM per-residue embeddings via cross-attention + gated merge.
    Inputs:
      h_lig: [B, N, d]
      h_esm: [B, N, d]  (if ESM length differs, align/resample upstream)
    Outputs:
      h_fused: [B, N, d]
      g_fused: [B, d]   (global token: pooled)
    """
    def __init__(self, d_model: int = 256, n_heads: int = 4):
        super().__init__()
        self.attn_l2e = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.attn_e2l = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid()
        )
        self.out_ln = nn.LayerNorm(d_model)

    def forward(self, h_lig: torch.Tensor, h_esm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q1, k1, v1 = h_lig, h_esm, h_esm
        q2, k2, v2 = h_esm, h_lig, h_lig
        x1, _ = self.attn_l2e(q1, k1, v1)
        x2, _ = self.attn_e2l(q2, k2, v2)

        # gated merge at residue level
        concat = torch.cat([x1, x2], dim=-1)
        g = self.gate(concat)                          # [B, N, d]
        fused = g * x1 + (1.0 - g) * x2
        fused = self.out_ln(fused)

        # global pooled token
        g_fused = fused.mean(dim=1)
        return fused, g_fused
