from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class StructuralEncoder(nn.Module):
    """
    Lightweight, interface-compatible protein-substrate complex structural encoder.
    Produces residue embeddings (and optional pair embeddings) conditioned on ligand.

    Inputs:
      backbone: [B, N, 4, 3] or [B, N, 3]
      ligand:   [B, M, 3] or None
    Outputs:
      h_lig: [B, N, d_model]
      pair  : Optional [B, N, N, d_pair]
    """
    def __init__(self, d_model: int = 256, d_pair: int = 64, use_pairs: bool = True):
        super().__init__()
        self.d_model = d_model
        self.use_pairs = use_pairs
        self.ca_proj = nn.Linear(4 * 3, d_model)  # flatten N,CA,C,O coords
        self.dist_rbf = nn.Linear(16, d_model)
        self.pair_proj = nn.Linear(32, d_pair) if use_pairs else None
        self.out_ln = nn.LayerNorm(d_model)

    @staticmethod
    def _to_ca_frame(backbone: torch.Tensor) -> torch.Tensor:
        # Accept [B,N,4,3] or [B,N,3]; return [B,N,12] flattened (N,CA,C,O)
        if backbone.dim() == 4 and backbone.size(2) == 4:
            return backbone.reshape(backbone.size(0), backbone.size(1), -1)
        elif backbone.dim() == 3 and backbone.size(2) == 3:
            # tile to mimic N,CA,C,O if only CA provided
            x = backbone
            return torch.cat([x, x, x, x], dim=2).reshape(x.size(0), x.size(1), -1)
        else:
            raise ValueError("Backbone must be [B,N,4,3] or [B,N,3].")

    @staticmethod
    def _rbf(d: torch.Tensor, K: int = 16, cutoff: float = 16.0) -> torch.Tensor:
        centers = torch.linspace(0, cutoff, K, device=d.device)
        widths = (cutoff / K) * torch.ones_like(centers)
        return torch.exp(-((d[..., None] - centers) ** 2) / (2 * widths ** 2))

    def forward(
        self,
        backbone: torch.Tensor,
        ligand: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N = backbone.size(0), backbone.size(1)
        ca_feat = self._to_ca_frame(backbone)  # [B,N,12]
        h = self.ca_proj(ca_feat)  # [B,N,d_model]

        if ligand is not None:
            # dist to nearest ligand atom
            ca = backbone[..., 1, :] if backbone.dim() == 4 else backbone  # use CA or provided coords
            # ca: [B,N,3], ligand: [B,M,3]
            dists = torch.cdist(ca, ligand)  # [B,N,M]
            min_d = dists.min(dim=-1).values  # [B,N]
            rbf = self._rbf(min_d)           # [B,N,16]
            h = h + self.dist_rbf(rbf)       # ligand-aware bias

        h = self.out_ln(F.relu(h))

        pair = None
        if self.use_pairs:
            # simple geometric pair features
            ca = backbone[..., 1, :] if backbone.dim() == 4 else backbone
            dij = torch.cdist(ca, ca)  # [B,N,N]
            rbf_ij = self._rbf(dij)    # [B,N,N,16]
            # orientation-ish proxy (normalized vectors aggregated)
            v = ca[:, :, None, :] - ca[:, None, :, :]  # [B,N,N,3]
            v = F.normalize(v + 1e-8, dim=-1)
            pair_feat = torch.cat([rbf_ij, v, -v], dim=-1)  # [B,N,N,16+3+3=22]
            # pad to 32 features
            pad = torch.zeros(B, N, N, 10, device=backbone.device)
            pair = self.pair_proj(torch.cat([pair_feat, pad], dim=-1))  # [B,N,N,d_pair]

        return h, pair
