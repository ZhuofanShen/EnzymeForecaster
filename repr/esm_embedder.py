from __future__ import annotations
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn

try:
    import esm  # fair-esm
    _ESM_AVAILABLE = True
except Exception:
    _ESM_AVAILABLE = False


class ESMEmbedder(nn.Module):
    """
    Wrapper over fair-esm ESM-2 models that returns:
      - per-residue embeddings (excluding special tokens)
      - pooled embedding (mean over residues, excluding specials)
      - aux dict with lengths and masks

    Outputs are projected to 'out_dim' so they plug directly into your fusion block.

    Args:
      model_name: one of esm2_* from fair-esm (e.g., "esm2_t33_650M_UR50D", "esm2_t6_8M_UR50D")
      repr_layer: which layer to extract (None => last layer)
      out_dim: output dimension to project to (if None, use ESM hidden size)
      device: "cuda" | "cpu" (default: auto from torch.cuda.is_available())
      fp16: cast model & inputs to float16 (CUDA only) to save memory
      finetune: if False -> model.eval() and freeze params; if True -> trainable
    """
    def __init__(
        self,
        model_name: str = "esm2_t33_650M_UR50D",
        repr_layer: Optional[int] = None,
        out_dim: Optional[int] = None,
        device: Optional[str] = None,
        fp16: bool = False,
        finetune: bool = False,
        # Back-compat with the earlier skeleton: allow d_model to alias out_dim
        d_model: Optional[int] = None,
    ):
        super().__init__()
        if not _ESM_AVAILABLE:
            raise ImportError(
                "fair-esm is not installed. Please 'pip install fair-esm torch' to use ESMEmbedder."
            )

        self.model_name = model_name
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.fp16 = bool(fp16)
        self.finetune = bool(finetune)

        # Load model & alphabet from fair-esm
        # Prefer esm.pretrained.<name>() if available; fallback to generic loader
        if hasattr(esm.pretrained, model_name):
            self.model, self.alphabet = getattr(esm.pretrained, model_name)()
        else:
            # generic loader works with HF-style names too
            self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)

        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.to(self.device)

        # Determine hidden size and default repr layer (last)
        # ESM-2 models expose 'embed_dim' and 'num_layers'
        self.hidden_dim = getattr(self.model, "embed_dim", None)
        self.num_layers = getattr(self.model, "num_layers", None) or \
                          (len(getattr(self.model, "layers", [])) or None)
        if self.hidden_dim is None:
            raise RuntimeError("Could not infer ESM hidden size (embed_dim missing).")
        if repr_layer is None:
            # last layer index for esm2_t{L}_... is 'L'
            repr_layer = int(self.num_layers) if self.num_layers is not None else 33
        self.repr_layer = int(repr_layer)

        # Project to out_dim (== d_model for fusion) if provided
        if out_dim is None and d_model is not None:
            out_dim = d_model
        self.out_dim = out_dim or self.hidden_dim
        self.proj = nn.Identity() if self.out_dim == self.hidden_dim else nn.Linear(self.hidden_dim, self.out_dim)

        # Casting / (un)freezing
        if not self.finetune:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False
        if self.fp16 and self.device.type == "cuda":
            self.model.half()
            # Note: we keep the projection in fp32 unless you also want it in fp16
            # self.proj.half()

    @torch.no_grad()
    def _encode_batch_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Run ESM forward to obtain token-level representations.
        tokens: [B, L] on correct device
        returns: tok_repr [B, L, H] (includes BOS/EOS positions)
        """
        if self.fp16 and self.device.type == "cuda":
            tokens = tokens.half()
        out = self.model(
            tokens.to(self.device),
            repr_layers=[self.repr_layer],
            return_contacts=False
        )
        tok_repr = out["representations"][self.repr_layer]  # [B, L, H]
        return tok_repr

    def forward(self, seqs: List[str]) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Args:
          seqs: list of strings (all should have the SAME length N for downstream fusion)

        Returns:
          per_res: [B, N, out_dim]  (excluding BOS/EOS; zero-padded if lengths differ)
          pooled : [B, out_dim]     (mean over valid residues per sequence)
          aux    : {"lengths": LongTensor[B], "mask": BoolTensor[B,N]}
        """
        # Build ESM batch
        data = [(f"seq{i}", s) for i, s in enumerate(seqs)]
        labels, strs, tokens = self.batch_converter(data)  # tokens on CPU by default
        tokens = tokens.to(self.device)
        # length including specials
        lengths_with_specials = (tokens != self.alphabet.padding_idx).sum(dim=1)  # [B]

        # Run model to get token reps
        # If finetuning: allow grad; else: no_grad from caller (we keep this safe: torch.no_grad() in _encode_batch_tokens)
        tok_repr = self._encode_batch_tokens(tokens)  # [B, L, H]

        B, L, H = tok_repr.shape
        # Exclude BOS (index 0) and EOS (index lengths-1) for each sequence
        per_res_list: List[torch.Tensor] = []
        valid_lengths: List[int] = []
        for b in range(B):
            Lb = int(lengths_with_specials[b].item())  # includes BOS/EOS
            # valid residues are [1 : Lb-1]
            if Lb <= 2:
                # degenerate case; create a single zero row
                x = tok_repr.new_zeros(1, H)
                valid_len = 1
            else:
                x = tok_repr[b, 1:Lb-1, :]  # [N_b, H]
                valid_len = x.size(0)
            per_res_list.append(x)
            valid_lengths.append(valid_len)

        # Check equal lengths (recommended for your fusion). If not equal, pad to max.
        Nmax = max(valid_lengths)
        per_res = tok_repr.new_zeros(B, Nmax, H)
        mask = torch.zeros(B, Nmax, dtype=torch.bool, device=per_res.device)
        for b, x in enumerate(per_res_list):
            n = x.size(0)
            per_res[b, :n, :] = x
            mask[b, :n] = True

        # Project to out_dim
        per_res = self.proj(per_res)  # [B, Nmax, out_dim]

        # Pooled (mean over valid residues only)
        denom = mask.sum(dim=1).clamp_min(1).unsqueeze(-1)  # [B,1]
        pooled = (per_res * mask.unsqueeze(-1)).sum(dim=1) / denom  # [B, out_dim]

        aux = {
            "lengths": torch.tensor(valid_lengths, device=per_res.device, dtype=torch.long),
            "mask": mask  # True for valid positions
        }
        return per_res, pooled, aux
