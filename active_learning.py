from __future__ import annotations
from typing import Dict, List, Tuple, Sequence, Optional
import json
import time
import torch
from core.schemas import Variant, StructuralTemplate, CandidateFeatures, Prediction, AcquisitionConfig, AssayResult
from repr.ligand_mpnn_encoder import LigandMPNNEmbedder
from repr.esm_embedder import ESMEmbedder
from fusion.cross_attention import CrossAttentionFusion
from surrogate.multitask_ensemble import MultiTaskEnsemble
from struct.sidechain_diffuse import SidechainDiffusion, sample_sidechains
from struct.local_relax import local_relax
from struct.checks import structural_checks
from acq.qehvi import select_batch_qEHVI
from acq.ucb_scalarized import select_batch_ucb
from candidates.enumerate_from_beneficials import enumerate_combinations

class ActiveLearningLoop:
    def __init__(
        self,
        template: StructuralTemplate,
        d_model: int = 256,
        device: str = "cpu",
        esm_kwargs: Optional[Dict] = None,  # NEW
    ):
        self.device = torch.device(device)
        self.template = template

        # Structural encoder stays as you had it
        self.ligand_encoder = LigandMPNNEncoder(d_model=d_model).to(self.device)

        # Real ESM embedder (projects to d_model). You can override via esm_kwargs.
        esm_kwargs = esm_kwargs or {}
        self.esm = ESMEmbedder(
            d_model=d_model,                        # project to fusion width
            model_name=esm_kwargs.get("model_name", "esm2_t6_8M_UR50D"),
            repr_layer=esm_kwargs.get("repr_layer", None),  # last by default
            device=esm_kwargs.get("device", device),
            fp16=esm_kwargs.get("fp16", False),
            finetune=esm_kwargs.get("finetune", False),
        )

        self.fuser = CrossAttentionFusion(d_model=d_model).to(self.device)
        self.surrogate = MultiTaskEnsemble(d_in=d_model, n_members=5, z_dim=0).to(self.device)
        self.diffuser = SidechainDiffusion(node_dim=d_model, num_atoms=6).to(self.device)
        self.cfg = AcquisitionConfig()

    def _embed_variants(self, variants: Sequence[Variant]) -> torch.Tensor:
        """Return fused global embedding per variant: [B, d]."""
        seqs = [v.sequence for v in variants]
        with torch.no_grad():
            h_esm, g_esm, _ = self.esm(seqs)  # [B,N,d], [B,d]
            B, N = h_esm.size(0), h_esm.size(1)
            bb = self.template.backbone_coords.unsqueeze(0).expand(B, -1, -1, -1).to(self.device)  # [B,N,4,3]
            lig = (self.template.ligand_coords.unsqueeze(0).expand(B, -1, -1).to(self.device)
                   if self.template.ligand_coords is not None else None)
            h_lig, _ = self.ligand_encoder(bb, lig)   # [B,N,d]
            h_fused, g_fused = self.fuser(h_lig, h_esm)   # [B,N,d], [B,d]
        return g_fused  # [B,d]

    def _structural_gate(self, variants: Sequence[Variant], h_res: torch.Tensor) -> List[CandidateFeatures]:
        """
        Place sidechains -> local relax -> compute structural features/feasibility.
        h_res: per-variant global or per-res embeddings (here global used only to condition if needed).
        """
        feats: List[CandidateFeatures] = []
        B = len(variants)
        # Toy: build per-residue conditioning by tiling global embedding
        h_per = h_res.unsqueeze(1).expand(-1, self.template.backbone_coords.size(0), -1)  # [B,N,d]
        with torch.no_grad():
            sc = sample_sidechains(self.diffuser, h_per, num_steps=64)  # [B,N,A,3]
        # Collapse to a neighborhood representation for checks (toy: flatten)
        for b in range(B):
            sc_b = sc[b].reshape(-1, 3)                     # [N*A,3]
            # local relax (optional)
            sc_relaxed = local_relax(sc_b.unsqueeze(0)).squeeze(0)
            lig = self.template.ligand_coords if self.template.ligand_coords is not None else torch.zeros(1,3)
            feat = structural_checks(sc_relaxed, lig, cat_geom={})
            feats.append(feat)
        return feats

    def _predict(self, g: torch.Tensor) -> List[Prediction]:
        """Run surrogate ensemble; here no explicit z (epistasis head) is passed."""
        preds = self.surrogate.predict(g)
        # Map structural feasibility into p_feasible if desired (done later for now).
        return preds

    def propose_batch(
        self,
        base_sequence: str,
        beneficial: Dict[int, List[str]],
        max_order: int = 3,
        use_qehvi: bool = True
    ) -> List[Variant]:
        # 1) Enumerate
        candidates = enumerate_combinations(base_sequence, beneficial, max_order=max_order)

        # 2) Embeddings (global)
        g = self._embed_variants(candidates)  # [M,d]

        # 3) Structural gate
        feats = self._structural_gate(candidates, g)  # list[CandidateFeatures]
        # integrate feasibility into predictions
        preds = self._predict(g)
        for p, f in zip(preds, feats):
            p.p_feasible = 1.0 if f.feasibility else 0.0

        # 4) Acquisition
        if use_qehvi:
            idx = select_batch_qEHVI(preds, self.cfg, batch_size=self.cfg.batch_size)
        else:
            embed_np = g.detach().cpu().numpy()
            idx = select_batch_ucb(preds, self.cfg, embed_np)

        return [candidates[i] for i in idx]

    def log_round(self, path: str, selected: Sequence[Variant], results: Optional[Sequence[AssayResult]] = None) -> None:
        record = {
            "timestamp": time.time(),
            "selected": [v.sequence for v in selected],
        }
        if results:
            record["results"] = [
                dict(seq=r.variant.sequence, stability=r.stability, activity=r.activity, selectivity=r.selectivity)
                for r in results
            ]
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
