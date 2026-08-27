"""Expert neural modules for MDRAMP."""

from __future__ import annotations

import torch


class MLPExpert(torch.nn.Module):
    """A compact expert head over frozen ESM3 global embeddings and descriptors."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(features)).squeeze(-1)


class PairAffinityExpert(torch.nn.Module):
    """Pair-level target affinity expert over peptide and target embeddings."""

    def __init__(
        self,
        peptide_dim: int,
        target_dim: int,
        physchem_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        input_dim = peptide_dim + target_dim + physchem_dim + 4
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        peptide_embedding: torch.Tensor,
        target_embedding: torch.Tensor,
        physchem: torch.Tensor,
    ) -> torch.Tensor:
        interaction = torch.cat(
            [
                peptide_embedding,
                target_embedding,
                physchem,
                (peptide_embedding * target_embedding).mean(dim=1, keepdim=True),
                torch.abs(peptide_embedding - target_embedding).mean(dim=1, keepdim=True),
                peptide_embedding.norm(dim=1, keepdim=True),
                target_embedding.norm(dim=1, keepdim=True),
            ],
            dim=1,
        )
        return torch.sigmoid(self.net(interaction)).squeeze(-1)


class AMPPriorExpert(MLPExpert):
    """AMP-like feasibility expert."""


class MembraneMechanismExpert(MLPExpert):
    """Membrane mechanism expert distilled from Pore-Forming teacher."""


class PaperAMPPriorExpert(torch.nn.Module):
    """AMP feasibility head with separate nnPU and manifold branches."""

    def __init__(self, input_dim: int, branch_weight: float, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        if not 0.0 <= branch_weight <= 1.0:
            raise ValueError("branch_weight must be in [0, 1]")
        self.register_buffer("branch_weight", torch.tensor(float(branch_weight)))
        self.backbone = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 256),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )
        self.pu_head = torch.nn.Linear(256, 1)
        self.manifold_head = torch.nn.Linear(256, 1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        representation = self.backbone(features)
        pu = torch.sigmoid(self.pu_head(representation)).squeeze(-1)
        manifold = torch.sigmoid(self.manifold_head(representation)).squeeze(-1)
        return {
            "pu": pu,
            "manifold": manifold,
            "probability": self.branch_weight * pu + (1.0 - self.branch_weight) * manifold,
            "representation": representation,
        }


class FeaturewiseGatedMembraneExpert(torch.nn.Module):
    """Paper-shaped membrane Expert with masked auxiliary outputs."""

    def __init__(self, esm_dim: int, physchem_dim: int = 32, hidden_dim: int = 256):
        super().__init__()
        self.sequence_branch = torch.nn.Sequential(
            torch.nn.Linear(esm_dim, 512), torch.nn.GELU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(512, hidden_dim), torch.nn.GELU()
        )
        self.physchem_branch = torch.nn.Sequential(
            torch.nn.Linear(physchem_dim, 128), torch.nn.GELU(),
            torch.nn.Linear(128, 64), torch.nn.GELU(),
            torch.nn.Linear(64, hidden_dim), torch.nn.GELU()
        )
        self.gate = torch.nn.Linear(hidden_dim * 2, hidden_dim)
        self.primary = torch.nn.Linear(hidden_dim, 1)
        self.auxiliary = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, 1) for _ in range(4)])

    def forward(self, peptide_embedding: torch.Tensor, physchem: torch.Tensor) -> dict[str, torch.Tensor]:
        seq = self.sequence_branch(peptide_embedding)
        pc = self.physchem_branch(physchem)
        gate = torch.sigmoid(self.gate(torch.cat([seq, pc], dim=-1)))
        fused = gate * seq + (1.0 - gate) * pc
        return {
            "primary": torch.sigmoid(self.primary(fused)).squeeze(-1),
            "auxiliary": torch.stack([torch.sigmoid(head(fused)).squeeze(-1) for head in self.auxiliary], dim=-1),
            "representation": fused,
        }


class ProjectedPairAffinityExpert(torch.nn.Module):
    """Pair Expert with projected interaction and rank-k bilinear features."""

    def __init__(self, peptide_dim: int, target_dim: int, physchem_dim: int = 32, projection_dim: int = 256, rank: int = 64):
        super().__init__()
        self.peptide_projection = torch.nn.Linear(peptide_dim, projection_dim)
        self.target_projection = torch.nn.Linear(target_dim, projection_dim)
        self.left_rank = torch.nn.Linear(projection_dim, rank, bias=False)
        self.right_rank = torch.nn.Linear(projection_dim, rank, bias=False)
        input_dim = projection_dim * 2 + physchem_dim + rank
        self.head = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512), torch.nn.GELU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(512, 256), torch.nn.GELU(), torch.nn.Linear(256, 1)
        )

    def pair_logits(self, peptide_embedding: torch.Tensor, target_embedding: torch.Tensor, physchem: torch.Tensor) -> torch.Tensor:
        pep = self.peptide_projection(peptide_embedding)
        target = self.target_projection(target_embedding)
        bilinear = self.left_rank(pep) * self.right_rank(target)
        features = torch.cat([torch.abs(pep - target), pep * target, physchem, bilinear], dim=-1)
        return self.head(features).squeeze(-1)

    def forward(self, peptide_embedding: torch.Tensor, target_embedding: torch.Tensor, physchem: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.pair_logits(peptide_embedding, target_embedding, physchem))

    def forward_with_attention(self, peptide_embedding: torch.Tensor, target_embedding: torch.Tensor, physchem: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.pair_logits(peptide_embedding, target_embedding, physchem)
        return {"probability": torch.sigmoid(logits), "attention_logit": logits}
