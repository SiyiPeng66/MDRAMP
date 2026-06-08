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

