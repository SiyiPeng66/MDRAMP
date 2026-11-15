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
    """V6 nnPU head with frozen nonparametric training-AMP support."""
    def __init__(self, input_dim, branch_weight=0.25, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.eta = float(branch_weight)
        self.backbone = torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim), torch.nn.GELU(), torch.nn.Dropout(dropout), torch.nn.Linear(hidden_dim,256), torch.nn.GELU(), torch.nn.Dropout(dropout))
        self.pu_head = torch.nn.Linear(256,1)
        self.register_buffer("reference", torch.empty(0,input_dim-32))
        self.register_buffer("reference_clusters",torch.empty(0,dtype=torch.long))
        self.register_buffer("radius",torch.tensor(0.))
        self.register_buffer("spread",torch.tensor(1.))
    @torch.no_grad()
    def fit_support(self, embeddings, clusters, k=10, chunk_size=256):
        """Exact leave-cluster-out distances using bounded query blocks."""
        z=torch.nn.functional.normalize(torch.as_tensor(embeddings,dtype=torch.float32),dim=1)
        if len(clusters)!=len(z) or chunk_size<1 or k<1:
            raise ValueError("Invalid support cluster count, block size or neighbour count")
        self.reference=z
        cluster_map={value:i for i,value in enumerate(dict.fromkeys(clusters))}
        self.reference_clusters=torch.tensor([cluster_map[c] for c in clusters],dtype=torch.long,device=z.device)
        counts=torch.bincount(self.reference_clusters)
        if len(z)-int(counts.max())<k:
            raise ValueError("Support requires ten references outside every query cluster")
        distances=[]
        for start in range(0,len(z),chunk_size):
            stop=min(start+chunk_size,len(z))
            block=1-z[start:stop]@z.T
            same=self.reference_clusters[start:stop,None]==self.reference_clusters[None,:]
            block.masked_fill_(same,float("inf"))
            distances.append(block.topk(k,largest=False,dim=1).values.mean(dim=1))
        d=torch.cat(distances)
        self.radius=torch.quantile(d,.95)
        self.spread=torch.quantile(d,.75)-torch.quantile(d,.25)

    def discriminative(self,features):
        """Only the trainable nnPU branch; support has no role in its loss."""
        return {"pu":torch.sigmoid(self.pu_head(self.backbone(features))).squeeze(-1)}

    def forward(self,features):
        pu=self.discriminative(features)["pu"]
        if len(self.reference)<10: raise ValueError("Frozen AMP support is not fitted")
        z=torch.nn.functional.normalize(features[:,:self.reference.shape[1]],dim=1)
        distances=1-z@self.reference.T
        # Known reference queries exclude their complete frozen identity cluster.
        matches=distances <= 1e-6
        for i in range(len(z)):
            if matches[i].any():
                clusters=self.reference_clusters[matches[i]]
                distances[i,torch.isin(self.reference_clusters,clusters)]=float("inf")
        distance=distances.topk(10,largest=False).values.mean(-1)
        support=torch.exp(-torch.relu(distance-self.radius)/(self.spread+1e-8))
        return {"pu":pu,"support":support,"probability":pu*((1-self.eta)+self.eta*support)}


class FeaturewiseGatedMembraneExpert(torch.nn.Module):
    """V6 single-task membrane Expert with feature-wise gating."""

    def __init__(self, esm_dim: int, physchem_dim: int = 32, hidden_dim: int = 256):
        super().__init__()
        self.sequence_branch = torch.nn.Sequential(
            torch.nn.Linear(esm_dim, 512), torch.nn.GELU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(512, hidden_dim), torch.nn.GELU(), torch.nn.Dropout(0.1)
        )
        self.physchem_branch = torch.nn.Sequential(
            torch.nn.Linear(physchem_dim, 128), torch.nn.GELU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(128, 64), torch.nn.GELU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(64, hidden_dim)
        )
        self.gate = torch.nn.Linear(hidden_dim * 2, hidden_dim)
        self.primary = torch.nn.Linear(hidden_dim, 1)
        self.sequence_norm = torch.nn.LayerNorm(hidden_dim)
        self.physchem_norm = torch.nn.LayerNorm(hidden_dim)
        torch.nn.init.zeros_(self.gate.bias)
        torch.nn.init.normal_(self.gate.weight, std=0.01)

    def forward(self, peptide_embedding: torch.Tensor, physchem: torch.Tensor) -> dict[str, torch.Tensor]:
        seq = self.sequence_norm(self.sequence_branch(peptide_embedding))
        pc = self.physchem_norm(self.physchem_branch(physchem))
        gate = torch.sigmoid(self.gate(torch.cat([seq, pc], dim=-1)))
        fused = gate * seq + (1.0 - gate) * pc
        return {
            "primary": torch.sigmoid(self.primary(fused)).squeeze(-1),
            "representation": fused,
        }


class ProjectedPairAffinityExpert(torch.nn.Module):
    """Pair Expert with projected interaction and rank-k bilinear features."""

    def __init__(self, peptide_dim: int, target_dim: int, physchem_dim: int = 32, projection_dim: int = 256, rank: int = 64):
        super().__init__()
        self.peptide_projection = torch.nn.Linear(peptide_dim, projection_dim)
        self.target_projection = torch.nn.Linear(target_dim, projection_dim)
        self.peptide_norm = torch.nn.LayerNorm(projection_dim)
        self.target_norm = torch.nn.LayerNorm(projection_dim)
        self.left_rank = torch.nn.Linear(projection_dim, rank, bias=False)
        self.right_rank = torch.nn.Linear(projection_dim, rank, bias=False)
        input_dim = projection_dim * 2 + physchem_dim + rank
        self.head = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512), torch.nn.GELU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(512, 256), torch.nn.GELU(), torch.nn.Dropout(0.1), torch.nn.Linear(256, 1)
        )

    def pair_logits(self, peptide_embedding: torch.Tensor, target_embedding: torch.Tensor, physchem: torch.Tensor) -> torch.Tensor:
        pep = self.peptide_norm(self.peptide_projection(peptide_embedding))
        target = self.target_norm(self.target_projection(target_embedding))
        bilinear = self.left_rank(pep) * self.right_rank(target)
        features = torch.cat([torch.abs(pep - target), pep * target, physchem, bilinear], dim=-1)
        return self.head(features).squeeze(-1)

    def forward(self, peptide_embedding: torch.Tensor, target_embedding: torch.Tensor, physchem: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.pair_logits(peptide_embedding, target_embedding, physchem))

    def forward_with_attention(self, peptide_embedding: torch.Tensor, target_embedding: torch.Tensor, physchem: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.pair_logits(peptide_embedding, target_embedding, physchem)
        return {"probability": torch.sigmoid(logits), "attention_logit": logits}
