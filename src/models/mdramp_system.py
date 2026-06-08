"""Combined MDRAMP scoring system."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.models.gated_integrator import GatedIntegrator


@dataclass
class ComponentScores:
    S_prior: np.ndarray
    S_mem: np.ndarray
    S_aff: np.ndarray


class MDRAMPSystem(torch.nn.Module):
    def __init__(
        self,
        amp_prior: torch.nn.Module,
        membrane: torch.nn.Module,
        target_affinity: torch.nn.Module | None,
        integrator: GatedIntegrator,
    ) -> None:
        super().__init__()
        self.amp_prior = amp_prior
        self.membrane = membrane
        self.target_affinity = target_affinity
        self.integrator = integrator

    def score_components(
        self,
        peptide_features: torch.Tensor,
        S_aff: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        S_prior = self.amp_prior(peptide_features)
        S_mem = self.membrane(peptide_features)
        integrated = self.integrator(S_prior, S_mem, S_aff)
        return {
            "S_prior": S_prior,
            "S_mem": S_mem,
            "S_aff": S_aff,
            **integrated,
        }

