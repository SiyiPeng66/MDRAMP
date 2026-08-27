"""Manuscript-faithful gated evidence integration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class GateParams:
    tau: float = 0.5
    gamma: float = 10.0
    lambda_syn: float = 0.3


def integrate_numpy(
    S_prior: np.ndarray,
    S_mem: np.ndarray,
    S_aff: np.ndarray,
    params: GateParams = GateParams(),
    prior_std: np.ndarray | None = None,
    mem_std: np.ndarray | None = None,
    aff_std: np.ndarray | None = None,
    kappa: float = 1.0,
) -> dict[str, np.ndarray]:
    S_prior = np.asarray(S_prior, dtype=np.float64)
    S_mem = np.asarray(S_mem, dtype=np.float64)
    S_aff = np.asarray(S_aff, dtype=np.float64)
    if prior_std is not None:
        S_prior = np.clip(S_prior - kappa * np.asarray(prior_std), 0.0, 1.0)
    if mem_std is not None:
        S_mem = np.clip(S_mem - kappa * np.asarray(mem_std), 0.0, 1.0)
    if aff_std is not None:
        S_aff = np.clip(S_aff - kappa * np.asarray(aff_std), 0.0, 1.0)
    g_prior = 1.0 / (1.0 + np.exp(-params.gamma * (S_prior - params.tau)))
    E_comp = 1.0 - (1.0 - S_mem) * (1.0 - S_aff)
    E_syn = S_mem * S_aff
    E_mech = (1.0 - params.lambda_syn) * E_comp + params.lambda_syn * E_syn
    p_cons = g_prior * E_mech
    return {
        "g_prior": g_prior,
        "E_comp": E_comp,
        "E_syn": E_syn,
        "E_mech": E_mech,
        "p_cons": p_cons,
    }


class GatedIntegrator(torch.nn.Module):
    def __init__(self, tau: float = 0.5, gamma: float = 10.0, lambda_syn: float = 0.3):
        super().__init__()
        self.tau = torch.nn.Parameter(torch.tensor(float(tau)), requires_grad=False)
        self.gamma = torch.nn.Parameter(torch.tensor(float(gamma)), requires_grad=False)
        self.lambda_syn = torch.nn.Parameter(torch.tensor(float(lambda_syn)), requires_grad=False)

    def forward(
        self,
        S_prior: torch.Tensor,
        S_mem: torch.Tensor,
        S_aff: torch.Tensor,
        prior_std: torch.Tensor | None = None,
        mem_std: torch.Tensor | None = None,
        aff_std: torch.Tensor | None = None,
        kappa: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        if prior_std is not None:
            S_prior = torch.clamp(S_prior - kappa * prior_std, 0.0, 1.0)
        if mem_std is not None:
            S_mem = torch.clamp(S_mem - kappa * mem_std, 0.0, 1.0)
        if aff_std is not None:
            S_aff = torch.clamp(S_aff - kappa * aff_std, 0.0, 1.0)
        g_prior = torch.sigmoid(self.gamma * (S_prior - self.tau))
        E_comp = 1.0 - (1.0 - S_mem) * (1.0 - S_aff)
        E_syn = S_mem * S_aff
        E_mech = (1.0 - self.lambda_syn) * E_comp + self.lambda_syn * E_syn
        p_cons = g_prior * E_mech
        return {
            "g_prior": g_prior,
            "E_comp": E_comp,
            "E_syn": E_syn,
            "E_mech": E_mech,
            "p_cons": p_cons,
        }
