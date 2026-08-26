"""Fixed-seed downstream ensembles with mean and dispersion outputs."""

from __future__ import annotations

from collections.abc import Callable

import torch


class ExpertEnsemble(torch.nn.Module):
    def __init__(self, factory: Callable[[], torch.nn.Module], seeds=(11, 23, 37, 51, 73)):
        super().__init__()
        self.seeds = tuple(int(seed) for seed in seeds)
        members = []
        for seed in self.seeds:
            torch.manual_seed(seed)
            members.append(factory())
        self.members = torch.nn.ModuleList(members)

    def forward(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        outputs = torch.stack([member(*args, **kwargs) for member in self.members], dim=0)
        return {"members": outputs, "mean": outputs.mean(dim=0), "std": outputs.std(dim=0, unbiased=False)}
