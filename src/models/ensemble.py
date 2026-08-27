"""Fixed-seed downstream ensembles with mean and dispersion outputs."""

from __future__ import annotations

from collections.abc import Callable

import torch


class ExpertEnsemble(torch.nn.Module):
    def __init__(
        self,
        factory: Callable[[], torch.nn.Module],
        seeds=(11, 23, 37, 51, 73),
        output_key: str | None = None,
    ):
        super().__init__()
        self.seeds = tuple(int(seed) for seed in seeds)
        self.output_key = output_key
        members = []
        for seed in self.seeds:
            torch.manual_seed(seed)
            members.append(factory())
        self.members = torch.nn.ModuleList(members)

    def forward(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        member_outputs = [member(*args, **kwargs) for member in self.members]
        if self.output_key is not None:
            member_outputs = [output[self.output_key] for output in member_outputs]
        if not all(isinstance(output, torch.Tensor) for output in member_outputs):
            raise TypeError("Ensemble members returned structured outputs; set output_key.")
        outputs = torch.stack(member_outputs, dim=0)
        return {"members": outputs, "mean": outputs.mean(dim=0), "std": outputs.std(dim=0, unbiased=False)}
