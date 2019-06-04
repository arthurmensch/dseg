from typing import List

import torch
from torch import autograd, nn as nn


def simultaneous_grad(values: List[torch.Tensor],
                      inputs: List[torch.Tensor],
                      create_graph=False):
    return [autograd.grad(v, (p,), retain_graph=True,
                          create_graph=create_graph,
                          )[0] for v, p in zip(values, inputs)]


class ExtrapolatedObjective(nn.Module):
    def __init__(self, objective_fn: nn.Module,
                 step_size: float, lola: bool = False):
        super().__init__()
        self.objective_fn = objective_fn
        self.step_size = step_size
        self.lola = lola

    def forward(self, parameters: List[torch.tensor]):
        values = self.objective_fn(parameters)
        gradients = simultaneous_grad(values, parameters,
                                      create_graph=self.lola)
        return self.objective_fn([p + self.step_size * g for p, g
                                  in zip(parameters, gradients)])
