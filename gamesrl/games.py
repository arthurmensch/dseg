from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn


def _softplus(x, players):
    x = - F.softplus(x)
    for ii, i in enumerate(players):
        x[ii, :, i] = - x[ii, :, i]
    return x


def _identity(x, players):
    return x


def check_index_mask(index, n_players):
    if isinstance(index, int):
        index = [index]
    players = np.arange(n_players)
    return players[index].tolist()


class Exploiter(nn.Module):
    def __init__(self, game, ref_strategies, index):
        super().__init__()
        self.game = game
        self.index = index
        self.ref_strategies = [strategy.detach() for strategy in ref_strategies]

    def forward(self, strategy):
        self.ref_strategies[self.index] = strategy
        return self.game(self.ref_strategies, index=self.index)[0]


class QuadraticGame(nn.Module):
    def __init__(self, matrix: torch.tensor, activation='identity', noise=0.):
        super().__init__()
        self.n_players, self.n_actions = matrix.shape[:2]
        self.register_buffer('matrix', matrix)
        if activation == 'softplus':
            self.activation = _softplus
        elif activation == 'identity':
            self.activation = _identity
        else:
            raise ValueError(f'Wrong `activation` parameter, got {activation}')
        self.noise = noise

    def forward(self, strategies, index=slice(None)):
        players = check_index_mask(index, self.n_players)
        stack_strategies = torch.cat([strategy[None, :] for strategy in strategies], dim=0)
        masked_strategies = stack_strategies[players]
        masked_matrix = self.matrix[players]
        prods = torch.einsum('ikjl,jl->ikj', masked_matrix, stack_strategies)
        prods = self.activation(prods, players)

        values = torch.einsum('ik,ikj->i', masked_strategies, prods)
        if self.noise > 0:
            eps = self.noise * torch.randn_like(masked_strategies)
            values += torch.sum(masked_strategies * eps, dim=1)
        return [value for value in values]


class MatchingPennies(QuadraticGame):
    def __init__(self):
        A = torch.tensor([[1, -1], [-1, 1]])
        matrix = torch.zeros((2, 2, 2, 2))
        for i in range(2):
            matrix[i, :, i, :] = A
        super().__init__(matrix)


class RockPaperScissor(QuadraticGame):
    def __init__(self):
        matrix = torch.zeros((2, 3, 2, 3))
        A = torch.tensor([[0, -1., 1], [1, 0, -1], [-1, 1, 0]])
        for i in range(2):
            matrix[i, :, i, :] = A
        super().__init__(matrix)


class IteratedGame(nn.Module):
    def __init__(self, payoff, discount: float = 0.96):
        super().__init__()
        self.register_buffer('payoff', payoff)
        self.discount = discount
        self.n_players = 2
        self.dim_strategy = 5

    def forward(self, strategies: List[torch.tensor], index=slice(None)):
        players = check_index_mask(index)
        assert len(strategies) == self.n_players
        assert strategies[0].shape[0] == self.dim_strategy
        # strategies[0] : p(C|start), p(C|CC,CD,DC,DD)
        c0 = strategies[0][:, None]
        d0 = 1 - c0
        c1 = strategies[1][:, None]
        d1 = 1 - c1
        P = torch.cat((c0 * c1, c0 * d1, d0 * c1, d0 * d1), dim=1)
        p = P[0]
        P = P[1:]
        P = torch.eye(4) - self.discount * P
        vs = torch.sum(p[:, None] * torch.solve(self.payoff[:, players], P)[0], dim=0) * (1 - self.discount)
        return [v for v in vs]


class IteratedPrisonerDilemna(IteratedGame):
    def __init__(self, discount: float = 0.96):
        super().__init__(discount=discount,
                         payoff=torch.tensor([[-1, -3, 0., -2],
                                              [-1, 0, -3, -2]]).transpose(0, 1))


class IteratedMatchingPennies(IteratedGame):
    def __init__(self, discount: float = 0.96):
        super().__init__(discount=discount,
                         payoff=torch.tensor([[-1, 1, 1., -1],
                                              [1, -1, -1, 1]]).transpose(0, 1))
