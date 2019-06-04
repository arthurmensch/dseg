import functools
import math
from typing import List

import numpy as np
import torch
from torch.nn import Parameter

import torch.nn as nn


def matrix_reward(payoffs: torch.tensor, strategies: List[torch.tensor],
                  mask=None):
    N = len(strategies)
    if mask is None:
        mask = torch.ones(N, dtype=torch.uint8)

    payoffs = payoffs[mask]
    for i in reversed(range(N)):
        payoffs = torch.matmul(payoffs, strategies[i])
    return [p for p in payoffs]


def quadratic_reward(hessian: torch.tensor, strategies: List[torch.tensor],
                     mask: torch.tensor = None):
    N = len(strategies)
    if mask is None:
        mask = torch.ones(N, dtype=torch.uint8)

    cat_strategies = torch.cat(strategies)
    offset = 0
    vs = []
    for i, strategy in enumerate(strategies):
        d = len(strategy)
        if mask[i]:
            this_hessian = hessian[offset:offset + d]
            v = torch.sum(torch.matmul(this_hessian, cat_strategies)
                          * strategy)
            v -= torch.sum(torch.matmul(this_hessian[:, offset:offset + d],
                                        strategy) * strategy) / 2
            vs.append(v)
        offset += d
    return vs


def simultaneous_gradient(reward_fn: callable, strategies: List[torch.tensor],
                          mask: torch.tensor = None):
    N = len(strategies)

    if mask is None:
        mask = torch.ones(N, dtype=torch.uint8)

    with torch.enable_grad():
        masked_strategies = []
        for i in range(N):
            if mask[i]:
                strategies[i] = Parameter(strategies[i])
                masked_strategies.append(strategies[i])
        rs = reward_fn(strategies, mask)
    gs = [torch.autograd.grad(r, (s,), retain_graph=True)[0] for r, s in
          zip(rs, masked_strategies)]
    return rs, gs


def make_mask(player_index: int, n_players: int, modelling='all', p=0.5):
    if modelling == 'all':
        mask = torch.ones(n_players, dtype=torch.uint8)
    elif modelling == 'others':  # AKA Look Ahead
        mask = torch.ones(n_players, dtype=torch.uint8)
        mask[player_index] = 0
    elif modelling == 'self':
        mask = torch.zeros(n_players, dtype=torch.uint8)
        mask[player_index] = 1
    elif modelling == 'random':
        mask = torch.empty(N).uniform_(0., 1.) < p
    return mask


step_size = .1
n_iter = 2000
game = 'quadratic'
modelling = 'all'
p = .5

# Two-player zeroum games
if 'matrix' in game:
    if game == 'matrix_rps':
        A = torch.tensor([[0, -1., 1], [1, 0, -1], [-1, 1, 0]])
        # A += torch.randn(3, 3) * 2
        payoffs = torch.cat([A[None, :], - A[None, :]], dim=0)
    elif game == 'matrix_mp':
        A = torch.tensor([[1, -1.], [-1, 1]])
        payoffs = torch.cat([A[None, :], - A[None, :]], dim=0)
    elif game == 'matrix_random':
        N = 3
        d = 3
        # torch.manual_seed(20)

        payoffs = torch.randn(N, *(d for i in range(N))) / math.sqrt(math.pow(N, N)) * 10
    N, d = payoffs.shape[:2]
    reward_fn = functools.partial(matrix_reward, payoffs)
elif game == 'quadratic':
    torch.manual_seed(200)

    N = 5
    d = 5
    A = torch.randn(N * d, N * d)
    hessian = torch.matmul(A, A.transpose(1, 0))
    reward_fn = functools.partial(quadratic_reward, hessian)
else:
    raise ValueError('Wrong game')

torch.manual_seed(100)
log_strategies = torch.randn(N, d)
log_strategies_list = []
rewards_list = []

for i in range(n_iter):
    strategies = [torch.exp(log_s) for log_s in log_strategies]
    rs, gs = simultaneous_gradient(reward_fn, strategies)
    gs = torch.cat([g[None, :] for g in gs])

    rs = torch.tensor([r.item() for r in rs])
    log_strategies_list.append(log_strategies.clone()[None, :])
    rewards_list.append(rs.detach()[None, :])

    if modelling in ['none', 'all']:  # Centralize gradient computation
        if modelling == 'all':
            log_strategies_e = log_strategies.clone()
            log_strategies_e -= step_size * gs
            log_strategies_e -= torch.logsumexp(log_strategies_e,
                                                dim=1)[:, None]
            strategies_e = [torch.exp(log_s) for log_s in log_strategies_e]
            _, gs = simultaneous_gradient(reward_fn, strategies_e)
            gs = torch.cat([g[None, :] for g in gs])
    else:
        gs_e = []
        for j in range(N):   # Only played by player j
            mask = make_mask(j, N, modelling, p)
            log_strategies_e = log_strategies.clone()
            log_strategies_e[mask] -= step_size * gs[mask]
            log_strategies_e[mask] -= torch.logsumexp(log_strategies_e[mask],
                                                      dim=1)[:, None]
            strategies_e = [torch.exp(log_s) for log_s in log_strategies_e]
            _, g = simultaneous_gradient(reward_fn, strategies_e,
                                          make_mask(j, N, 'self'))
            gs_e.append(g[0])
        gs = torch.cat([g[None, :] for g in gs_e])
    log_strategies -= step_size * gs
    log_strategies -= torch.logsumexp(log_strategies, dim=1)[:, None]

strategies = torch.softmax(torch.cat(log_strategies_list, dim=0),
                           dim=2).numpy()
rewards = torch.cat(rewards_list, dim=0).numpy()

import matplotlib.pyplot as plt

strategies = np.cumsum(strategies, axis=2)
fig, axes = plt.subplots(1, N + 1, figsize=(3 * (N + 1), 3))
for i in range(N):
    for a in range(d):
        if a == 0:
            ref = np.zeros(n_iter)
        else:
            ref = strategies[:, i, a - 1]
        axes[i].fill_between(range(n_iter), ref, strategies[:, i, a],
                             label=f'Action #{a}')
    axes[i].set_title(f'Player #{i}')
    axes[i].set_xlabel('Iteration')
    axes[i].set_ylabel('Mixed strategy')
    axes[i].set_ylim([0, 1])
    axes[i].set_xlim([0, n_iter - 1])
axes[N - 1].legend()
for i in range(N):
    axes[N].plot(range(20, n_iter), rewards[20:, i], label=f'P{i}')
# axes[n_players].legend(ncol=4, loc='upper right')
axes[N].set_title('Reward')
axes[N].set_xlabel('Iteration')
# axes[n_players].set_ylim([-5, 5])
plt.show()
