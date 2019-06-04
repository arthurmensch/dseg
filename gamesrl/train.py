import copy
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils import check_random_state
from torch import autograd
from torch.nn import Parameter
from torch.optim import SGD

from gamesrl.games import Exploiter
from gamesrl.optimizers import ExtraOptimizer, ExtraOptimizerVR
from gamesrl.schedulers import FullScheduler, AlternatedOneScheduler, BernouilliScheduler, RandomSubsetScheduler


class ParamAverager(nn.Module):
    def __init__(self, module, mode='uniform', beta=0.9):
        super().__init__()
        self.module = module
        self.average = copy.deepcopy(module)
        self.mode = mode
        self.n_step = 0
        self.beta = beta

    def step(self):
        self.n_step += 1
        for avg, p in zip(self.average.parameters(), self.module.parameters()):
            if self.mode == 'uniform':
                avg.data *= (1 - 1 / self.n_step)
                avg.data += p.data / self.n_step
            else:
                avg.data *= self.beta
                avg.data += p.data * (1 - self.beta)


def mirror_prox_minimize(objective, player, extrapolation=True,
                         n_iter=100, step_size=1e-2, eval_every=1000, tol=1e-1,
                         averaging=True, ):
    if averaging:
        averager = ParamAverager(player)
        average = averager.average
    else:
        average = player

    optimizer = SGD(player.parameters(), lr=step_size)
    if extrapolation:
        optimizer = ExtraOptimizer(optimizer)

    elapsed_time = 0.
    next_eval_iter = 0
    for n_computations in range(n_iter):
        if eval_every and n_computations >= next_eval_iter:
            next_eval_iter += eval_every
            loss, err = compute_dual_gap(objective, average)
            if 0 < err < tol * loss:
                break

        extrapolate = extrapolation and n_computations % 2 == 0
        update = not extrapolate

        t0 = time.perf_counter()

        optimizer.zero_grad()
        loss = objective(player())
        loss.backward()
        if extrapolation:
            optimizer.step(extrapolate=extrapolate)
        else:
            optimizer.step()
        if update and averaging:
            averager.step()

        elapsed_time += time.perf_counter() - t0

    loss, err = compute_dual_gap(objective, average)
    err = 0
    if averaging:
        return loss, err, average
    else:
        return loss, err, player


def compute_dual_gap(objective, player):
    strategy = Parameter(player())
    loss = objective(strategy)
    grad = autograd.grad(loss, (strategy,))[0]
    err = torch.sum(grad * strategy) - torch.min(grad)
    return loss.item(), err.item()


def compute_nash_gap(game, players, tol=1e-1, step_size=1e-2, n_iter=1000):
    strategies = [player() for player in players]
    err = 0
    nash_gap = 0
    losses = []
    for i, player in enumerate(players):
        exploiter = Exploiter(game, strategies, index=i)
        player = copy.deepcopy(player)
        loss = game(strategies, index=i)[0].item()
        losses.append(loss)
        value, this_err, _ = mirror_prox_minimize(exploiter, player, n_iter=n_iter, tol=tol,
                                                  step_size=step_size)
        nash_gap += loss - value
        err += this_err
    nash_gap_l = nash_gap
    nash_gap_u = nash_gap + err
    return losses, nash_gap_l, nash_gap_u, players


def mirror_prox_nash(game, players, seed=None, extrapolation=True,
                     n_iter=1000, step_size=1., eval_every=100, tol=0,
                     inner_n_iter=1000, inner_step_size=1e-2, inner_tol=1e-1,
                     sampling='all', averaging=True, callback=None, callback_every=0,
                     variance_reduction=False, _run=None):
    random_state = check_random_state(seed)

    players = np.array(players)
    n_players = len(players)

    if averaging:
        averagers = np.array([ParamAverager(player) for player in players])
        averages = [averager.average for averager in averagers]
    else:
        averages = players

    if sampling == 'all':
        scheduler = FullScheduler(n_players, extrapolation)
        subsampling = 1
    elif sampling == 'alternated':
        scheduler = AlternatedOneScheduler(n_players, extrapolation, random_state)
        subsampling = 1 / n_players
    elif sampling == 'bernouilli':
        scheduler = BernouilliScheduler(n_players, extrapolation, random_state, batch_size=1)
        subsampling = 1 / n_players
    elif isinstance(sampling, int):
        player_batch_size = sampling
        scheduler = RandomSubsetScheduler(n_players, extrapolation, random_state, batch_size=player_batch_size)
        subsampling = player_batch_size / n_players
    else:
        raise ValueError(f'Wrong `sampling` argument, got {sampling}')

    def make_optimizer(strategy):
        optimizer = SGD(strategy.parameters(), lr=step_size)
        if extrapolation:
            if variance_reduction:
                return ExtraOptimizerVR(optimizer, subsampling=subsampling)
            else:
                return ExtraOptimizer(optimizer, subsampling=subsampling)
        else:
            return optimizer

    optimizers = np.array([make_optimizer(player) for player in players])

    elapsed_time = 0.
    n_computations = 0.
    next_eval_iter = 0
    next_callback_iter = 0.
    update = True

    # with torch.autograd.profiler.profile() as prof:
    while n_computations < n_iter:
        if eval_every and n_computations >= next_eval_iter:
            next_eval_iter += eval_every

            losses, nash_gap_l, nash_gap_u, _ = compute_nash_gap(game, averages, tol=inner_tol,
                                                                 step_size=inner_step_size,
                                                                 n_iter=inner_n_iter)

            if 0 < nash_gap_u < tol:
                break

            print(f'Iter {n_computations:.0f},'
                  f' nash gap {(nash_gap_l + nash_gap_u) / 2:.1e} ± {(nash_gap_u - nash_gap_l) / 2:.0e}')
            if _run is not None:
                _run.log_scalar('nash_gap_u_vs_computations', nash_gap_u, step=n_computations)
                _run.log_scalar('nash_gap_u_vs_time', nash_gap_u, step=elapsed_time)
                _run.log_scalar('nash_gap_l_vs_computations', nash_gap_l, step=n_computations)
                _run.log_scalar('nash_gap_l_vs_time', nash_gap_l, step=elapsed_time)
        if update and n_computations >= next_callback_iter:
            next_callback_iter += callback_every
            callback(averages, n_computations)

        t0 = time.perf_counter()

        for optimizer in optimizers:
            optimizer.zero_grad()
        for i, player in enumerate(players):
            for p in player.parameters():
                p.requires_grad = False

        indices, extrapolate = next(scheduler)
        update = not extrapolate
        n_computations += len(indices) / n_players

        for i in indices:
            for p in players[i].parameters():
                p.requires_grad = True

        masked_losses = game([player() for player in players], index=indices)
        # Grad computation
        for loss, player in zip(masked_losses, players[indices]):
            for i, other_player in enumerate(players):
                requires_grad = other_player is player
                for p in other_player.parameters():
                    p.requires_grad = requires_grad
            loss.backward(retain_graph=True)
        # Update
        if variance_reduction:
            indices = np.arange(n_players)
        for optimizer, averager in zip(optimizers[indices], averagers[indices]):
            if extrapolation:
                optimizer.step(extrapolate=extrapolate)
            else:
                optimizer.step()
            if update and averaging:
                averager.step()

        elapsed_time += time.perf_counter() - t0
    losses, nash_gap_l, nash_gap_u, _ = compute_nash_gap(game, averages)
    return losses, nash_gap_l, nash_gap_u, averages