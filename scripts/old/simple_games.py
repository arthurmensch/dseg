import json
import os

import math
import time
from os.path import expanduser, join

import matplotlib.pyplot as plt

import torch
from gamesrl.extrapolation import simultaneous_grad
from gamesrl.games import RockPaperScissor, \
    MatchingPennies, RandomMatrixGame, RandomQuadraticGame
from gamesrl.strategies import BypassedSoftmax
from joblib import delayed, Parallel
from torch.nn import Parameter

import seaborn as sns
import numpy as np

output_dir = expanduser('~/output/games_rl/subsampling_big')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def objective_fn(game, strategies, mask=None):
    parameters = [BypassedSoftmax.apply(s, 0) for s in strategies]
    return game(parameters, mask)


def run_all(game, n_iter=10000, step_size=10, n_jobs=4,
            same_mask=False,
            extrapolate='la', ps=None):

    Parallel(n_jobs=n_jobs)(delayed(run_single)(game, same_mask=same_mask,
                                                step_size=step_size,
                                                extrapolate=extrapolate,
                                                n_iter=n_iter, p=p,
                                                index=i)
                            for i, p in enumerate(ps))


def run_single(game, same_mask=False, step_size=10., extrapolate='la',
               n_iter=100, p=1, index=0):
    torch.manual_seed(100)

    if game == 'rps':
        game = RockPaperScissor()
    elif game == 'mp':
        game = MatchingPennies()
    elif game == 'random_matrix':
        game = RandomMatrixGame(n_players=2, n_actions=3)
    elif game == 'quadratic':
        game = RandomQuadraticGame(n_players=3, n_actions=3)
        print(game.L)
    n_players = game.n_players
    n_actions = game.n_actions
    create_graph = extrapolate == 'lola'
    strategies = [Parameter(torch.randn(n_actions)) for _ in range(n_players)]

    strategies_r = [[] for _ in range(n_players)]
    values_r = [[] for _ in range(n_players)]
    gradient_computations_r = []
    iterations_r = []
    timings_r = []
    gap_r = []
    avg_strategies = [torch.softmax(s.detach(), dim=0) for s in strategies]

    this_step_size = step_size * math.sqrt(p)
    gradient_computations = 0
    timings = 0

    for t in range(n_iter):
        t0 = time.perf_counter()
        step_mask = torch.empty(n_players).uniform_(0, 1) < p

        if extrapolate != 'none':
            if same_mask:
                extra_mask = step_mask
            else:
                extra_mask = torch.empty(n_players).uniform_(0, 1) < p
            if torch.any(extra_mask):
                masked_values = objective_fn(game, strategies, extra_mask)

                masked_strategies = [s for i, s in enumerate(strategies) if
                                     extra_mask[i]]
                masked_gradients = simultaneous_grad(masked_values,
                                                     masked_strategies,
                                                     create_graph=create_graph)

                masked_players = torch.sum(extra_mask.float()).item()
                gradient_computations += masked_players

                extra_strategies = []
                ii = 0
                for i in range(n_players):
                    if extra_mask[i]:
                        extra_strategies.append(masked_strategies[ii]
                                                - this_step_size *
                                                masked_gradients[
                                                    ii])
                        ii += 1
                    else:
                        extra_strategies.append(strategies[i])
            else:
                extra_strategies = strategies
        else:
            extra_strategies = strategies

        if torch.any(step_mask):
            masked_values = objective_fn(game, extra_strategies, step_mask)
            masked_strategies = [s for i, s in enumerate(strategies) if
                                 step_mask[i]]
            masked_gradients = simultaneous_grad(masked_values,
                                                 masked_strategies)

            masked_players = torch.sum(extra_mask.float()).item()
            gradient_computations += masked_players

            ii = 0
            for i in range(n_players):
                if step_mask[i]:
                    strategies[i].data -= this_step_size * masked_gradients[ii]
                    ii += 1

        # Averaging all
        avg_strategies = [torch.softmax(avg_s, 0) * (1 - 1 / (t + 1))
                          + torch.softmax(s.detach(), 0) / (t + 1)
                          for avg_s, s in zip(avg_strategies, strategies)]
        timings += time.perf_counter() - t0

        # Timer stopped for recordings
        values = objective_fn(game, avg_strategies)
        for l, v, l_rec, v_rec in zip(avg_strategies, values, strategies_r,
                                      values_r):
            l_rec.append(l.data.detach().clone()[None, :])
            v_rec.append(v.item())
        gradient_computations_r.append(gradient_computations)
        iterations_r.append(t)
        timings_r.append(timings)

        log_us = [Parameter(torch.log(s.detach())) for s in avg_strategies]
        for i in range(50):
            us = [BypassedSoftmax.apply(u, 0) for u in log_us]
            values = game(us)
            gradients = simultaneous_grad(values, log_us, create_graph=True)
            gap = 0
            for g, u, s in zip(gradients, us, avg_strategies):
                gap += torch.sum(g * (s - u))
            if i == 0:
                initial_gap = gap.item()
            gradients = torch.autograd.grad(gap, log_us)
            for u, g in zip(log_us, gradients):
                u.data += g / math.sqrt(i + 1) / 2 / game.L_quad
        gap_r.append(gap.item())
        print(f'Iteration {t}, gap {gap.item()}, initial_gap {initial_gap}')
    strategies_r = [torch.cat(strategy_r, dim=0).tolist()
                    for strategy_r in strategies_r]

    res = {'n_players': n_players, 'n_actions': n_actions,
           'p': p, 'same_mask': same_mask, 'step_size': step_size,
           'iterations': iterations_r,
           'timings': timings_r,
           'gradient_computations': gradient_computations_r,
           'strategies': strategies_r,
           'values': values_r,
           'gap': gap_r,
           }

    with open(join(output_dir, f'results_{index}.json'), 'w+') as f:
        json.dump(res, f)


def plot_single(index):
    with open(join(output_dir, f'results_{index}.json'), 'r') as f:
        res = json.load(f)

    n_players = res['n_players']
    n_actions = res['n_actions']
    iterations = res['iterations']
    gap = res['gap']
    # timings = res['timings']
    # gradient_computations = res['gradient_computations']
    strategies = res['strategies']
    values = res['values']

    iterations = np.array(iterations)
    strategies = np.array(strategies)

    fig, axes = plt.subplots(1, n_players + 2,
                             figsize=(3 * (n_players + 2), 3))
    for i, (ax, strategy) in enumerate(zip(axes, strategies)):
        strategy = np.cumsum(strategy, axis=1)
        for a in range(n_actions):
            ref = np.zeros_like(strategy[:, a]) if a == 0 else strategy[:,
                                                               a - 1]
            ax.fill_between(iterations, ref, strategy[:, a],
                            label=f'Action #{a}')
        ax.set_title(f'Player #{i}')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Mixed strategy')
        ax.set_ylim([0, 1])
    axes[n_players - 1].legend()
    for i, value in enumerate(values):
        axes[n_players].plot(iterations, value, label=f'P{i}')
    axes[n_players].set_title('Reward')
    axes[n_players].set_xlabel('Iteration')

    axes[n_players + 1].plot(iterations, gap, label=f'P{i}')
    axes[n_players + 1].set_title('Gap')
    axes[n_players + 1].set_xlabel('Iteration')

    plt.savefig(join(output_dir, f'convergence_{index}.png'))

    plt.show()


def plot_compare():
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), constrained_layout=True)

    handles = []
    player_handles = []
    labels = []
    player_labels = []
    for index in range(4):
        with open(join(output_dir, f'results_{index}.json'), 'r') as f:
            res = json.load(f)
        timings = res['timings']
        iterations = res['iterations']
        values = res['values']
        p = res['p']
        n_players = res['n_players']
        gap = res['gap']
        for player in range(n_players):
            cmap = sns.light_palette((23 * player, 90, 60), input="husl",
                                     n_colors=10, reverse=True)
            h, = axes[0].plot(timings, values[player], color=cmap[index],
                              )
            axes[1].plot(iterations, values[player], color=cmap[index])
            if index == 0:
                player_handles.append(h)
                player_labels.append(f'Player {player}')
        cmap = sns.light_palette((0, 90, 60), input="husl",
                                 n_colors=5, reverse=True)
        h, = axes[2].plot(timings, gap, color=cmap[index])
        handles.append(h)
        labels.append(f'p = {p:.1f}')
        axes[3].plot(iterations, gap, color=cmap[index])

    fig.legend(handles, labels, ncol=2,
               bbox_to_anchor=[0.6, 0.9],
               loc='upper left', frameon=False)
    fig.legend(player_handles, player_labels, ncol=2,
               bbox_to_anchor=[0.1, 0.9],
               loc='upper left', frameon=False)
    axes[0].set_xlabel('CPU time')
    axes[1].set_xlabel('Iteration')
    axes[2].set_xlabel('CPU time')
    axes[2].set_ylabel('VI Gap')
    axes[3].set_xlabel('Iteration')
    axes[0].set_ylabel('Reward')
    sns.despine(fig)
    plt.show()


run_all('quadratic', n_iter=100, step_size=10,
        ps=np.linspace(0.25, 1, 4), n_jobs=4)
# for i in range(10):
#     plot_single(i)
plot_compare()
