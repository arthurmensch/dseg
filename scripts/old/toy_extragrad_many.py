import copy
import random
from typing import List

import numpy as np
import torch
from scipy._lib._util import check_random_state
from torch.nn import Parameter
from torch.optim import SGD

from gamesrl.numpy.games import make_positive_matrix
from gamesrl.optimizers import ExtraOptimizer, ExtraOptimizerVR

seed = 0
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

n_players = 3
n_actions = 3
A = torch.from_numpy(make_positive_matrix(n_players, n_actions, skewness=0.95).squeeze()).float()


def loss(xs: List[torch.tensor]):
    blocks = [[A[i, :, j, :] @ xs[j] for i in range(n_players) for j in range(n_players)]]
    return [torch.sum(xs[i] * sum(blocks[i])) for i in range(n_players)]


def all_scheduler(n_players, extrapolation):
    i = 0
    while True:
        extrapolate = i % 2 == 0 and extrapolation
        i += 1
        yield [True for _ in range(n_players)], extrapolate


def alternated_scheduler(n_players, extrapolation, random_state=None, shuffle=True):
    random_state = check_random_state(random_state)
    if extrapolation:
        all_pairs = [(i, j) for i in range(n_players) for j in range(n_players) if i != j]
        if shuffle:
            random_state.shuffle(all_pairs)
        while True:
            for (i, j) in all_pairs:
                mask = [False for _ in range(n_players)]
                mask[i] = True
                yield mask, True
                mask[i] = False
                mask[j] = True
                yield mask, False
            if shuffle:
                random_state.shuffle(all_pairs)
    else:
        all_players = list(range(n_players))
        if shuffle:
            random_state.shuffle(all_players)
        while True:
            for i in all_players:
                mask = [False for _ in range(n_players)]
                mask[i] = True
                yield mask, False
            if shuffle:
                random_state.shuffle(all_players)


def one_scheduler(n_players, extrapolation, random_state=None):
    random_state = check_random_state(random_state)
    all_players = list(range(n_players))
    i = 0
    while True:
        extrapolate = i % 2 == 0 and extrapolation
        i += 1
        mask = [False for _ in range(n_players)]
        select = random_state.choice(all_players)
        mask[select] = True
        yield mask, extrapolate


def bernouilli_scheduler(n_players, extrapolation, random_state=None, p=None):
    random_state = check_random_state(random_state)
    if p is None:
        p = 1. / n_players
    i = 0
    while True:
        extrapolate = i % 2 == 0 and extrapolation
        i += 1
        mask = (random_state.uniform(1) < p).tolist()
        yield mask, extrapolate


def solve(variance_reduction, extrapolation, sampling, averaging, seed):
    lr = .1
    length = 1000

    np.random.seed(seed)
    xs = [Parameter(torch.randn(n_actions)) for i in range(n_players)]

    if extrapolation:
        if variance_reduction:
            def make_optimizer(x):
                return ExtraOptimizerVR(SGD([x], lr=lr))
        else:
            def make_optimizer(x):
                return ExtraOptimizer(SGD([x], lr=lr))
    else:
        def make_optimizer(x):
            return SGD([x], lr=lr)

    optimizers = [make_optimizer(x) for x in xs]

    if averaging:
        xs_avg = copy.deepcopy(xs)
        n_update = 0
    computation = 0.
    computations = [computation]

    random_state = check_random_state(0)
    if sampling == 'all':
        scheduler = all_scheduler(n_players, extrapolation)
    elif sampling == 'alternated':
        scheduler = alternated_scheduler(n_players, extrapolation, random_state)
    elif sampling == 'bernouilli':
        scheduler = bernouilli_scheduler(n_players, extrapolation, random_state)
    elif sampling == 'one':
        scheduler = one_scheduler(n_players, extrapolation, random_state)

    for i in range(length - 1):
        mask, extrapolate = next(scheduler)
        update = not extrapolate
        for x, select in zip(xs, mask):
            x.requires_grad = select
        ls = loss(xs)
        for optimizer in optimizers:
            optimizer.zero_grad()

        for l, x, optimizer, select in zip(ls, xs, optimizers, mask):
            if not select:
                continue
            for x_ in xs:
                x_.requires_grad = x_ is x
            computation += 1
            l.backward(retain_graph=True)
            if extrapolate:
                optimizer.step(extrapolate=True)
            else:
                optimizer.step()

        if update:
            if extrapolation:
                for optimizer in optimizers:
                    optimizer.deextrapolate()
            if averaging:
                n_update += 1
                for x_avg, x in zip(xs_avg, xs):
                    x_avg *= (1 - 1 / n_update)
                    x_avg += x.item() / n_update
            computations.append(computation)
    computations = np.array(computations, dtype=float)
    return xs, computations


# exps = [dict(name='gd_all', variance_reduction=False, extrapolation=False, sampling='all'),
#         dict(name='gd_alternated', variance_reduction=False, extrapolation=False, sampling='alternated')]
# exps = []
# for variance_reduction in [True, False]:
#     for sampling in ['all', 'alternated', 'bernouilli', 'one']:
#         name = f'extra_{sampling}'
#         if variance_reduction:
#             name += '_vr'
#         exps.append(dict(name=name, extrapolation=True, sampling=sampling, variance_reduction=variance_reduction))
#
# n_repeat = 1
# seeds = np.random.randint(0, 10000, size=n_repeat)
# res = Parallel(n_jobs=3)(delayed(solve)(exp['variance_reduction'],
#                                         exp['extrapolation'], exp['sampling'],
#                                         True, this_seed)
#                          for exp in exps for this_seed in seeds)

