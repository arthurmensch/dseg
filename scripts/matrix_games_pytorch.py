import os
from os.path import expanduser

import torch
from joblib import Memory
from sacred import Experiment
from sacred.observers import FileStorageObserver

from gamesrl.games import QuadraticGame
from gamesrl.numpy.games import make_positive_matrices
from gamesrl.strategies import MatrixGameStrategy
from gamesrl.train import mirror_prox_nash

exp = Experiment('matrix_games_clean')
exp_dir = expanduser(f'~/output/games_rl/{exp.path}')


@exp.config
def default():
    n_players = 50
    n_actions = 5
    gaussian_noise = 1

    n_iter = 10000
    eval_every = 1000
    step_size = .1

    inner_n_iter = 10000
    inner_tol = 1e-1
    inner_step_size = 1e-2

    variance_reduction = False
    extrapolation = False
    averaging = True
    sampling = 'alternated'
    conditioning = 0.01
    skewness = 0.9
    seed = 1000


@exp.main
def single(n_players, n_actions, _seed, conditioning, skewness, extrapolation,
           averaging, n_iter, eval_every, sampling, variance_reduction, _run):
    mem = Memory(location=expanduser('~/cache'))
    H = mem.cache(make_positive_matrices)(n_players, n_actions, 1, conditioning, skewness, 0, _seed)[0]
    H = torch.from_numpy(H).float()
    game = QuadraticGame(H, activation='softplus')
    players = [MatrixGameStrategy(n_actions, mirror=True) for _ in range(n_players)]

    losses, nash_gap_l, nash_gap_u, players = mirror_prox_nash(game, players, extrapolation=extrapolation,
                                                               averaging=averaging, n_iter=n_iter,
                                                               eval_every=eval_every,
                                                               sampling=sampling, variance_reduction=variance_reduction,
                                                               _run=_run)

    _run.info['losses'] = losses


if __name__ == '__main__':
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    exp.observers = [FileStorageObserver.create(exp_dir)]
    exp.run_commandline()
