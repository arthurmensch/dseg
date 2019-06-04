import copy
import os
import shutil
from os.path import expanduser, join

import numpy as np
from joblib import Parallel, delayed
from matrix_games import exp
from sacred.observers import FileStorageObserver
from sklearn.model_selection import ParameterGrid


def run_exp(output_dir, config_updates, _id, mock=False):
    """Boiler plate function that has to be put in every multiple
        experiment script, as exp does not pickle."""
    if not mock:
        observer = FileStorageObserver.create(basedir=output_dir)
        run = exp._create_run(config_updates=config_updates, )
        run._id = _id
        run.observers = [observer]
        run()
    else:
        exp.run('print_config', config_updates=config_updates, )


def common():
    n_players = 50
    n_actions = 5
    n_matrices = 1
    stochastic_noise = 0

    n_iter = 50000
    eval_every = 5000

    conditioning = 0.01

    extrapolation = True
    averaging = 'uniform'
    schedule = 'constant'

    sampling = 'all'
    variance_reduction = 'auto'
    skewness = 1
    l1_penalty = 0.
    gaussian_noise = 0
    step_size = 1.

    eval = {'step_size': .01, 'tol': 1e-1, 'n_iter': 10000, 'extrapolation': True,
            'averaging': 'uniform', 'schedule': 'constant'}
    seed = 100
    training_seed = 100


def grid(n_jobs=1):
    exp.config(common)

    output_dir = expanduser(f'~/output/games_rl/matrix_games_final_skew_noise')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(join(output_dir, '_sources'))
    seeds = [100, 120, 140, 160, 170]
    # base_list_5 = list(ParameterGrid({'sampling': ['all', 'alternated', 1, 2],
    #                                   'step_size': np.logspace(-6, 1, 16),
    #                                   'variance_reduction': [False],
    #                                   'seed': seeds,
    #                                   'n_players': [5]}))
    training_seeds = [100, 200]
    # full_list_5 = []
    # for base_update in base_list_5:
    #     for (skewness, l1_penalty, gaussian_noise) in [(0.9, 0., 0.,),
    #                                                    (0.9, 0., 1.,),
    #                                                    (0.9, 2e-2, 1.,),
    #                                                    (1, 2e-2, 1.,)]:
    #         update = copy.deepcopy(base_update)
    #         update['skewness'] = skewness
    #         update['l1_penalty'] = l1_penalty
    #         update['gaussian_noise'] = gaussian_noise
    #         if base_update['sampling'] in [1, 2]:
    #             for training_seed in training_seeds:
    #                 seed_update = copy.deepcopy(update)
    #                 seed_update['training_seed'] = training_seed
    #                 full_list_5.append(seed_update)
    #         else:
    #             full_list_5.append(update)
    #
    # base_list_50 = list(ParameterGrid({'sampling': ['all', 'alternated', 1, 5, 25],
    #                                    'step_size': np.logspace(-6, 1, 16),
    #                                    'variance_reduction': [True, False],
    #                                    'seed': seeds,
    #                                    'n_players': [50]}))
    # full_list_50 = []
    # for base_update in base_list_50:
    #     for (skewness, l1_penalty, gaussian_noise) in [(0.9, 0., 0.,),
    #                                                    (0.9, 0., 1.,),
    #                                                    (0.9, 2e-2, 1.,),
    #                                                    (1, 2e-2, 50.,)]:
    #         update = copy.deepcopy(base_update)
    #         update['skewness'] = skewness
    #         update['l1_penalty'] = l1_penalty
    #         update['gaussian_noise'] = gaussian_noise
    #         if base_update['sampling'] in [1, 5, 25]:
    #             for training_seed in training_seeds:
    #                 seed_update = copy.deepcopy(update)
    #                 seed_update['training_seed'] = training_seed
    #                 full_list_50.append(seed_update)
    #         else:
    #             full_list_50.append(update)

    base_list_noise = list(ParameterGrid({'sampling': ['all', 'alternated', 1, 5, 25],
                                          'step_size': np.logspace(-6, 1, 16),
                                          'skewness': [0.95, 1],
                                          'variance_reduction': [True, False],
                                          'gaussian_noise': [0., 1., 10., 100.],
                                          'l1_penalty': [0.],
                                          'seed': seeds,
                                          'n_players': [50]}))
    full_list_noise = []
    for base_update in base_list_noise:
        update = copy.deepcopy(base_update)
        if base_update['sampling'] in [1, 5, 25]:
            for training_seed in training_seeds:
                seed_update = copy.deepcopy(update)
                seed_update['training_seed'] = training_seed
                full_list_noise.append(seed_update)
        else:
            full_list_noise.append(update)

    # full_list = full_list_5 + full_list_50 + full_list_noise
    full_list = full_list_noise
    # print(len(full_list) / n_jobs)
    Parallel(n_jobs=n_jobs, backend='multiprocessing')(delayed(run_exp)(output_dir,
                                                                        config_updates, _id)
                                                       for _id, config_updates
                                                       in enumerate(full_list))


n_jobs = 50
grid(n_jobs=n_jobs)
