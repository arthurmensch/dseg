import os
from os.path import expanduser, join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from joblib import Parallel, delayed

from gamesrl.numpy.games import make_positive_matrix

output_dir = expanduser('~/games_rl/paper/figures/radius')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def keep_player(player: int, n_actions: int, A: np.ndarray):
    """Compute M_i A"""
    Ai = np.zeros_like(A)
    Ai[((player - 1) * n_actions):(player * n_actions), :] = A[((player - 1) * n_actions):(player * n_actions), :]
    return Ai


def get_spectral_radius(B, step_size, method='random'):
    B = B * step_size
    if method == 'random':
        Id = np.identity(B.shape[0])
        sum_4 = (4 * Id - 2 * B + np.matmul(B, B)) / 4
        A = np.matmul(sum_4, sum_4)
    elif method == 'alternated':
        B1 = keep_player(1, n_actions, B)
        B2 = keep_player(2, n_actions, B)
        Id = np.identity(n_players * n_actions)

        X12 = Id - B1 + B1 @ B2
        X21 = Id - B2 + B2 @ B1
        A = np.matmul(X12, X21)
    else:
        Id = np.identity(B.shape[0])
        A = Id - B + np.matmul(B, B)
    return np.max(np.abs(scipy.linalg.eigvals(A)))


def make_distribution():
    n_actions = 3
    n_samples = 1000
    seeds = np.random.randint(1000000, size=n_samples)

    step_sizes = np.logspace(-3, 0, 32)
    methods = ['random', 'alternated', 'full']
    res = []
    for skewness in np.linspace(0., 1., 5):
        for conditioning in [0., 0.01, .1, .5]:
            for seed in seeds:
                n_players = 2
                matrix = make_positive_matrix(n_players, n_actions, skewness=skewness, conditioning=conditioning,
                                              seed=seed).reshape(n_players * n_actions, n_players * n_actions)
                radii = Parallel(n_jobs=4)(delayed(get_spectral_radius)(matrix, step_size, method)
                                           for step_size in step_sizes for method in methods)
                for step_size in step_sizes:
                    for method in methods:
                        radius = radii[0]
                        radii = radii[1:]
                        res.append(dict(seed=seed, skewness=skewness, conditioning=conditioning, method=method,
                                        step_size=step_size, radius=radius))
    res = pd.DataFrame(res)
    res.set_index(['skewness', 'conditioning', 'method', 'seed', 'step_size'], inplace=True)
    index = res['radius'].groupby(['skewness', 'conditioning', 'method', 'seed']).idxmin()
    res = res.loc[index]
    res.to_pickle(join(output_dir, 'radius.pkl'))


def plot():
    idx = pd.IndexSlice
    res = pd.read_pickle(join(output_dir, 'radius.pkl'))
    res = res.loc[idx[np.linspace(0., 1., 5)[1:], :, :], :]
    res.reset_index(inplace=True)
    res['method'] = res['method'].replace({'full': 'All players', 'alternated': 'Cyclic', 'random': 'Random'})
    res.rename(columns={'conditioning': r'$\mu$', 'method': 'Sampling', 'skewness': r'$\alpha$',
                        'radius': 'Alg. spectral radius'}, inplace=True)
    g = sns.FacetGrid(res, col=r'$\alpha$', row=r'$\mu$', hue='Sampling', height=1.5, aspect=1.6)

    g.map(sns.violinplot, 'Alg. spectral radius', 'Sampling', color=".8",
          order=['Cyclic', 'Random', 'All players'])
    g.map(sns.stripplot, 'Alg. spectral radius', 'Sampling',
          order=['Cyclic', 'Random', 'All players'])
    g.axes[0, 0].set_title(r'Conditioning $\mu = 0.0$ | Skewness $\alpha = 0.25$', fontsize=10)
    plt.savefig(join(output_dir, 'radius.pdf'))

plot()
