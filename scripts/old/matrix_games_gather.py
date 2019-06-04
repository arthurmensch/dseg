import json
import os
import re
from os.path import expanduser, join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

idx = pd.IndexSlice

output_dirs = [expanduser('~/output/games_rl/matrix_games_big_final'),
               expanduser('~/output/games_rl/matrix_games_big_final_high_noise'),
               expanduser('~/output/games_rl/matrix_games_big_final_50'),
               expanduser('~/output/games_rl/matrix_games_big_final_50_high_noise'),
               ]

output_dirs = [expanduser('~/output/games_rl/matrix_games_final')]

output_dir = expanduser('~/games_rl/paper/figures/matrix_games_clean')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def gather():
    regex = re.compile(r'[0-9]+$')

    records = []

    for this_output_dir in output_dirs:
        for this_dir in filter(regex.match, os.listdir(this_output_dir)):
            this_exp_dir = join(this_output_dir, this_dir)
            this_dir = int(this_dir)
            print(this_exp_dir)
            try:
                config = json.load(
                    open(join(this_exp_dir, 'config.json'), 'r'))
                metrics = json.load(open(join(this_exp_dir, 'metrics.json'), 'r'))
            except (FileNotFoundError, json.decoder.JSONDecodeError):
                print('Skipping exp %i' % this_dir)
                continue
            sampling = config['sampling']
            step_size = config['step_size']
            n_players = config['n_players']

            for g_l, g_u, c in zip(metrics['nash_gap_l_vs_computations']['values'],
                                   metrics['nash_gap_u_vs_computations']['values'],
                                   metrics['nash_gap_l_vs_computations']['steps']):
                records.append({'step_size': step_size,
                                'n_players': n_players,
                                'sampling': str(sampling),
                                'schedule': config['schedule'],
                                'variance_reduction': config['variance_reduction'],
                                'gaussian_noise': config['gaussian_noise'],
                                'skewness': config['skewness'],
                                'l1_penalty': config['l1_penalty'],
                                'nash_gap_u': g_l,
                                'nash_gap_l': g_u,
                                'computations': c})

    records = pd.DataFrame(records)

    records['mean_nash_gap'] = (records['nash_gap_u'] + records['nash_gap_l']) / 2
    records.set_index(['n_players', 'skewness', 'gaussian_noise', 'l1_penalty', 'schedule', 'variance_reduction',
                       'sampling', 'step_size', 'seed'], inplace=True)
    records.sort_index(inplace=True)
    records.to_pickle(join(output_dir, 'records.pkl'))


def select(exp):
    """Were the curation of experiments is done"""
    records = pd.read_pickle(join(output_dir, 'records.pkl'))

    records = records.loc[idx[:, :, :, :, 'constant', :, :, :], :]
    dfs = []
    if exp == '5_players':
        dfs.append(records.loc[idx[5, 0.9, 0, 0, :, True, :, :], :])
        dfs.append(records.loc[idx[5, 0.9, 1e-1, 0, :, True, :, :], :])
        dfs.append(records.loc[idx[5, 0.9, 1e-1, 1e-2, :, False, :, :], :])
        dfs.append(records.loc[idx[5, 1, 1e-1, 1e-2, :, False, :, :], :])
    elif exp == '50_players':
        smooth = records.loc[idx[50, 0.9, 0, 0, :, True, :, :], :]
        noisy = records.loc[idx[50, 0.9, 1e-2, 0., :, True, :, :], :]
        nonsmooth = records.loc[idx[50, 0.9, 1e-2, 1e-2, :, False, :, :], :]
        skew = records.loc[idx[50, 1, 50.0, 0., :, True, :, :], :]
        dfs = [smooth, noisy, nonsmooth, skew]
    elif exp == 'noise':
        dfs = [records.loc[idx[50, 1, [0, 1e-2, 10, 50.], 0., :, True, :, :], :]]
    else:
        raise ValueError
    for df in dfs:
        assert(len(df) > 0), ValueError('Wrong index')
    records = pd.concat(dfs)
    records.to_pickle(join(output_dir, f'records_{exp}.pkl'))


def reduce(exp, selection='mean', gamma=0, tol=1e-3):
    records = pd.read_pickle(join(output_dir, f'records_{exp}.pkl'))

    def score_fn(df):
        computations = df['computations']
        nash_gap = df['mean_nash_gap']
        if selection == 'tol':
            index = np.where(nash_gap < tol)[0]
            if len(index) > 0:
                index = index[0]
            else:
                index = -1
            score = computations[index]
        else:
            weights = np.float_power(gamma, computations[-1] - computations)
            weights /= weights.sum()
            score = np.sum(nash_gap * weights)
        return score

    records['score'] = records.groupby(
        level=['n_players', 'skewness', 'gaussian_noise', 'l1_penalty', 'schedule', 'variance_reduction',
               'sampling', 'step_size']).apply(score_fn)
    idxmin = records['score'].groupby(level=['n_players', 'skewness',
                                             'gaussian_noise', 'l1_penalty', 'sampling']).idxmin()

    best_records = records.loc[idxmin]

    best_records.reset_index(level=['schedule', 'variance_reduction', 'step_size'], inplace=True)

    best_records.to_pickle(join(output_dir, f'best_records_{exp}.pkl'))


def plot_final(exp):
    best_records = pd.read_pickle(join(output_dir, f'best_records_{exp}.pkl'))
    if exp == '5_players':
        samplings = ['alternated', '1', '2', 'all']
        labels_dict = {'1': '1/5 players', '2': '2/5 players', 'all': 'all players',
                       'alternated': '1/5 player (cyclic)'}
    elif exp in ['50_players', 'noise']:
        samplings = ['alternated', '1', '5', '25', 'all']
        labels_dict = {'1': '1/50 players', '5': '5/50 players', '25': '25/50 players',
                       'all': 'all players', 'alternated': '1/50 player (cyclic)'}
    if exp in ['5_players', '50_players']:
        titles = ['smooth\nno noise', 'smooth\nnoise', 'non-smooth\nnoise', 'non-smooth\nnoise\nfully skew game']
    cmap_seq = sns.cubehelix_palette(len(samplings) - 1, start=1.5, rot=0.5, dark=0.3, light=.8, reverse=True)
    cmap_seq_alternated = sns.cubehelix_palette(len(samplings) - 1, start=3, rot=0.5, dark=0.3, light=.8,
                                                reverse=True)

    cmap = {}
    for key, color in zip(samplings[1:], cmap_seq):
        cmap[key] = color
    cmap['alternated'] = cmap_seq_alternated[0]
    scale = 50
    fig, axes = plt.subplots(1, 4, figsize=(397.48499 / scale, 70 / scale))
    plt.subplots_adjust(top=0.96, bottom=.3, left=0.1, right=0.99, wspace=.25)

    labels = []
    handles = []
    for i, (index, best_subrecords) in enumerate(best_records.groupby(['n_players', 'skewness', 'gaussian_noise', 'l1_penalty'])):
        print(index)
        ax = axes[i]
        best_subrecords = best_records.loc[index]
        for sampling in reversed(samplings):
            record = best_subrecords.loc[sampling]
            h, = ax.plot(record['computations'][1:], record['nash_gap_l'][1:], color=cmap[sampling],
                         label=sampling, zorder=2 * i + 1 + 100)
            # ax.fill_between(record['computations'][1:], record['nash_gap_l'][1:],
            #                 record['nash_gap_u'][1:], color=cmap[sampling],
            #                 label=sampling, alpha=0.2, zorder=2 * i + 100)
            if i == 0:
                handles.append(h)
                labels.append(labels_dict[sampling])
            # if exp == '5_players':
            #     if i == 0:
            #         ax.set_ylim(1e-5, 1e-2)
            #     elif i == 1:
            #         ax.set_ylim(1e-2, 1)
            #     elif i == 2:
            #         ax.set_ylim(1e-2, 1)
            #     elif i == 3:
            #         ax.set_ylim(1e-1, 10)

        ax.set_yscale('log')
        ax.grid(axis='y')
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.tick_params(axis='both', which='minor', labelsize=5)
        if exp in ['5_players', '50_players']:
            ax.annotate(titles[i], xy=(1, .9), xycoords='axes fraction', va='top', ha='right', fontsize=8)
        else:
            noise = index[2]
            ax.annotate(rf"$\sigma = {noise:.2f}$", xy=(1, .9), xycoords='axes fraction', va='top', ha='right', fontsize=8)
    axes[0].set_ylabel('Nash gap')
    axes[0].annotate('Computat°', xy=(0, 0), xytext=(-3, -6), textcoords='offset points',
                     xycoords='axes fraction', ha='right', va='top')
    axes[0].legend(handles, labels, ncol=5, frameon=False, bbox_to_anchor=(-.6, -.55), loc='lower left')
    sns.despine(fig, axes)
    filename = join(output_dir, f"convergence_{exp}.pdf")
    plt.savefig(filename)
    filename = join(output_dir, f"convergence_{exp}.png")
    plt.savefig(filename)
    plt.close(fig)


# gather()
for exp in ['5_players', '50_players', 'noise']:
    select(exp)
    reduce(exp)
    plot_final(exp)
