import json
import os
import re
from os.path import expanduser, join
import matplotlib
from matplotlib import rc

matplotlib.rcParams['backend'] = 'pdf'
rc('text', usetex=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import math

idx = pd.IndexSlice


output_dirs = [expanduser('~/output/games_rl/matrix_games_final_2')]

output_dir = expanduser('~/games_rl/paper/figures/matrix_games')


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
                                'gaussian_noise': config['gaussian_noise'],
                                'skewness': config['skewness'],
                                'l1_penalty': config['l1_penalty'],
                                'nash_gap_u': g_u,
                                'nash_gap_l': g_l,
                                'seed': str(config['seed']) + '_' + str(config['training_seed']),
                                'computations': c})

    records = pd.DataFrame(records)

    records['mean_nash_gap'] = (records['nash_gap_u'] + records['nash_gap_l']) / 2
    records.set_index(['n_players', 'skewness', 'gaussian_noise', 'l1_penalty',
                       'sampling', 'seed', 'step_size'], inplace=True)
    records.sort_index(inplace=True)
    records.to_pickle(join(output_dir, 'records.pkl'))


def select(exp):
    """Were the curation of experiments is done"""

    dfs = []
    if exp == '5_players':
        records = pd.read_pickle(join(output_dir, 'records_crunchy6.pkl'))

        dfs.append(records.loc[idx[5, 0.9, 0, 0, :, :, :], :])
        dfs.append(records.loc[idx[5, 0.9, 1, 0, :, :, :], :])
        # dfs.append(records.loc[idx[5, 0.9, 1, 2e-2, :, :, :], :])
        dfs.append(records.loc[idx[5, 1, 1, 2e-2, :, :, :], :])
    elif exp == 'skew':
        records = pd.read_pickle(join(output_dir, 'records_old.pkl'))
        records['seed'] = 0
        records.reset_index(inplace=True)
        records.set_index(['n_players', 'skewness', 'gaussian_noise', 'l1_penalty',
                           'sampling', 'seed', 'step_size'], inplace=True)
        dfs = [records.loc[idx[50, 1., [0, 10, 50.], 0., :, :, :], :]]
    else:
        records = pd.read_pickle(join(output_dir, 'records_drago4.pkl'))
        vr = 'vr' in exp
        if exp == '50_players_auto':
            smooth = records.loc[idx[50, 0.9, 0, 0, :, :, True, :], :]
            noisy = records.loc[idx[50, 0.9, 1, 0., :, :, True, :], :]
            nonsmooth = records.loc[idx[50, 0.9, 1, 2e-2, :, :, False, :], :]
            skew = records.loc[idx[50, 1, 50.0, 2e-2, :, :, False, :], :]
            dfs = [smooth, nonsmooth, skew]
        elif '50_players' in exp:
            smooth = records.loc[idx[50, 0.9, 0, 0, :, :, vr, :], :]
            noisy = records.loc[idx[50, 0.9, 1, 0., :, :, vr, :], :]
            nonsmooth = records.loc[idx[50, 0.9, 1, 2e-2, :, :, vr, :], :]
            skew = records.loc[idx[50, 1, 50.0, 2e-2, :, :, vr, :], :]
            dfs = [smooth, nonsmooth, skew]
        elif 'noise' in exp:
            dfs = [records.loc[idx[50, 0.95, [1, 10, 100.], 0., :, :, vr, :], :]]
    for df in dfs:
        assert (len(df) > 0), ValueError('Wrong index')
    records = pd.concat(dfs)
    if exp not in ['5_players', 'skew']:
        records.reset_index('variance_reduction', drop=True, inplace=True)
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

    scores = records.groupby(
        level=['n_players', 'skewness', 'gaussian_noise', 'l1_penalty', 'sampling', 'seed', 'step_size']).apply(
        score_fn)
    idxmin = scores.groupby(level=['n_players', 'skewness',
                                   'gaussian_noise', 'l1_penalty', 'sampling', 'seed']).idxmin().values.tolist()
    best_records = records.loc[idx[idxmin], :]
    best_records.reset_index(inplace=True)
    best_records.set_index(
        ['n_players', 'skewness', 'gaussian_noise', 'l1_penalty', 'sampling', 'computations', 'seed'],
        inplace=True)
    best_records = best_records.groupby(level=['n_players', 'skewness',
                                               'gaussian_noise', 'l1_penalty', 'sampling', 'computations']).agg(
        ['mean', 'std'])
    best_records.reset_index('computations', inplace=True)
    best_records.to_pickle(join(output_dir, f'best_records_{exp}.pkl'))


def plot_final(exp):
    best_records = pd.read_pickle(join(output_dir, f'best_records_{exp}.pkl'))
    if exp in '5_players':
        samplings = ['alternated', '1', '2', 'all']
        labels_dict = {'1': '1/5 players', '2': '2/5 players', 'all': r'Full extra-gradient (Juditsky \textit{et al.})',
                       'alternated': '1/5 player (cyclic)'}
    else:
        samplings = ['alternated', '1', '5', '25', 'all']
        if exp == 'skew':
            labels_dict = {'1': '1/50 players', '5': '5/50 players', '25': '25/50 players',
                           'all': r'Full extra-gradient (Juditsky \textit{et al.})', 'alternated': '1/50 player (cyclic)'}
        else:
            labels_dict = {'1': r'1/50 players', '5': '5/50 players', '25': '25/50 players',
                           'all': 'Full extra-gradient', 'alternated': '1/50 player (cyclic)'}
    cmap_seq = sns.cubehelix_palette(len(samplings) - 1, start=2, rot=0.5, dark=0.3, light=.8, reverse=True)
    cmap_seq_alternated = sns.cubehelix_palette(len(samplings) - 1, start=.5, rot=0.5, dark=0.3, light=.8,
                                                reverse=True)
    cmap_seq_ref = sns.cubehelix_palette(len(samplings) - 1, start=0.5, rot=0., dark=0.3, light=.7, reverse=True)

    cmap = {}
    for key, color in zip(samplings[1:], cmap_seq):
        cmap[key] = color

    cmap['alternated'] = cmap_seq_alternated[0]
    cmap['all'] = cmap_seq_ref[-1]
    scale = 50
    if exp == '50_players_auto':
        fig, axes = plt.subplots(1, 3, figsize=(397.48499 / scale, 0.7 / .8 * 60 / scale))
        plt.subplots_adjust(top=0.95, bottom=0.2, left=0.1, right=0.99, wspace=.25)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(397.48499 / scale, 60 / scale))
        plt.subplots_adjust(top=0.95, bottom=.3, left=0.1, right=0.99, wspace=.25)

    labels = []
    handles = []
    for i, (index, best_subrecords) in enumerate(best_records.groupby(['n_players', 'skewness', 'gaussian_noise',
                                                                       'l1_penalty'])):
        print(index)
        ax = axes[i]
        offset = 0
        best_subrecords = best_records.loc[index]
        for sampling in reversed(samplings):
            record = best_subrecords.loc[sampling]
            computations = record['computations'].values[offset:] / 1000
            nash_gap = record[('nash_gap_l', 'mean')].values[offset:]
            err = record[('nash_gap_l', 'std')].values[offset:]
            h, = ax.plot(computations, nash_gap, color=cmap[sampling],
                         linestyle='-' if sampling != 'all' else '--', linewidth=2.5, alpha=.8,
                         label=sampling, zorder=2 * i + 1 + 100)
            # ax.fill_between(record['computations'][offset:], record['nash_gap_l'][offset:],
            #                 record['nash_gap_u'][offset:], color=cmap[sampling],
            #                 label=sampling, alpha=0.2, zorder=2 * i + 100)
            # ax.fill_between(computations, nash_gap - err, nash_gap + err,
            #                      color=cmap[sampling], alpha=.3,
            #                      label=sampling, zorder=2 * i + 100)
            if i == 0:
                handles.append(h)
                labels.append(labels_dict[sampling])
            if exp == '50_players_auto':
                if i == 0:
                    ax.set_ylim(1.5e-3, 3e-2)
                elif i == 1:
                    ax.set_ylim(5e-3, 4e-2)
                elif i == 2:
                    ax.set_ylim(13, 60)
            elif exp == '5_players':
                if i == 0:
                    ax.set_ylim(3e-5, 7e-4)
                elif i == 1:
                    ax.set_ylim(2e-4, 7e-3)
                elif i == 2:
                    ax.set_ylim(1e-2, 1.1e-1)
            elif exp == 'noise_vr':
                if i == 0:
                    ax.set_ylim(4e-3, 6e-2)
                elif i == 1:
                    ax.set_ylim(1e-1, 1.1)
                elif i == 2:
                    ax.set_ylim(10, 60)
            elif exp == 'skew':
                if i == 0:
                    ax.set_ylim([1e-2, 10])
                elif i == 1:
                    ax.set_ylim([2.5, 20])
                elif i == 2:
                    ax.set_ylim([13, 100])
            if exp != 'skew':
                ax.set_xlim([2, 45])
            else:
                ax.set_xlim([1, 45])
        ax.set_yscale('log')
        # ax.grid(axis='y')
        # ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.tick_params(axis='both', which='minor', labelsize=6)

        def format(scalar):
            power = math.floor(math.log(scalar, 10))
            scalar /= 10 ** power
            return rf'{scalar:.0f}\cdot 10^{{{power}}}'

        if 'players' in exp:
            l = index[3]
            sigma = index[2]
            skew = index[1]
            kwargs = dict(xy=(.95, .9), textcoords='offset points',
                          xycoords='axes fraction', va='top', ha='right', fontsize=10, zorder=1000)

            offset = 0
            if skew == 1.:
                if exp == '5_players':
                    title = r'fully skew (${\sim}$GAN)'
                else:
                    title = r'fully skew'
                ax.annotate(title, xytext=(0, offset), **kwargs)
                offset -= 10
            title = 'smooth (with VR)' if l == 0 else 'non-smooth'  # rf'$\lambda = {format(l)}$'
            ax.annotate(title, xytext=(0, offset), **kwargs)
            title = 'no noise' if sigma == 0 else rf"$\sigma = {sigma:.0f}$"
            offset -= 10
            ax.annotate(title, xytext=(0, offset), **kwargs)
        else:
            noise = index[2]
            ax.annotate(rf"$\sigma = {noise:.0f}$", xy=(1, .9), xycoords='axes fraction', va='top', ha='right',
                        fontsize=10)
    axes[0].set_ylabel('Nash gap')
    axes[0].annotate(r'Computations', xy=(0, 0), xytext=(-1, -6), textcoords='offset points',
                     xycoords='axes fraction', ha='right', va='top', fontsize=8)
    axes[0].annotate(r'$\times 10^3$', xy=(1, 0), xytext=(-8, -5), textcoords='offset points',
                     xycoords='axes fraction', ha='left', va='center', fontsize=7)
    if exp in ['5_players', 'noise_vr', 'skew']:
        axes[0].legend(handles, labels, ncol=5, columnspacing=.7, frameon=False, bbox_to_anchor=(-.4, -.6),
                       fontsize=10,
                       loc='lower left')
    sns.despine(fig, axes)
    filename = join(output_dir, f"convergence_{exp}.pdf")
    plt.savefig(filename)
    filename = join(output_dir, f"convergence_{exp}.png")
    plt.savefig(filename)
    plt.close(fig)


# gather()
for exp in ['skew', '5_players', '50_players_auto', 'noise_vr']:
    # select(exp)
    # reduce(exp)
    plot_final(exp)
