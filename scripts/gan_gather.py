import glob
import json
import os
from os.path import expanduser, join
from shutil import copyfile
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
from matplotlib import rc
import matplotlib
matplotlib.rcParams['backend'] = 'pdf'
rc('text', usetex=True)
import matplotlib.pyplot as plt


def our_mean_std(df: pd.DataFrame):
    res = {}
    for col, xs in df.iteritems():
        xs = xs.values
        max_length = max(len(x) for x in xs)
        masks = [np.concatenate([np.ones(len(x)), np.zeros(max_length - len(x))]) for x in xs]
        xs = [np.concatenate([np.array(x), np.zeros(max_length - len(x))]) for x in xs]
        xs = np.concatenate([x[None, :] for x in xs], axis=0)
        masks = np.concatenate([mask[None, :] for mask in masks], axis=0)
        count = np.sum(masks, axis=0)
        mean = np.sum(xs, axis=0) / count
        xs -= mean[None, :]
        masks = masks.astype(np.bool)
        xs[~masks] = 0
        res[f'{col}_count'] = count.tolist()
        count = count - 1
        single = count == 0
        count[single] = 1
        std = np.sqrt(np.sum(xs ** 2, axis=0) / count)
        std[single] = 0
        res[f'{col}_mean'] = mean.tolist()
        res[f'{col}_std'] = std.tolist()
    return pd.Series(res)


data_dir = expanduser('~/data/games_rl')
stat_dir = expanduser('~/data/games_rl/stats')
output_dir = expanduser('~/games_rl/paper/figures/gan')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

interesting_metrics = ['is_vs_G_iter', 'fid_vs_G_iter', 'is_vs_time', 'fid_vs_time']


def make_record():
    res = []

    for exp_id in range(1, 19):
        exp_dir = expanduser(f'~/output/games_rl/gan_clean/{exp_id}')
        with open(join(exp_dir, 'metrics.json'), 'r') as f:
            metrics = json.load(f)
        with open(join(exp_dir, 'config.json'), 'r') as f:
            config = json.load(f)
        if config['variance_reduction']:
            continue
        sampling = config['sampling']
        lr = config['lr'] * config['G_lr_factor']
        seed = config['seed']
        scores_dict = {}
        min_length = 1e6
        for metric in interesting_metrics:
            try:
                scores_dict[metric] = metrics[metric]['values']
                scores_dict[f'{metric}_steps'] = metrics[metric]['steps']
                min_length = min(min_length, len(scores_dict[metric]))
            except KeyError:
                min_length = 0
                break
        print(exp_id, sampling, lr, seed, min_length)
        if min_length > 1:
            if lr == 5e-5:
                list_of_files = glob.glob(join(exp_dir, 'artifacts/checkpoint_*'))
                checkpoint = max(list_of_files, key=os.path.getctime)
                print(f'model_{sampling}_{exp_id}.pkl')
                copyfile(checkpoint, join(output_dir, f'model_{sampling}_{exp_id}.pkl'))
            res.append(dict(sampling=sampling, seed=seed, lr=lr, **scores_dict))

    res = pd.DataFrame(res)
    res.set_index(['sampling', 'seed', 'lr'], inplace=True)
    res = res.groupby(['sampling', 'lr']).apply(our_mean_std)

    res.to_pickle(join(output_dir, 'records.pkl'))


def plot(lr=8e-5):
    res = pd.read_pickle(join(output_dir, 'records.pkl'))

    res = res.loc[pd.IndexSlice[['all', 'alternated'], lr], :]
    colors_ours = sns.color_palette('Reds', n_colors=1)
    colors_theirs = sns.color_palette('Blues', n_colors=1)

    colors = {'alternated': colors_ours[0],
              'all': colors_theirs[0],
              }

    for metric in interesting_metrics:
        fig, ax = plt.subplots(1, 1)

        for (sampling, lr), data in res.iterrows():
            steps = np.array(data[f'{metric}_steps_mean'])[:43]
            mean = np.array(data[f'{metric}_mean'])[:43]
            std = np.array(data[f'{metric}_std'])[:43]
            label = 'Doubly-stochastic extra-gradient' if sampling == 'alternated' else "Full extra-gradient (Gidel et al., Juditsky et al.)"
            ax.plot(steps, mean, label=label, linewidth=2, color=colors[sampling],
                    zorder=100 if sampling == 'alternated' else 10)
            ax.fill_between(steps, mean - std, mean + std, color=colors[sampling],
                            alpha=.3, zorder=99 if sampling == 'alternated' else 9)

        if 'time' in metric:
            ax.set_xlabel('Wall-clock time (s)')
        else:
            ax.set_xlabel('Generator updates')

        ax.set_ylabel('Inception score')
        ax.tick_params(axis='x', labelsize=7)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0e'))

        if 'is' in metric:
            ax.set_ylim([4, 8.8])
        else:
            ax.set_ylim([15, 50])

        ax.legend(fontsize=6)
        sns.despine(fig, ax)
        ax.grid(axis='y')
        plt.show()
        plt.savefig(join(output_dir, f'{metric}_{lr:.0e}.pdf'))


def plot_final(lr=8e-5):
    res = pd.read_pickle(join(output_dir, 'records.pkl'))

    res = res.loc[pd.IndexSlice[['all', 'alternated'], lr], :]
    colors_ours = sns.color_palette('Reds', n_colors=1)
    colors_theirs = sns.color_palette('Blues', n_colors=1)

    colors = {'alternated': colors_ours[0],
              'all': colors_theirs[0],
              }
    scale = 50
    fig, axes = plt.subplots(1, 3, figsize=(397.48499 / scale, 70 / scale), gridspec_kw=dict(width_ratios=[1, 1, 1]))
    fig.subplots_adjust(left=0.08, bottom=0.15, top=0.95, right=0.98, wspace=0.25)

    for metric, ax in zip(['is_vs_time', 'fid_vs_time'], axes[[0, 2]]):

        for (sampling, lr), data in res.iterrows():
            steps = np.array(data[f'{metric}_steps_mean'])[:43] / 3600
            mean = np.array(data[f'{metric}_mean'])[:43]
            std = np.array(data[f'{metric}_std'])[:43]
            label = r'\textbf{Doubly-stochastic}' + "\n" + r"\textbf{extra-gradient}" if sampling == 'alternated' else "Full extra-gradient\n" + r"(Gidel \textit{et al.}, Juditsky \textit{et al.})"
            ax.plot(steps, mean, label=label, linewidth=2, color=colors[sampling],
                    zorder=100 if sampling == 'alternated' else 10)
            ax.fill_between(steps, mean - std, mean + std, color=colors[sampling],
                            alpha=.3, zorder=99 if sampling == 'alternated' else 9)
        if 'is' in metric:
            ax.set_ylabel('Inception score')
        else:
            ax.set_ylabel('FID (10k)')
        ax.tick_params(axis='x', labelsize=8)
        # ax.xaxis.set_major_formatter(FormatStrFormatter('%.0e'))

        if 'is' in metric:
            ax.set_ylim([5.98, 8.5])
        else:
            ax.set_ylim([15, 40])

        sns.despine(fig, ax)
        ax.grid(axis='y')
    img = make_images(join(output_dir, 'model_alternated_1.pkl'))
    # img_ref = make_images(join(output_dir, 'model_all_2.pkl'))
    axes[1].imshow(img)
    axes[1].axis('off')
    # axes[3].imshow(img_ref)
    # axes[3].axis('off')
    axes[0].annotate('Time (h)', xy=(0, 0), xytext=(-3, -5), textcoords='offset points',
                     xycoords='axes fraction', ha='right', va='top')
    axes[1].annotate('DSEG generated images', xy=(0.5, 0), xytext=(0, -5), textcoords='offset points',
                     xycoords='axes fraction', ha='center', va='top')
    axes[0].legend(fontsize=9, frameon=False, loc='lower right', bbox_to_anchor=(1.3, .07))
    plt.savefig(join(output_dir, f'gan.pdf'), dpi=1000)


def make_images(checkpoint):
    import torch
    from torchvision.utils import make_grid
    from gamesrl.generative import GeneratorResNet32

    torch.manual_seed(200)
    device = torch.device('cpu')

    ngf = 128
    nz = 128
    nc = 3

    net_G = GeneratorResNet32(n_in=nz, n_out=nc, num_filters=ngf)

    with open(checkpoint, 'rb') as f:
        state_dict = torch.load(checkpoint, map_location=lambda storage, location: storage)
    net_G.load_state_dict(state_dict['avg_net_G'])
    net_G.to(device)

    noise = torch.randn(28, nz, 1, 1, device=device)
    images = net_G(noise).detach()
    grid = make_grid(images, nrow=7, padding=2, pad_value=1,
                     normalize=False, range=None, scale_each=False).numpy()
    grid += 1
    grid /= 2
    # grid = grid.add(1).mul_(255 / 2).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
    return grid.transpose(1, 2, 0)


# make_record()
# for lr in [5e-5, 8e-5]:
#     plot(lr=lr)
plot_final(5e-5)
